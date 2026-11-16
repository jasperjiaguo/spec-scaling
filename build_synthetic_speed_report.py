"""Audit preserved speed summaries and print the complete (possibly partial) grid."""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE/'evidence/synthetic-speed-20260922'
NFS='/workspace/specforge/dspark-synthetic-speed-20260922'
ORDER=['0p125b','0p25b','0p5b','1b','2b','4b','8b','16b','32b']
LABEL=['0.125B','0.25B','0.5B','1B','2B','4B','8B','16B','32B']
TAUS=[x/4 for x in range(6,21)]


def check_workload(r):
    assert (r['requests'],r['concurrency'],r['input_tokens'],r['output_tokens_each'])==(256,32,1,8192)
    assert r['total_output_tokens']==256*8192 and r['all_exact_token_counts']
    assert math.isclose(r['output_tokens_per_second'],256*8192/r['elapsed_seconds'],rel_tol=1e-12)


def main():
    baselines={p.parent.parent.name:json.loads(p.read_text()) for p in (ROOT/'workers').glob('*/baseline/metrics.json')}
    assert len(baselines)==3
    for b in baselines.values():
        check_workload(b)
        assert b['spec_verify_ct']==0
    models={r['name']:r for r in json.loads((ROOT/'models.json').read_text())}
    rows=[]
    sources={}
    for p in sorted((ROOT/'results').glob('*/tau-*/metrics.json')):
        r=json.loads(p.read_text())
        check_workload(r)
        name=p.parent.parent.name
        tau=float(p.parent.name.removeprefix('tau-'))
        assert name in ORDER and tau in TAUS and r['synthetic_tau']==tau
        assert r['model']==models[name]
        hist=r['correct_drafts_histogram']
        assert sum(hist)==r['spec_verify_ct']>0
        observed=1+sum(i*c for i,c in enumerate(hist))/sum(hist)
        assert math.isclose(observed,r['observed_tau_including_bonus'],abs_tol=1e-12)
        assert abs(observed-tau)<.025
        assert all(c==0 or i in (math.floor(tau-1),math.ceil(tau-1)) for i,c in enumerate(hist))
        speedup=r['output_tokens_per_second']/baselines[r['worker']]['output_tokens_per_second']
        assert math.isclose(speedup,r['speedup'],rel_tol=1e-12)
        rows.append(dict(model=name,tau=tau,observed_tau=observed,worker=r['worker'],
            tokens_per_second=r['output_tokens_per_second'],speedup=speedup,
            elapsed_seconds=r['elapsed_seconds'],raw_request_records=r['request_records'],
            metrics_path=NFS+'/'+str(p.relative_to(ROOT))))
        sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    lookup={(r['model'],r['tau']):r for r in rows}
    assert len(lookup)==len(rows)
    missing=[dict(model=m,tau=t) for m in ORDER for t in TAUS if (m,t) not in lookup]
    workers={p.parent.name:json.loads(p.read_text()) for p in (ROOT/'workers').glob('*/status.json')}
    artifact=dict(completed_points=len(rows),expected_points=135,missing=missing,results=rows,
        baselines=baselines,workers=workers,source_sha256=sources,
        max_abs_tau_error=max(abs(r['tau']-r['observed_tau']) for r in rows),
        audit='Recomputed throughput/speedup and acceptance histograms; checked saved exact-token-count flags. Full per-request records are preserved at the listed NFS paths.')
    (ROOT/'summary-audit.json').write_text(json.dumps(artifact,indent=2)+'\n')
    with (ROOT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    complete_models=[m for m in ORDER if all((m,t) in lookup for t in TAUS)]
    complete=not missing and len(complete_models)==9
    if complete:
        assert all((ROOT/'results'/m/'COMPLETED.json').exists() for m in ORDER)
    heading='Complete' if complete else 'Incomplete'
    missing_text=('All nine models have all 15 acceptance points.' if complete else
                  'Still missing: '+', '.join(LABEL[ORDER.index(m)] for m in ORDER if m not in complete_models)+'.')
    md=['## 11. Synthetic acceptance / speed sweep — '+datetime.now(timezone.utc).strftime('%B %d, %Y')+' snapshot','',
        f'**{heading}: {len(rows)}/135 points.** {missing_text} '
        'The original workers for 16B and 0.125B stopped at a model transition with `GPU occupied; refusing to collide`. '
        'Recovery added bounded CUDA-release waiting and owned-child cleanup before the next server launch. '
        'The recovery runs preserve the original workload, baselines and completed points. '
        + ('All required points are now present and audited.' if complete else 'This is a partial result; do not treat it as a completed nine-model experiment.'),'',
        '### Workload and interpretation','',
        'Target: **Qwen3-235B-A22B-Thinking-2507**, BF16 TP8, no EP, eight H200s per server. '
        'Each point issues **256 requests**, concurrency/server maximum **32**, exactly **one raw input token** (`[100]`) and **8,192 output tokens per request**. '
        'No chat template, reasoning cap, stop strings or EOS termination; temperature 0.6, top-p 0.95, top-k 20. '
        'Radix cache is disabled; context length 16,384, chunked prefill 8,192, KV token cap 270,336, decode CUDA graph maximum batch 32. '
        'The runner excludes loading and explicit warmup from timed request wall-clock. '
        'It measures all 2,097,152 output tokens divided by elapsed wall time; speedup divides that throughput by the **same-pod target-only baseline**.','',
        'Draft block size is 7; τ includes the bonus token. All 15 setpoints from 1.50 through 5.00 use 0.25 increments. '
        'The isolated engine patch overrides accepted-prefix length with a deterministic, balanced floor/ceiling mixture rotated across rows and steps, while retaining real draft and target forward computation. '
        'Static verify-all is used. This is not a real quality/acceptance test and does not reproduce an arbitrary acceptance distribution. '
        'Forced draft tokens also change future context and target MoE routing. Therefore these are synthetic workload-specific speedups, not code-quality or real coding-workload speed predictions.','',
        'The 16B/32B configurations are untrained random-weight drafters, not trained checkpoints. '
        'Existing trained exports use zero-padded attention heads for TP8, which adds compute. '
        'Native parameter labels below exclude frozen shared target embeddings/output head; executed padded dimensions are recorded explicitly. '
        'No repeated-run error bars or post-sweep target-only drift measurements are available; small cross-size differences should not be treated as significant.','',
        '### Matched target-only baselines','',
        '| Worker | Output tok/s | Wall seconds | Used for completed models |','|---|---:|---:|---|']
    for worker,b in sorted(baselines.items()):
        used=[LABEL[ORDER.index(m)] for m in ORDER if any(r['model']==m and r['worker']==worker for r in rows)]
        md.append(f"| {worker.upper()} | {b['output_tokens_per_second']:,.2f} | {b['elapsed_seconds']:,.2f} | {', '.join(used) or 'None yet'} |")
    md+=['','### Draft configurations','',
         '| Nominal | Native parameters | Padded parameters | K=L | d_model | d_ff | Native Q/KV heads | Executed Q/KV heads | Weights |',
         '|---|---:|---:|---:|---:|---:|---|---|---|']
    for name,label in zip(ORDER,LABEL):
        m=models[name]
        md.append(f"| {label} | {m['native_parameters']:,} | {m['padded_parameters']:,} | {m['L']} | {m['d_model']:,} | {m['d_ff']:,} | {'/'.join(map(str,m['native_heads']))} | {'/'.join(map(str,m['executed_heads']))} | {'Untrained random' if m['dummy'] else 'Trained cosine final'} |")
    for title,key,fmt in [('Output throughput (tokens/s)','tokens_per_second',',.1f'),('Speedup versus matched target-only','speedup','.3f')]:
        md+=['',f'### {title}','', '“—” means missing, not zero.','',
             '| τ incl. bonus | '+' | '.join(LABEL)+' |', '|---:'+'|---:'*len(ORDER)+'|']
        for tau in TAUS:
            cells=[format(lookup[m,tau][key],fmt)+('×' if key=='speedup' else '') if (m,tau) in lookup else '—' for m in ORDER]
            md.append(f'| {tau:.2f} | '+' | '.join(cells)+' |')
    md+=['','### Audit and preserved artifacts','',
        f"All {len(rows)} completed summaries passed workload-shape checks and recomputation of throughput, same-pod speedup, and histogram-weighted acceptance. "
        f"Maximum absolute difference between requested and observed τ is **{artifact['max_abs_tau_error']:.8f}**. "
        'The benchmark client saved per-request exact-length checks and the raw metadata; the report audit checks the saved summaries and histograms.','',
        '- [Full-precision results and audit](evidence/synthetic-speed-20260922/summary-audit.json), [CSV](evidence/synthetic-speed-20260922/results.csv), and [report builder](build_synthetic_speed_report.py).',
        '- Per-point metrics, server commands, completion/failure markers and baseline summaries are mirrored under `evidence/synthetic-speed-20260922/`.',
        f'- Original per-request records, full server logs and five-minute recorder snapshots remain under `{NFS}/`; exact raw paths are in the audit JSON.',
        '- Training-loss data for all 18 validated trained checkpoints are preserved in [section 10](#training-loss-for-every-validated-checkpoint) and [full-precision evidence](evidence/validated-checkpoint-losses.json). Untrained 16B/32B training losses remain N/A.','']
    print('\n'.join(md))


if __name__=='__main__': main()
