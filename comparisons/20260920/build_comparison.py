"""Reproducible matched-campaign tables from preserved evaluation summaries."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
HANDOFF=HERE.parents[1]
sources={
    'previous_holdout':HANDOFF/'scaling-eval-20260916/summary-all5-20260917.json',
    'previous_external':HERE.parent/'dspark-scaling-20260917/evidence/external-benchmark-summary.json',
    'new_final':HERE/'evidence/new-final-summary.json',
    'stable_curve':HANDOFF/'shared-prefill-training-20260918/evidence/stable-curve-audit-20260920/validated-summary.json',
    'curve_audit':HANDOFF/'shared-prefill-training-20260918/evidence/stable-curve-audit-20260920/audit.json'}
sources={k:p if p.exists() else HERE/'evidence'/f'{k}.json' for k,p in sources.items()}
data={k:json.loads(p.read_text()) for k,p in sources.items()}
sets=['opencode_fresh5k_scaling_20260910','bigcodebench_instruct_v0.1.4','livecodebench_codegen_release_v6']
titles=['OpenCode 5K','BigCodeBench','LiveCodeBench']
cases=[('new','0p125b','0.125B',114170625,1,'Cosine'),
       ('new','0p25b','0.25B',202160001,2,'Cosine'),
       ('previous','0p5b','0.5B',530429057,3,'Cosine'),
       ('previous','1b','1B',915384833,4,'Cosine'),
       ('previous','2b','2B',1876167169,5,'Cosine'),
       ('previous','4b','4B',3926503681,7,'Cosine'),
       ('new','4b-wsd-stable','4B',3926503681,7,'WSD stable (no decay)'),
       ('new','4b-wsd-cooldown','4B',3926503681,7,'WSD + cooldown'),
       ('previous','8b','8B',8302654593,9,'Cosine')]
lookup={('previous',d['model_size_label'],d['dataset_name']):d for d in data['previous_holdout']+data['previous_external']}
lookup.update({('new',d['model'],d['dataset_name']):d for d in data['new_final']})
fields=['sample_size','validation_ids_sha256','chat_template','temperature','top_p','top_k','min_p',
        'seed_namespace','thinking_budget','answer_budget','concurrency','generation_policy','checkpoint_step']
for dataset,n in zip(sets,[5000,1140,1055]):
    reference=lookup['previous','4b',dataset]
    assert reference['sample_size']==n
    for campaign,key,*_ in cases:
        row=lookup[campaign,key,dataset]
        assert all(row[k]==reference[k] for k in fields),(campaign,key,dataset)

out=['# DSpark: previous and new results side by side — September 20, 2026','',
     'Matched 1.210B-token proportional-family campaign. Target: **Qwen3-235B-A22B-Thinking-2507**. '
     'All final models trained on the same 549,626-row corpus for one epoch / 68,703 updates '
     '(549,624 rows consumed at global batch 8). Parameter counts below exclude frozen target embeddings/output head.','',
     'Evaluation: same fresh OpenCode 5,000, BigCodeBench-Instruct v0.1.4 1,140, and LiveCodeBench release v6 1,055 prompts. '
     'Full prompts; Qwen3-thinking-nosystem; existing two-pass 2,048-thinking-phase / nominal 1,024-answer budget; '
     'T=0.6, top-p=0.95, top-k=20; same per-ID seed policy; TP8 BF16, concurrency 32, radix cache disabled, draft block size 7. '
     'Input-ID hashes and all recorded evaluation settings were checked for equality.','',
     '## Final acceptance length','',
     'Acceptance length τ = completion tokens / verification calls, **including bonus**, across both phases. '
     'This is not correctness/pass@1.','',
     '| Nominal model | Actual params | K=L | LR schedule | Campaign | OpenCode 5K | BigCodeBench | LiveCodeBench |',
     '|---|---:|---:|---|---|---:|---:|---:|']
csvrows=[]
for campaign,key,label,params,depth,schedule in cases:
    rows=[lookup[campaign,key,d] for d in sets]
    acc=[d['spec_accept_length_including_bonus'] for d in rows]
    out.append(f'| {label} | {params/1e9:.3f}B | {depth} | {schedule} | {campaign} | '+' | '.join(f'{v:.3f}' for v in acc)+' |')
    record=dict(model=key,campaign=campaign,parameters=params,layers=depth,schedule=schedule)
    for name,d in zip(['holdout','bigcodebench','livecodebench'],rows):
        record[name+'_acceptance']=d['spec_accept_length_including_bonus']
        record[name+'_tokens_per_s']=d['completion_tokens_per_s']
        record[name+'_truncated']=d['answer_truncated_count']
    csvrows.append(record)

out+=['','### What changed for 4B','',
      '| Comparison (acceptance delta) | OpenCode 5K | BigCodeBench | LiveCodeBench |',
      '|---|---:|---:|---:|']
for name,left,right in [('Cooldown − stable',('new','4b-wsd-cooldown'),('new','4b-wsd-stable')),
                       ('Cooldown − previous cosine',('new','4b-wsd-cooldown'),('previous','4b')),
                       ('Previous 8B − 4B cooldown',('previous','8b'),('new','4b-wsd-cooldown'))]:
    diffs=[lookup[left+(d,)]['spec_accept_length_including_bonus']-lookup[right+(d,)]['spec_accept_length_including_bonus'] for d in sets]
    out.append('| '+name+' | '+' | '.join(f'{x:+.3f}' for x in diffs)+' |')
out+=['','The cooldown branch starts from stable step 61,833 and decays LR over the same final 6,870 updates. '
      'Both 4B branch endpoints have equal data exposure; computing both branches adds 10% extra 4B training work to the campaign. '
      'The old cosine run used a different training execution layout; comparisons against it are not a perfectly isolated LR-only ablation. '
      'All are single runs, without seed-repeat confidence intervals.','',
      '## 4B stable trajectory — without cooldown','',
      '| Training fraction | Step | Traces consumed | OpenCode 5K τ |',
      '|---|---:|---:|---:|']
for i,d in enumerate(data['stable_curve'],1):
    out.append(f'| {i*10}% | {d["checkpoint_step"]:,} | {d["training_traces_seen"]:,} | {d["spec_accept_length_including_bonus"]:.3f} |')
out+=['','The curve rises throughout, with diminishing gains: 80%→100% adds about 0.093 tokens. '
      'This measures the unannealed trajectory, not optimal cooled performance at each budget, and does not establish saturation. '
      'Earlier checkpoints were evaluated only on the 5K holdout; no external checkpoint curves are implied.','',
      '## Observed end-to-end output throughput (tokens/s)','',
      '**These are the padded TP8 implementations, not native-model speedups.** '
      'Runs occurred at different times/on different pods, so do not interpret small differences as statistically reliable speed advantages. '
      'No new target-only baseline was run; no speedup ratio is claimed.','',
      '| Model / schedule | OpenCode 5K | BigCodeBench | LiveCodeBench |',
      '|---|---:|---:|---:|']
for campaign,key,label,params,depth,schedule in cases:
    out.append(f'| {label} / {schedule} | '+' | '.join(f'{lookup[campaign,key,d]["completion_tokens_per_s"]:,.1f}' for d in sets)+' |')
out+=['','## Output-budget caveat','',
      '| Model / schedule | Holdout truncated / 5,000 | BCB truncated / 1,140 | LCB truncated / 1,055 |',
      '|---|---:|---:|---:|']
for campaign,key,label,params,depth,schedule in cases:
    out.append(f'| {label} / {schedule} | '+' | '.join(str(lookup[campaign,key,d]['answer_truncated_count']) for d in sets)+' |')
out+=['','All responses, including truncated ones, remain in the headline acceptance scores. '
      'Around one-third of LiveCodeBench answers hit the answer limit. No executable correctness grading was done. '
      'Earlier DSpark/EAGLE experiments with different corpora, holdouts, architectures or budgets are deliberately not merged into this matched table.','',
      '## Audit and source artifacts','',
      'The stable-curve queue accidentally duplicated steps 6,870, 48,092 and 61,833 because cross-node file locks did not exclude workers. '
      'Retained raw files were revalidated (5,000 unique prompts per checkpoint; no summary/raw mismatch). '
      'Earlier duplicate full outputs may have been overwritten; scalar logs survive. No best-of-repeat selection was applied. '
      'The queue now uses exclusive directory creation, tested across all three nodes.','',
      '- [Machine-readable comparison](comparison.json) and [CSV](comparison.csv).',
      '- [Validated stable curve](evidence/stable_curve.json) and [duplicate-work audit](evidence/curve_audit.json).',
      '- [Source checksums and paths](evidence/source-manifest.json).',
      '- [Original scaling writeup](../dspark-scaling-20260917/dspark-scaling-writeup.md).','']
(HERE/'comparison.md').write_text('\n'.join(out))
(HERE/'comparison.json').write_text(json.dumps(csvrows,indent=2)+'\n')
with (HERE/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
for k,value in data.items():
    (HERE/'evidence'/f'{k}.json').write_text(json.dumps(value,indent=2)+'\n')
(HERE/'evidence/source-manifest.json').write_text(json.dumps({k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in sources.items()},indent=2)+'\n')
print(HERE/'comparison.md')
