"""Read immutable campaign artifacts; write only this report's evidence directory."""
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/workspace/specforge/dspark-proportional-1p2b-20260912')
CORPUS = Path('/data/opencode-qwen3-235b-thinking-scaling-1p2b-20260910')
GEN = Path('/workspace/specforge/qwen3-235b-thinking-scaling-1p2b-20260910')
OUT = ROOT / 'reports/dspark-scaling-20260917/evidence'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(GEN/'scripts'))
from prune_repetitive_traces import internal_repetition, normalize

def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')

for name in ['corpus-token-audit.json', 'prepared.json', 'worker0.pruning.json',
             'worker1.pruning.json', 'quality-audit-20260912.json']:
    d = json.loads((CORPUS/'manifests'/name).read_text())
    save(name, {k:v for k,v in d.items() if k not in ['examples', 'shards']})
save('architecture-plan.json', json.loads((ROOT/'manifests/architecture-plan.json').read_text()))
evalroot = ROOT/'eval/fresh5k-20260916'
save('data-preflight.json', json.loads((evalroot/'data-preflight.json').read_text()))
prompts = {r['id']: r for r in map(json.loads, (CORPUS/'data/fresh_holdout.source.jsonl').open())}
results = {}
example_ids = sorted(prompts, key=lambda x: hashlib.sha256((x+':writeup-example-v1').encode()).digest())[:2]
examples = []
for size in ['0p5b','1b','2b','4b','8b']:
    source = evalroot/size/'metrics.json'
    data = json.loads(source.read_text())
    counts = Counter(); hist = [0]*8; meta_keys = set(); seen_r = set(); seen_a = set()
    flagged_meta = Counter(); flagged_examples = []; unflagged_meta = Counter()
    ids = [r['id'] for r in data['records']]
    assert len(ids) == len(set(ids)) == 5000 and set(ids) == set(prompts)
    for row in data['records']:
        for field in ['phase1_meta_info','phase2_meta_info']:
            m = row.get(field)
            if not m: continue
            meta_keys.update(m)
            if 'spec_correct_drafts_histogram' not in m:
                assert m.get('spec_verify_ct', 0) == 0, (size, row['id'], m)
                assert m.get('spec_num_correct_drafts', 0) == 0
                counts['phases_without_verification'] += 1
                continue
            h = m['spec_correct_drafts_histogram']
            assert sum(h) == m['spec_verify_ct']
            assert sum(i*n for i,n in enumerate(h)) == m['spec_num_correct_drafts']
            assert m['spec_num_proposed_drafts'] == 7*m['spec_verify_ct']
            assert len(h) <= 8
            for i,n in enumerate(h): hist[i] += n
        text = row['generated_text']
        reasoning, sep, answer = text.partition('</think>')
        answer = answer.replace('<|im_end|>', '').strip()
        counts['missing_think_close'] += not bool(sep)
        counts['empty_reasoning'] += not bool(reasoning.strip())
        counts['empty_answer'] += not bool(answer)
        counts['answer_truncated'] += bool(row['answer_truncated'])
        counts['forced_early_stop'] += bool(row['forced_early_stop'])
        counts['answer_odd_code_fence'] += answer.count('```') % 2 != 0
        counts['answer_over_budget_in_phase1'] += row['already_answer_tokens_after_phase1'] > 1024
        rr, rs = internal_repetition(reasoning, row['forced_early_stop'])
        ar, ast = internal_repetition(answer, False)
        counts['reasoning_repetition_flags'] += bool(rr)
        counts['answer_repetition_flags'] += bool(ar)
        counts['any_repetition_flag'] += bool(rr or ar)
        for label, txt, seen in [('reasoning',reasoning,seen_r),('answer',answer,seen_a)]:
            h = hashlib.sha256(normalize(txt).encode()).hexdigest()
            counts['duplicate_'+label+'_extra_rows'] += h in seen
            seen.add(h)
        dest = flagged_meta if rr or ar else unflagged_meta
        for key,value in row['meta_info'].items(): dest[key] += value
        if (rr or ar) and len(flagged_examples)<3:
            flagged_examples.append(dict(id=row['id'],reasoning_flags=rr,answer_flags=ar,
                reasoning_stats=rs,answer_stats=ast,prompt=prompts[row['id']]['conversations'][0]['content'],
                generated_text=text))
        if row['id'] in example_ids:
            examples.append(dict(model=size,id=row['id'],prompt=prompts[row['id']]['conversations'][0]['content'],
                generated_text=text,forced_early_stop=row['forced_early_stop'],answer_truncated=row['answer_truncated']))
    verify=sum(hist); correct=sum(i*n for i,n in enumerate(hist))
    assert verify==data['spec_verify_ct'] and correct==data['spec_num_correct_drafts']
    survival=[sum(hist[j:])/verify for j in range(1,8)]
    conditional=[survival[j]/(survival[j-1] if j else 1) for j in range(7)]
    result={k:v for k,v in data.items() if k!='records'}
    result.update(quality_counts=dict(counts),histogram_no_bonus=hist,
        position_prefix_survival=survival,position_conditional_acceptance=conditional,
        histogram_checks_passed=True,phase_metadata_keys=sorted(meta_keys),
        mean_generated_tokens=data['completion_tokens']/5000,
        unflagged_acceptance=unflagged_meta['completion_tokens']/unflagged_meta['spec_verify_ct'],
        flagged_acceptance=flagged_meta['completion_tokens']/flagged_meta['spec_verify_ct'] if flagged_meta['spec_verify_ct'] else None,
        flagged_generated_token_share=flagged_meta['completion_tokens']/data['completion_tokens'],
        source=str(source))
    results[size]=result
    save(size+'-flagged-examples.json',flagged_examples)
    print(size, 'eval audited', counts['any_repetition_flag'], 'flagged rows',flush=True)
save('evaluation-audit.json',results)
save('holdout-examples.json',examples)

# Deterministic illustrative examples: first natural and forced row in each source component.
chosen={}
with (CORPUS/'data/opencode_scaling.combined.cleaned.jsonl').open() as f:
    for index,line in enumerate(f,1):
        group='prior' if index<=196816 else ('worker0' if index<=373237 else 'worker1')
        if (group,False) in chosen and (group,True) in chosen:
            if len(chosen)==6: break
            continue
        row=json.loads(line); forced=bool(row['trace_generation']['forced_early_stop'])
        chosen.setdefault((group,forced),dict(line_number=index,component=group,row=row))
save('training-examples.json',list(chosen.values()))
with (OUT/'training-examples.jsonl').open('w') as f:
    for item in chosen.values(): f.write(json.dumps(item['row'],ensure_ascii=False)+'\n')
print('training samples collected',flush=True)

# Authoritative histories: original prefix + final selected resumed suffix only.
log_sources={
 '0p5b': [('attempts/0p5b-migrate-fa9c-20260916/previous-pod.training.log',0,19000),('logs/0p5b.training.log',19000,68703)],
 '1b': [('logs/1b.training.log',0,68703)],
 '2b': [('logs/2b.training.log',0,68703)],
 '4b': [('attempts/capture64-4b-20260914/original.training.log',0,47000),('logs/4b.training.log',47000,68703)],
 '8b': [('attempts/capture96-8b-20260914/original.training.log',0,42000),('logs/8b.training.log',42000,68703)],
}
loss_history={}
keys=['loss','ce_loss','l1_loss','confidence_loss','lr','tau_probabilistic']
for size,sources in log_sources.items():
    steps={}; provenance=[]
    for rel,lo,hi in sources:
        p=ROOT/rel; seen=set()
        with p.open() as f:
            for line in f:
                match=re.match(r'^step (\d+):',line)
                if not match:continue
                step=int(match[1])
                if step in seen or not lo<step<=hi:continue
                values={}
                for key in keys:
                    m=re.search(r"'train/"+key+r"': ([\deE+.\-]+)",line)
                    if m:values[key]=float(m[1])
                if 'loss' in values:steps[step]=values;seen.add(step)
        provenance.append(dict(path=str(p),min_step=min(seen) if seen else None,max_step=max(seen) if seen else None,points=len(seen)))
    points=[dict(step=s,**v) for s,v in sorted(steps.items())]
    with (OUT/(size+'-loss-history.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=['step']+keys);w.writeheader();w.writerows(points)
    windows=[]
    for endpoint in [1000,10000,30000,50000,68703]:
        subset=[v for s,v in steps.items() if endpoint-1000<s<=endpoint]
        windows.append(dict(through_step=endpoint,logged_batches=len(subset),**{k:statistics.fmean(v[k] for v in subset if k in v) for k in keys}) if subset else dict(through_step=endpoint,logged_batches=0))
    loss_history[size]=dict(sources=provenance,windows=windows,final_logged=points[-1])
    print(size,'loss history',len(points),flush=True)
save('loss-history-summary.json',loss_history)
print('REPORT_EVIDENCE_COMPLETE',flush=True)
