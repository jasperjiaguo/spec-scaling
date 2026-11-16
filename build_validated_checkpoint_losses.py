"""Join preserved training logs to every validated checkpoint in this scaling family.

Writes reproducible JSON/CSV evidence; prints a Markdown table for the report.
No model loading or training/evaluation is performed.
"""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence'
COMPARISON = HERE.parent / 'dspark-comparison-20260920'
NEW_ROOT = '/workspace/specforge/dspark-shared-prefill-wsd-20260918/dual-prefill-224'
OLD_ROOT = '/workspace/specforge/dspark-proportional-1p2b-20260912'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history(model):
    if model in ('0p5b', '1b', '2b', '4b', '8b'):
        path = EVIDENCE / f'{model}-loss-history.csv'
        with path.open() as f:
            rows = [{k: int(v) if k == 'step' else float(v) for k, v in r.items()}
                    for r in csv.DictReader(f)]
        sources = json.loads((EVIDENCE/'loss-history-summary.json').read_text())[model]['sources']
    else:
        filename = model+'.jsonl' if model.startswith('4b-wsd-') else model+'-stable.jsonl'
        path = EVIDENCE/'training-loss-sources'/filename
        rows = []
        for line in path.read_text().splitlines():
            obj = json.loads(line)
            if obj.get('stage') == 'training':
                r = obj['latest']
                assert r['step'] == obj['step']
                rows.append({k:r[k] for k in ('step','loss','lr')})
        sources = [dict(path=NEW_ROOT+'/logs/'+filename)]
        if model == '4b-wsd-stable':
            prefix_path = EVIDENCE/'training-loss-sources/4b-wsd-original-stable.jsonl'
            prefix = [json.loads(line) for line in prefix_path.read_text().splitlines()]
            prefix = [{k:r['latest'][k] for k in ('step','loss','lr')}
                      for r in prefix if r.get('stage')=='training' and r['step'] <= 6000]
            rows = prefix + rows
            sources.insert(0,dict(path=NEW_ROOT.removesuffix('/dual-prefill-224')+'/logs/4b-wsd-stable.jsonl',
                through_step=6000,local_evidence=str(prefix_path.relative_to(HERE)),sha256=digest(prefix_path)))
    # Retain the continued trajectory's last record at a duplicated resume step.
    by_step = {}
    duplicates = 0
    for row in rows:
        duplicates += row['step'] in by_step
        by_step[row['step']] = row
    ordered = [by_step[s] for s in sorted(by_step)]
    assert all(math.isfinite(r['loss']) for r in ordered)
    return ordered, dict(local_evidence=str(path.relative_to(HERE)), sha256=digest(path),
                         source_logs=sources,duplicate_logged_steps=duplicates)


def main():
    final_path = COMPARISON/'comparison.json'
    curve_path = COMPARISON/'evidence/stable_curve.json'
    finals = json.loads(final_path.read_text())
    curve = json.loads(curve_path.read_text())
    final_by_model = {r['model']:r for r in finals}
    assert len(final_by_model) == 9 and len(curve) == 10
    cases = [(m, 68703, final_by_model[m]['holdout_acceptance'])
             for m in ('0p125b','0p25b','0p5b','1b','2b','4b','8b')]
    cases += [('4b-wsd-stable',r['checkpoint_step'],r['spec_accept_length_including_bonus'])
              for r in sorted(curve,key=lambda x:x['checkpoint_step'])]
    cases += [('4b-wsd-cooldown',68703,final_by_model['4b-wsd-cooldown']['holdout_acceptance'])]
    assert len(cases) == len(set((m,s) for m,s,_ in cases)) == 18
    cache = {}
    result = []
    for model, step, tau in cases:
        if model not in cache:
            cache[model] = history(model)
        records, provenance = cache[model]
        before = [r for r in records if r['step'] <= step]
        assert before and step-before[-1]['step'] < 20
        latest = before[-1]
        # Same definition as section 4: sampled logged minibatch losses, NOT
        # the new logger's moving average (which averages every intervening step).
        window = [r for r in before if step-1000 < r['step'] <= step and r['step'] % 20 == 0]
        assert len(window) == 50, (model,step,len(window))
        if model.startswith('4b-wsd-'):
            branch = model.removeprefix('4b-wsd-')
            checkpoint = f'{NEW_ROOT}/checkpoints/4b-wsd/{branch}/step{step}'
            schedule = 'WSD stable' if branch == 'stable' else 'WSD cooldown'
        elif model in ('0p125b','0p25b'):
            checkpoint = f'{NEW_ROOT}/checkpoints/{model}/stable/step{step}'
            schedule = 'Cosine'
        else:
            checkpoint = f'{OLD_ROOT}/checkpoints/{model}/dspark-proportional-{model}-step{step}'
            schedule = 'Cosine'
        row = dict(model=model,schedule=schedule,checkpoint_step=step,source_checkpoint=checkpoint,
                   last_logged_step=latest['step'],last_logged_loss=latest['loss'],
                   checkpoint_step_loss=latest['loss'] if latest['step']==step else None,
                   last_logged_lr=latest['lr'],window_loss=statistics.mean(r['loss'] for r in window),
                   window_start_exclusive=step-1000,window_end_inclusive=step,
                   window_logged_samples=len(window),holdout_acceptance=tau,
                   bigcodebench_acceptance=None,livecodebench_acceptance=None,
                   training_loss_provenance=provenance,
                   validation_summary=str((curve_path if model=='4b-wsd-stable' else final_path).relative_to(HERE.parent)))
        if step == 68703:
            for key in ('bigcodebench_acceptance','livecodebench_acceptance'):
                row[key] = final_by_model[model][key]
        result.append(row)
    artifact = dict(scope='18 distinct validated trained checkpoints in the proportional scaling family; older unrelated campaigns excluded',
        definition='Training objective, not held-out loss or perplexity. Window = mean of 50 logged minibatch losses every 20 steps in (checkpoint_step-1000,checkpoint_step]. Latest minibatch and window are different quantities.',
        untrained_synthetic_models={'16b':None,'32b':None},
        validation_source_sha256={str(p.relative_to(HERE.parent)):digest(p) for p in (final_path,curve_path)},
        checkpoints=result)
    (EVIDENCE/'validated-checkpoint-losses.json').write_text(json.dumps(artifact,indent=2)+'\n')
    columns = [k for k in result[0] if k != 'training_loss_provenance']
    with (EVIDENCE/'validated-checkpoint-losses.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result)
    print('| Model / schedule | Checkpoint step | Last loss-log step | Last logged loss | Prior 1K-step sampled mean | OpenCode 5K τ |')
    print('|---|---:|---:|---:|---:|---:|')
    names={'0p125b':'0.125B','0p25b':'0.25B','0p5b':'0.5B','1b':'1B','2b':'2B','4b':'4B','8b':'8B',
           '4b-wsd-stable':'4B','4b-wsd-cooldown':'4B'}
    for r in result:
        print(f"| {names[r['model']]} / {r['schedule']} | {r['checkpoint_step']:,} | {r['last_logged_step']:,} | {r['last_logged_loss']:.6f} | {r['window_loss']:.6f} | {r['holdout_acceptance']:.3f} |")


if __name__ == '__main__':
    main()
