# DSpark scaling experiments

Start with [the results report](dspark-scaling-writeup.md) for architecture,
data quality, loss histories, completed acceptance/throughput measurements and
the synthetic timing sweep. Read [the training method](dspark-training-method.md)
for the historical topologies and latest two-producer / one-consumer campaign.

## Latest update — September 25, 2026

The new **1B and 16B stable WSD runs are complete**. All ten milestone losses
per model and the raw training logs are preserved in
[milestone evidence](evidence/wsd-1b-16b-20260925/milestone-losses.json).
Logging occurs every20 steps; report tables identify the latest available
logging step rather than inventing an exact checkpoint loss.

Their cooldown has **not run**. A newly authorized 5K validation campaign is
running on three fresh 8×H200 pods. Each audited completed evaluation is
published incrementally with compressed full responses and SHA-256 provenance
in [the live results section](dspark-scaling-writeup.md#13-stable-1b16b-validation--live-audited-results).
The queue covers all ten stable checkpoints per size, but the remaining five-hour
window prioritizes both final checkpoints, then paired midpoint/early/late anchors.
The full twenty-checkpoint sweep is not promised within that window.
See [validation launch and measurement contract](#validation-launch).
Do not confuse the trained16B with the untrained16B synthetic timing model.

## Validation launch

September 25, 18:40 UTC startup snapshot: pods `pod-07`,
`pod-04`, and `pod-03` are Running;
NVML sees eight H200 GPUs on each, and keep-alives are active. The previous
preservation-only pause is superseded for validation, not new training.

**19:56 UTC update:** all three baselines are generating normally. Their successors
are waiting on per-worker locks and will reuse completed same-node baselines.
Unclaimed legacy jobs are fenced so they cannot race the new priority queue.
No active generation was stopped, and sample size/output budgets are unchanged.
The planning cutoff is September 26, 00:51 UTC; no new job starts with less than
90 minutes left. Active work is not forcibly killed at that cutoff.

Each worker runs its own full 5,000-prompt target-only baseline, then claims
distinct checkpoint jobs. Full prompts, concurrency 32, Qwen3 thinking template,
2,048-initial/1,024-answer two-pass generation, temperature 0.6/top-p 0.95/top-k 20,
and seven draft positions match the established contract. Acceptance is real,
not simulated. Throughput speedups use the same worker's baseline. TP8 zero-head
padding adds serving compute, so speed is not an estimate of unpadded execution.

NFS output root:
`/workspace/specforge/dspark-wsd-1b-16b-20260923/eval/holdout-curve-20260925/`

The output root retains scripts, environment freezes, server/client logs, full
generated responses, position-wise acceptance summaries and job completion or
failure markers. Worker processes survive client disconnection; failure markers
require inspection and are not automatically treated as completed work. External
benchmark runs remain subsequent work. No checkpoint binaries are published.
An independent five-minute publisher downloads only fully audited completed
evaluations, updates the write-up and raw evidence, commits and pushes each one.
Transfer/Git errors are logged and retried; incomplete runs are never scored.

## Evidence and scope

- [New 1B/16B milestone CSV](evidence/wsd-1b-16b-20260925/milestone-losses.csv),
  raw [1B](evidence/wsd-1b-16b-20260925/1b-stable.jsonl) and
  [16B](evidence/wsd-1b-16b-20260925/16b-stable.jsonl) logs,
  and [checksums](evidence/wsd-1b-16b-20260925/SHA256SUMS).
- [Previous cosine/small-model/4B-WSD comparison](comparisons/20260920/comparison.md)
  and [validated-checkpoint losses](evidence/validated-checkpoint-losses.csv).
- [Synthetic speed measurements](evidence/synthetic-speed-20260922/results.csv)
  and [audit](evidence/synthetic-speed-20260922/summary-audit.json).
- [Data examples](evidence/training-examples.jsonl),
  [holdout output examples](evidence/holdout-examples.json), and
  [external-output quality audit](external-output-audit/README.md).

This repository contains reports, compact evidence, raw loss logs and small
checkpoint metadata, **not model weights or optimizer states**. Full training
corpus and full validation-generation archives remain on NFS at the paths in
the report. Example rows are not a copy of the entire dataset. Historical plans
are retained for provenance; they are not current launch instructions.

Host names and storage paths in the reports and evidence files have been
replaced with neutral placeholders (`pod-01` to `pod-09`, `/workspace/...`,
`/models/...`, `/data/...`, `/home/user/...`), and the checksums recorded in
this repository were regenerated to match. No measurement was changed.

To verify the new milestone evidence after cloning:

```bash
cd evidence/wsd-1b-16b-20260925
sha256sum -c SHA256SUMS
```
