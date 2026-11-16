# Incremental stable WSD validation evidence

Only complete, audited 5,000-prompt evaluations are published. The full sweep
contains ten 1B and ten 16B checkpoints. The September 25 unattended recovery
supersedes the five-hour cutoff: finish final stable 16B on the holdout,
LiveCodeBench and BigCodeBench first, then all remaining 16B/1B milestones.
The two previously completed 1B holdouts and three baselines are retained.

External results under `external/` use all 1,055 LiveCodeBench release_v6 or
1,140 BigCodeBench-Instruct v0.1.4 problems, rather than 5,000. Their manifests
record the actual row count. `external-results.json` is separate from the
holdout's `results.json` and `results.csv`.

Recovered 16B runs disable prefill CUDA graphs after a reproducible illegal
memory access in breakable prefill graph replay. Decode graphs remain enabled.
The generation contract is unchanged, but the preserved target-only baselines
used prefill graphs, so recovered 16B speedup is explicitly null/not matched.
Acceptance and output-token throughput remain reported. The 1B runtime and its
matched-baseline speedup calculation are unchanged.

Each `baseline-{a,b,c}` directory contains a same-node target-only baseline.
Each `{size}-stable-step{N}` directory contains a trained checkpoint evaluation.
Files appear only after full row/identity/counter/histogram checks pass:

- `metrics.json.gz`: lossless compressed original result, including all 5,000
  generated responses, per-phase metadata and acceptance counters.
- `metrics-summary.json`: independently recomputed summary.
- `result.json`: checkpoint summary with its same-worker baseline and speedup.
  Baseline directories do not have this checkpoint-specific file.
- `COMPLETED.json`: worker completion marker for checkpoints.
- `server-command.json`, `metrics-command.json`: exact commands.
- `manifest.json`: source NFS directory, file byte counts/SHA-256 digests and
  the SHA-256 of the decompressed original metrics JSON.

To read one raw result after cloning:

```bash
gzip -dc evidence/wsd-validation-20260925/16b-stable-step15588/metrics.json.gz | less
gzip -dc evidence/wsd-validation-20260925/16b-stable-step15588/metrics.json.gz | sha256sum
```

Compare the latter digest to `raw_metrics_sha256` in that directory's manifest.
An absent directory means the result has not been published; it is not a zero
score. The top-level `results.json`/`results.csv` join completed validation with
the previously preserved training-loss windows. The original loss evidence is
unchanged. No model/optimizer weights or incomplete response files are committed.
Audited completions are committed and pushed by a five-minute unattended watcher.
Previous failed attempts and their logs are preserved on NFS, not counted as
completed evaluations.

Acceptance includes the bonus token. Position curves are prefix-survival
probabilities. Throughput speedup compares natural-length generation to the
same-node baseline; it includes padded TP8 serving overhead and is not a code
correctness measurement. The full 5K contract is unchanged.
