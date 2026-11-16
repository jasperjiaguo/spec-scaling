# Two-prefill / one-trainer pilot — September 18, 2026

The 100-update pilot passed and was promoted to continued training from step
6,100. The original step-6,000 checkpoints remain intact for recovery.

## Active layout

| Role | Pod | Configuration |
|---|---|---|
| Target prefill A, Mooncake master and broker | pod-05 | TP8, no EP, 64 requests, 200K prefill-token cap |
| Target prefill B | pod-02 | Same target and capture configuration |
| All three DSpark trainers | pod-09 | 114M: GPUs 0–1; 202M: GPUs 2–3; 4B: GPUs 4–7 |

Global training batch is still **eight traces per model per optimizer update**.
Model architectures, corpus/order, masks, objectives, anchor count, schedules,
LRs and 10% checkpoint milestones are unchanged. The 4B stable branch and its
separate final cooldown comparison remain configured.

## State preservation and correctness checks

- Full checkpoint readback verified **bit-exact model weights, FP32 masters,
  Adam first/second moments, Adam steps and scheduler state** after 4→2 / 8→4
  rank conversion, across all 114,170,625 / 202,160,001 / 3,926,503,681 parameters.
- Source and converted checkpoints are separate; conversion did not apply an
  optimizer update or reset LR. Source-rank RNG states were retained according
  to the recorded mapping. Regrouped batches change stochastic anchor draws,
  so the subsequent trajectory is not claimed to be bitwise identical.
- The pilot processed exactly 800 distinct indexed traces: 400 from each
  target replica. All three consumers acknowledged step 6,100. Both owners'
  reserved feature counts returned to zero at completion; no payloads were
  retired before all consumers finished their updates.
- All three saved full step-6,100 checkpoints. Losses and gradients remained
  finite. Global batch remains eight with per-rank batches 4 / 4 / 2.
- Four optimizer-conversion numerical tests, five dual-producer ordering/
  ownership/budget tests, and the original three fan-out tests passed.

## Timing observations

Baseline: original topology, windows ending 5,500–5,980. Pilot: windows ending
6,040–6,100 (80 updates, excluding the initial warmup window). These are
**different adjacent data windows**, not a matched-input benchmark.

| Model | Old seconds/update | Pilot seconds/update | Old data wait | Pilot data wait | Pilot compute |
|---|---:|---:|---:|---:|---:|
| 114M | 2.225 | 1.311 | 2.048 | 0.987 | 0.317 |
| 202M | 2.225 | 1.313 | 2.014 | 0.929 | 0.376 |
| 4B WSD | 2.223 | 0.927 | 1.787 | 0.108 | 0.809 |

The pilot supports continuing the topology: roughly **1.7×** improvement in
the small trainers' measured rate, with enough 4B compute capacity to keep up.
Do **not** interpret the 4B's 2.4× per-window ratio as aggregate steady-state
speedup: it initialized later and consumed an existing buffer while catching up.
Longer production windows are needed for a stable common throughput estimate.
Checkpoint save time and initial model loading are not included in these window
averages.

Observed GPU memory during concurrent training was approximately 16 / 20 / 55
GB per GPU for the three models. Rank-0 peak allocated tensors were 11.6 / 14.9 /
40.9 GiB; allocator reservation and other runtime memory explain the difference.
GPU-utilization snapshots are not MFU measurements.

## Setup issue addressed

The new prefill pod lacked Python development headers. Installed the exact
matching `python3-devel-3.12.9-5.azl3` package (and its `expat-devel` dependency)
after Triton's first-time compile failed with missing `Python.h`. The Python
interpreter and serving package versions were not changed. The target server
then started successfully. The idev skill's `pod.sh` / rsync workflow was used;
all keep-alives stayed running.

## Artifacts and monitoring

Active NFS root:
`/workspace/specforge/dspark-shared-prefill-wsd-20260918/dual-prefill-224/`

- `manifests/conversion-readback.json`: exact state comparison results.
- `manifests/reshard-*-step6000-rank*.json`: rank mappings and tensor digests.
- `manifests/pilot-results.json`: pilot timings, sample/token counts and final ownership state.
- `manifests/promotion.json`: reviewed continuation decision.
- `logs/producer-to-6100.log`, `logs/*-stable.jsonl`: raw pilot evidence.
- `logs/producer-to-68703.log`: active dual-prefill production stream.
- `status/*.json`, `logs/watch-5min.jsonl`: live status and five-minute recorder.
- `scripts/`: source snapshot, plan, tests and this report.

The parent run's `status/active-continuation.json` points to this active root.
The recorder is a status ledger, not an automatic repair or chat-alert service.
