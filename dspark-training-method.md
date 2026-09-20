# DSpark shared-prefill training — running

Launched September 18, 2026. See [PLAN.md](PLAN.md) for architecture rounding, data and schedule details.

- All three models completed a 20-step full-setting pilot, saved all model/optimizer/RNG state, restarted, and continued past step 100 with finite losses/gradients.
- Initial launch audit verified full model parameter counts equal the sum of all rank-local FP32 master parameters, correct optimizer/scheduler restoration, and step-21 LR `2.0388349514563107e-6`.
- Actual trainable sizes: 114,170,625; 202,160,001; 3,926,503,681. The first two are aspect-prioritized approximate nominal 0.125B/0.25B sizes, not exact parameter budgets. All are DSpark. The WSD model's architectural config exactly equals the original 4B config.
- One Qwen target TP8 prefill producer serves all three through Mooncake RDMA. It captures the union of eight required target layers and the final hidden state once per training sequence in the main phase.
- Three-consumer retention/backpressure unit tests passed; pilot owner emptied its queue only after all three consumers acknowledged. Master memory remained bounded during the continued run, with zero allocation failures in the observed startup window.
- Prior original-capture sequence lengths matched the identical cached/shuffled order for 10,000 4B refs and all 5,120 available original 1B refs. Full-sequence lengths max at 4,773. The largest 128-row window requires 23,770,198,496 raw feature bytes, below the 34,359,738,368-byte owner segment.

## Training method with four nodes

Use the same four GPU nodes in two sequential phases. This is the intended
four-node workflow; the running three-node experiment is listed separately below.

### Phase 1 — Generate training traces on all four nodes

Run a target-model generation server on each node and distribute **disjoint
prompt shards** across all four workers. For an eight-GPU node, use the established
SGLang TP8, no-EP generation setup and the same target checkpoint, Qwen3 thinking
chat template, sampling parameters, thinking budget and answer-budget policy on
every worker. Exclude validation/holdout prompts and already-selected training
prompts before assigning shards, so workers do not generate duplicate selections.

Persist generated text traces incrementally to separate per-worker files on NFS,
including source IDs, reasoning, final answers, finish reasons and budget metadata.
After **all four generation shards finish**, merge and audit them: check coverage,
duplicates, holdout overlap, repetition and answer completeness; apply the agreed
quality filtering; then freeze a versioned training corpus with row/token counts
and checksums. Preserve the separate validation sets and filtering manifests.

This phase generates and saves text. The reusable hidden-state stream is computed
from those saved traces in Phase 2; there is no need to regenerate their answers
during training.

### Phase 2 — One shared prefill node, three drafter-training nodes

Reassign the four nodes after trace generation is complete:

```text
Frozen training traces on NFS
              |
       Node 0: target prefill
       + Mooncake feature owner
              |
       shared hidden-state stream
              |
       +------+------+------+
       |             |      |
    Node 1        Node 2   Node 3
    small         medium  large
    DSpark        DSpark  DSpark
    trainer       trainer trainer
```

- **Prefill node:** replay the frozen corpus through the frozen target in a
  deterministic order. Capture the union of target layers required by the three
  draft architectures, plus the final target hidden state. Reuse each capture for
  all three trainers instead of running three independent target prefills.
- **Trainer nodes:** each node trains a different-sized DSpark, using its GPUs
  for distributed draft training. Each trainer selects its own required layer
  slices from the union, computes teacher distributions using the frozen target
  LM head, and updates only its drafter. Keep comparable data exposure, global
  batch and objective settings across sizes; record any schedule differences.
- **Mooncake pipeline:** transfer captured hidden states, token IDs and loss masks
  directly from the prefill node to all three trainer nodes, using cross-node RDMA
  where available. HTTP carries only control metadata; NFS stores durable text,
  manifests, logs and checkpoints, not the transient hidden-state payload stream.
  This is hidden-state sharing, not target KV-cache sharing.
- **Overlap and ownership:** pipeline prefill of upcoming batches, Mooncake
  transfer/prefetch, and training of already-received batches. Use bounded buffers,
  independent trainer cursors and backpressure. The owner must retain each shared
  feature object until **all three trainers** acknowledge the corresponding
  optimizer update; one fast trainer must not delete data another still needs.
- **Recovery and comparison:** save model, optimizer, scheduler and RNG state,
  along with data position, to NFS. Preserve the chosen checkpoint cadence (10%
  milestones in this campaign). For WSD, retain unannealed main-branch checkpoints
  and keep cooldown comparisons in separate branches. On recovery, recapture the
  needed deterministic suffix rather than trusting stale in-memory feature keys.

One producer amortizes target-prefill work across three models, but does **not**
guarantee that all three trainers are compute-saturated. Track target capture
throughput, each trainer's data-wait/transfer/compute time, and buffer occupancy;
the slowest stage determines sustained progress. The existing driver also pins
node addresses and GPU assignments, so a four-node deployment requires explicit
launcher/config updates and a cross-node smoke/resume test—not merely a fourth
node becoming available.

## Current three-node topology

| Role | Pod | Access |
|---|---|---|
| Prefill, Mooncake owner, metadata broker, status recorder | fa9c5d839b8394bb4ae9-n0-0 | root `pod.sh` |
| Small DSpark trainers, four GPUs each | f820921679644432bb13-n0-0 | `../idev-continuation-a/pod.sh` |
| 4B DSpark WSD, eight GPUs | fe5cff5295a954629a31-n0-0 | `../idev-continuation-b/pod.sh` |

All keep-alives remain running. lva1 was not queried or changed.

## NFS artifacts

`/shared/user/H200/specforge/dspark-shared-prefill-wsd-20260918/`

- `configs/`: exact model configurations.
- `manifests/architecture.json`, `token-order.json`, `order-buffer-verification.json`, `launch-verification.json`.
- `logs/{0p125b,0p25b,4b-wsd}-{pilot,main}.log`: process logs.
- `logs/{0p125b,0p25b,4b-wsd}-stable.jsonl`: step/loss/LR and wait/compute/transfer windows.
- `logs/producer-main.log`: capture batch timings and independent acknowledgement cursors.
- `logs/watch-5min.jsonl`: on-pod five-minute status snapshots. This is a recorder, not automatic chat reporting or a crash-recovery agent.
- `checkpoints/{size}/stable/step{N}/`: complete training states, with `COMPLETED.json` written only after all ranks have saved.
- `checkpoints/4b-wsd/cooldown/`: separately queued final cooldown branch.
- `scripts/`: launch/runtime/test source snapshot.

Keep every rounded 10% checkpoint; keep only the newest two additional recovery checkpoints. Main 4B checkpoints remain unannealed after warmup. Final cooldown is queued from step 61,833 through step 68,703, with the same final data suffix, not another epoch. No claim of loss parity with the prior cosine endpoint (~0.630) is established yet.

## Recovery boundaries

The short checkpoint/resume test passed. For a subsequent interrupted run, stop only these owned trainer/broker processes before rebuilding their stream; do not independently restart a trainer behind a live acknowledgement cursor. For a crashed producer with potentially orphaned pinned objects, restart the **dedicated** capture/master as well to clear the previous process-lifetime store; do not touch other workloads or keep-alive. Relaunch the producer from the slowest complete checkpoint and all consumers from their own latest complete checkpoints. New session keys prevent stale locator reuse. Ahead consumers skip their already-trained prefix.

Use `run_trainers.sh SIZE` on the assigned node. `producer.py --with-cooldown` schedules the shared main stream and final cooldown suffix. The producer and trainers must use the same run-local configuration and metadata authentication token. The token is not included in this source handoff.

This is a new metadata/fan-out training driver using the existing SpecForge architecture, objective, FSDP backend and optimizer, not an upstream production feature. Early steady-state throughput and longer-term LR behavior still need observation. Total run horizon is 68,703 updates per main model; the additional 4B cooldown fork costs 6,870 updates and a corresponding suffix recapture.

## More efficient alternative: two producers, one consumer node — September 20 update

The original topology and launch notes above are preserved as the historical
design. We subsequently tested and adopted a **more efficient allocation for this
prefill-bound workload**: two target-prefill nodes feeding one node that hosts all
three draft trainers. Training and the planned evaluations have now completed;
the earlier references to “running” and “queued” describe the original launch.

### Why change the allocation?

In the original three-node deployment, one producer supplied two trainer nodes.
The drafters spent most of their update time waiting for captured target features.
Consolidating the three trainers onto one eight-H200 node freed another node for
target prefill, increasing feature supply without adding GPUs. This is a measured
improvement over the original **three-node** deployment, not a benchmark against
the proposed four-node topology above.

```text
Frozen traces -> deterministic order -> coordinator assigns disjoint blocks
                                         /                  \
                              Producer A: TP8      Producer B: TP8
                              feature owner        feature owner
                                         \                  /
                                     Mooncake RDMA reads
                                              |
                                  One eight-H200 consumer node
                                  /            |             \
                           GPUs 0–1        GPUs 2–3        GPUs 4–7
                           0.125B          0.25B           4B WSD
```

“One consumer” means **one machine, not one model or optimizer**. There remain
three independent logical consumers with their own model state, schedule, data
cursor and acknowledgement stream.

| Role | Pod | Allocation |
|---|---|---|
| Producer A, Mooncake master/owner, broker | fa9c5d839b8394bb4ae9-n0-0 | Frozen target TP8 |
| Producer B, Mooncake owner | f820921679644432bb13-n0-0 | Independent frozen target TP8 replica |
| All three trainers | fe5cff5295a954629a31-n0-0 | 0.125B: 2 GPUs; 0.25B: 2 GPUs; 4B: 4 GPUs |

### Capture, transfer and training contract

- Both producers run the same frozen Qwen target with BF16, TP8, no EP,
  **64 concurrent capture requests** and `--max-prefill-tokens 200000` each.
  These are two replicas, not TP16. The token cap is a scheduler limit, not a
  guarantee that every scheduled batch contains 200K tokens.
- The coordinator assigns **disjoint contiguous blocks of up to 64 traces** from
  the same seed-42 shuffled corpus. A trace is captured by one producer and reused
  by all three models. Each capture includes the union of eight target taps and
  the final hidden state, plus input IDs and loss masks; no answers are regenerated.
- Blocks can finish out of order, but consumers see only the completed contiguous
  prefix. The corpus, masks, objective and per-model optimizer batch remain
  unchanged: **eight traces per update**, with per-rank batches **4 / 4 / 2** for
  the 2 / 2 / 4-rank trainers. Producer concurrency does not change training batch.
- Mooncake RDMA carries feature tensors into trainer-side host buffers, followed
  by GPU copies. HTTP carries control metadata; NFS holds durable artifacts.
  Each logical trainer fetches the union and selects its own taps locally. This
  is hidden-state sharing, not KV-cache sharing or a zero-copy GPU pipeline.
  Prefetch of the next update overlaps current computation.
- Global lookahead is **256 traces** beyond the slowest consumer. Each producer
  has a **32-GiB owner segment** with a conservative **26-GiB raw-feature reservation
  budget**, including in-flight work. Reservations use exact sequence lengths;
  blocks shrink in multiples of eight when necessary. Features are retired only
  after every active model acknowledges completion of the corresponding update.

### Measured efficiency improvement

The 100-update migration pilot passed. Timing windows excluded initial warmup
and checkpoint saves; the old and new windows used adjacent, different data,
so this is descriptive evidence rather than a matched-input controlled benchmark.

| Model | Original seconds/update | Two-producer pilot seconds/update | Original data wait | Pilot data wait |
|---|---:|---:|---:|---:|
| 0.125B | 2.225 | 1.311 | 2.048 s | 0.987 s |
| 0.25B | 2.225 | 1.313 | 2.014 s | 0.929 s |
| 4B WSD | 2.223 | 0.927 | 1.787 s | 0.108 s |

The small trainers improved by approximately **1.7× at the same total 24-GPU
allocation**. The 4B pilot initialized later and caught up on already-buffered
features; its apparent 2.4× improvement must **not** be treated as sustained
pipeline speedup. See [pilot evidence](DUAL-PREFILL-RESULTS.md).

This allocation is preferable when target-prefill supply is the bottleneck and
the drafters fit and keep up on the consolidated node. It is not universally
better for larger or more compute-intensive drafters. Track capture throughput,
trainer wait/compute/RDMA/H2D time, and buffer occupancy. Block-wide publication
and the slowest trainer can still limit progress; GPU utilization is not MFU.
The implemented coordinator supports exactly two producers. A third producer
would require code/config changes and new capacity, ordering and recovery tests.

### Migration, checkpoints and artifacts

The original step-6,000 checkpoints remain preserved. Rank resharding verified
bit-exact model weights, FP32 masters, Adam moments/steps and scheduler state;
the new topology passed a 100-update pilot and continued from step 6,100. Rank
regrouping changes stochastic RNG draws, so future updates are not claimed to
be bitwise identical to the old topology.

All main models completed **68,703 updates**. The 4B WSD stable branch and every
10% checkpoint remain unannealed; a separate cooldown fork starts at step 61,833
and runs the same final 6,870-update data suffix. Both branches were preserved
and evaluated. Recovery still uses complete NFS optimizer/RNG checkpoints and
freshly recaptured deterministic suffixes, not stale Mooncake keys.

Continuation NFS root:
`/shared/user/H200/specforge/dspark-shared-prefill-wsd-20260918/dual-prefill-224/`

- `scripts/dual_producer.py`: two-owner allocation, ordered publication and retention.
- `scripts/start_dual_capture.sh`, `dual_env.sh`, `run_dual_trainers.sh`: new launch path;
  the single-producer launch commands above are historical, not this deployment.
- `manifests/conversion-readback.json`, `pilot-results.json`, `promotion.json`: migration evidence.
- `logs/producer-to-68703.log`, `logs/*-stable.jsonl`: producer and trainer measurements.
- `checkpoints/`: separate stable/cooldown branches and complete training states.
- `eval/final-three-datasets/`, `eval/4b-stable-trajectory-5k/`: completed validation.

These are recorded training assignments, not live pod-status claims. No workload
needs restarting to use this documentation; unrelated workloads and keep-alives
remain outside the training driver's scope.
