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
