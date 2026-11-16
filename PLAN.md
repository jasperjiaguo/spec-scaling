# Shared-prefill DSpark campaign — 2026-09-18

## User scope

All three drafters are DSpark. One H200_8 target-prefill node supplies two small trainers on a second H200_8 node and a WSD trainer on the third. The WSD model uses the existing **4B config**, superseding the original 1B request.

| Node | Role |
|---|---|
| pod-05 | Qwen3-235B-A22B-Thinking-2507 BF16 TP8, no EP; capture union; Mooncake RDMA owner |
| pod-02 | 0.125B nominal on GPUs 0–3; 0.25B nominal on GPUs 4–7 |
| pod-09 | Existing 3,926,503,681-parameter DSpark architecture, WSD |

Keep-alive remains running on every node. No prior model checkpoint, corpus, or environment is overwritten.

## Architecture rounding

Retain the official-reference scaling constraints: K=L, FFN/width=3.4, head dimension 128, Q/KV ratio 5, block size 7, Markov rank 256, target width 4096. The fixed Markov matrices alone contain 77,791,232 parameters, so sub-0.5B total-parameter scaling is not cubic. Layer/head integer rounding is significant.

Proposed aspect-prioritized configurations (actual counts, not exact nominal labels):

| Name | Actual trainable parameters | K=L | Width | FFN | Q/KV | Target taps |
|---|---:|---:|---:|---:|---|---|
| 0.125B nominal | 114,170,625 | 1 | 1280 | 4352 | 10/2 | 47 |
| 0.25B nominal | 202,160,001 | 2 | 1920 | 6528 | 15/3 | 1,92 |
| 4B WSD | 3,926,503,681 | 7 | 6400 | 21760 | 50/10 | 1,16,31,46,62,77,92 |

The two-layer model has width/depth=960, within 6.25% of reference 1024. The one-layer model's closest legal width is 1280 (+25%); do not claim exact proportionality. User was presented the alternative two-layer 2560-wide model (284.9M) and the aspect-prioritized recommendation before training launch. A single target tap uses the existing model helper's midpoint convention, layer 47. Frozen shared embedding/head parameters are excluded, as in prior runs.

## Data and objective

Reuse the immutable 549,626-row, 1.210B-full-template-token corpus and previously verified cached tokenization. Same qwen3-thinking-nosystem parsing, preserved reasoning, masked user tokens, 8192 sequence limit; no generation, pruning, or holdout contamination. The 32 original cache shards already contain the preprocessing shuffle; apply the original producer's Python seed-42 shuffle to their row indices. Global batch 8 yields 68,703 updates and drops the same two tail rows.

Small trainers: four FSDP ranks × microbatch 2. 4B: eight FSDP ranks × microbatch 1. No accumulation. The pooled DSpark loss is normalized over the complete global optimizer batch by the existing objective. Small-model anchor RNG draws can differ with batching; this is not a bitwise replay of an eight-rank layout.

Use original SpecForge model, loss and FSDPTrainingBackend/BF16Optimizer implementations from the preserved code snapshot. BF16 parameters, FP32 masters, AdamW defaults, weight decay 0, clip 1, seed 42, SDPA, 512 anchors and 64-block objective chunks; CE=.1, distribution L1=.9, confidence=1, gamma=4.

## LR and checkpoints

Small models: original cosine schedule, peak 1e-4, 1,030 warmup steps, 68,703 total.

4B main branch: 1,030-step warmup to 1e-4, then stable throughout the epoch. This LR has a prior stability reference in the **same 4B architecture**, but is not claimed optimal for WSD. Monitor finite gradients and early losses against the original 4B curve; no guarantee that an unannealed endpoint equals a cooled endpoint.

Save complete model, optimizer masters/moments and rank RNG at rounded 10% milestones: 6,870; 13,741; 20,611; 27,481; 34,352; 41,222; 48,092; 54,962; 61,833; 68,703. All main-branch 4B milestone checkpoints are unannealed. Additional recovery saves every 2,000 steps retain the newest two, never prune milestone checkpoints.

User approved a separate cooldown comparison: after the stable endpoint, restore its 90% checkpoint (61,833), including optimizer and RNG, and train the final 6,870 steps with cosine decay from 1e-4 toward zero, replaying the same final 10% suffix. Preserve stable outputs. Each final branch sees one epoch; campaign compute is +10% for the extra 4B branch and suffix target recapture. Compare against prior 4B cosine final training-loss window ~0.630 using matching log windows; also retain the original logs. WSD motivation: https://arxiv.org/abs/2410.05192 . Constant-LR loss alone cannot establish data saturation.

## Shared capture and recovery

SGLang capture union [1,16,31,46,47,62,77,92]. Prefill requests up to 64 concurrently; bounded resident window 128 samples. Capture only once per text during the main phase, then each trainer slices its required tap identities in the original order. Final target hidden states supply teacher distribution supervision; this is hidden-feature fan-out, not KV-cache training or regenerated text.

RDMA uses direct pod IPs, proven NIC selection mlx5_0,mlx5_3,mlx5_4,mlx5_5. Tensor payloads never traverse HTTP or NFS. The producer is the only deletion authority; a sample is released only after all active models acknowledge completion of its optimizer update. Independent trainer cursors prevent consume-once deletion bugs. HTTP is metadata-only and authenticated with a run-local token.

Checkpoints are staged and published only after every rank finishes. Restart all consumers and producer from the latest complete checkpoints, with new capture generation keys and deterministic recapture from the slowest checkpoint. Ahead consumers skip their already-trained prefix. A failure must not advance a cursor or promote a partial checkpoint. Long stalls fail rather than grow the queue indefinitely. This initial rollout requires a short full-setting run plus checkpoint/restart test before committing to the full epoch.

Artifacts: /workspace/specforge/dspark-shared-prefill-wsd-20260918 . Source scripts remain in the matching local handoff directory and are copied into NFS.
