# DSpark: previous and new results side by side — September 20, 2026

Matched 1.210B-token proportional-family campaign. Target: **Qwen3-235B-A22B-Thinking-2507**. All final models trained on the same 549,626-row corpus for one epoch / 68,703 updates (549,624 rows consumed at global batch 8). Parameter counts below exclude frozen target embeddings/output head.

Evaluation: same fresh OpenCode 5,000, BigCodeBench-Instruct v0.1.4 1,140, and LiveCodeBench release v6 1,055 prompts. Full prompts; Qwen3-thinking-nosystem; existing two-pass 2,048-thinking-phase / nominal 1,024-answer budget; T=0.6, top-p=0.95, top-k=20; same per-ID seed policy; TP8 BF16, concurrency 32, radix cache disabled, draft block size 7. Input-ID hashes and all recorded evaluation settings were checked for equality.

## Final acceptance length

Acceptance length τ = completion tokens / verification calls, **including bonus**, across both phases. This is not correctness/pass@1.

| Nominal model | Actual params | K=L | LR schedule | Campaign | OpenCode 5K | BigCodeBench | LiveCodeBench |
|---|---:|---:|---|---|---:|---:|---:|
| 0.125B | 0.114B | 1 | Cosine | new | 2.067 | 1.721 | 1.721 |
| 0.25B | 0.202B | 2 | Cosine | new | 3.541 | 2.652 | 2.760 |
| 0.5B | 0.530B | 3 | Cosine | previous | 4.226 | 3.260 | 3.389 |
| 1B | 0.915B | 4 | Cosine | previous | 4.439 | 3.442 | 3.567 |
| 2B | 1.876B | 5 | Cosine | previous | 4.581 | 3.578 | 3.715 |
| 4B | 3.927B | 7 | Cosine | previous | 4.765 | 3.770 | 3.890 |
| 4B | 3.927B | 7 | WSD stable (no decay) | new | 4.640 | 3.713 | 3.850 |
| 4B | 3.927B | 7 | WSD + cooldown | new | 4.802 | 3.821 | 3.950 |
| 8B | 8.303B | 9 | Cosine | previous | 4.854 | 3.831 | 3.988 |

### What changed for 4B

| Comparison (acceptance delta) | OpenCode 5K | BigCodeBench | LiveCodeBench |
|---|---:|---:|---:|
| Cooldown − stable | +0.162 | +0.107 | +0.100 |
| Cooldown − previous cosine | +0.037 | +0.050 | +0.060 |
| Previous 8B − 4B cooldown | +0.052 | +0.010 | +0.038 |

The cooldown branch starts from stable step 61,833 and decays LR over the same final 6,870 updates. Both 4B branch endpoints have equal data exposure; computing both branches adds 10% extra 4B training work to the campaign. The old cosine run used a different training execution layout; comparisons against it are not a perfectly isolated LR-only ablation. All are single runs, without seed-repeat confidence intervals.

## 4B stable trajectory — without cooldown

| Training fraction | Step | Traces consumed | OpenCode 5K τ |
|---|---:|---:|---:|
| 10% | 6,870 | 54,960 | 3.527 |
| 20% | 13,741 | 109,928 | 3.926 |
| 30% | 20,611 | 164,888 | 4.122 |
| 40% | 27,481 | 219,848 | 4.257 |
| 50% | 34,352 | 274,816 | 4.351 |
| 60% | 41,222 | 329,776 | 4.419 |
| 70% | 48,092 | 384,736 | 4.491 |
| 80% | 54,962 | 439,696 | 4.547 |
| 90% | 61,833 | 494,664 | 4.592 |
| 100% | 68,703 | 549,624 | 4.640 |

The curve rises throughout, with diminishing gains: 80%→100% adds about 0.093 tokens. This measures the unannealed trajectory, not optimal cooled performance at each budget, and does not establish saturation. Earlier checkpoints were evaluated only on the 5K holdout; no external checkpoint curves are implied.

## Observed end-to-end output throughput (tokens/s)

**These are the padded TP8 implementations, not native-model speedups.** Runs occurred at different times/on different pods, so do not interpret small differences as statistically reliable speed advantages. No new target-only baseline was run; no speedup ratio is claimed.

| Model / schedule | OpenCode 5K | BigCodeBench | LiveCodeBench |
|---|---:|---:|---:|
| 0.125B / Cosine | 1,697.7 | 1,458.7 | 1,413.8 |
| 0.25B / Cosine | 2,794.3 | 2,165.7 | 2,184.7 |
| 0.5B / Cosine | 3,274.6 | 2,608.1 | 2,624.0 |
| 1B / Cosine | 3,414.2 | 2,727.0 | 2,740.9 |
| 2B / Cosine | 3,487.4 | 2,803.3 | 2,818.8 |
| 4B / Cosine | 3,555.4 | 2,892.1 | 2,889.6 |
| 4B / WSD stable (no decay) | 3,449.2 | 2,840.0 | 2,843.0 |
| 4B / WSD + cooldown | 3,568.4 | 2,925.2 | 2,919.9 |
| 8B / Cosine | 3,548.7 | 2,877.9 | 2,900.4 |

## Output-budget caveat

| Model / schedule | Holdout truncated / 5,000 | BCB truncated / 1,140 | LCB truncated / 1,055 |
|---|---:|---:|---:|
| 0.125B / Cosine | 245 | 7 | 339 |
| 0.25B / Cosine | 247 | 9 | 338 |
| 0.5B / Cosine | 244 | 6 | 362 |
| 1B / Cosine | 245 | 2 | 347 |
| 2B / Cosine | 251 | 7 | 353 |
| 4B / Cosine | 255 | 5 | 352 |
| 4B / WSD stable (no decay) | 245 | 6 | 349 |
| 4B / WSD + cooldown | 236 | 7 | 349 |
| 8B / Cosine | 250 | 7 | 336 |

All responses, including truncated ones, remain in the headline acceptance scores. Around one-third of LiveCodeBench answers hit the answer limit. No executable correctness grading was done. Earlier DSpark/EAGLE experiments with different corpora, holdouts, architectures or budgets are deliberately not merged into this matched table.

## Audit and source artifacts

The stable-curve queue accidentally duplicated steps 6,870, 48,092 and 61,833 because cross-node file locks did not exclude workers. Retained raw files were revalidated (5,000 unique prompts per checkpoint; no summary/raw mismatch). Earlier duplicate full outputs may have been overwritten; scalar logs survive. No best-of-repeat selection was applied. The queue now uses exclusive directory creation, tested across all three nodes.

- [Machine-readable comparison](comparison.json) and [CSV](comparison.csv).
- [Validated stable curve](evidence/stable_curve.json) and [duplicate-work audit](evidence/curve_audit.json).
- [Source checksums and paths](evidence/source-manifest.json).
- [Original scaling writeup](../../dspark-scaling-writeup.md).
