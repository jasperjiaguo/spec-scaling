# DSpark proportional-scaling experiment

*Updated September 20, 2026: smaller drafters, final WSD stable/cooldown evaluations, and the ten-checkpoint WSD stable validation curve.*

**September 20 update:** [Combined previous/new comparison](../dspark-comparison-20260920/comparison.md)
adds 0.125B, 0.25B, 4B WSD stable/cooldown, all three datasets, throughput,
and the ten-point unannealed 4B learning curve. These results are also included directly in [section 10](#10-smaller-drafters-and-4b-wsd-results-september-20-update). Sections 0–9 retain the original five-model cosine campaign and its audits; their loss/quality audits do not automatically extend to the new models.

We trained five DSpark drafters on the same **549,626-trace, 1.210B-token corpus**. All completed one epoch. On the same 5,000 held-out coding prompts, acceptance length increased from **4.226 to 4.854 tokens per verification**, including the bonus token, as draft size increased from 0.530B to 8.303B trainable parameters. Gains diminish with size; these results alone do not establish the fastest model or data saturation.

The completed external evaluations show the same size ordering: **3.260 → 3.831 on BigCodeBench** and **3.389 → 3.988 on LiveCodeBench**. These are acceptance measurements under the same bounded-output policy, not code-correctness scores.

Measured end-to-end output throughput at **concurrency 32** is nearly tied between 4B and 8B: approximately **3,550 tokens/s on OpenCode** and **2,890 tokens/s on the external benchmarks**. The higher acceptance of 8B does not translate into a clear throughput advantage in these runs; see section 8.

## 0. Target and draft models

The frozen target is **`Qwen/Qwen3-235B-A22B-Thinking-2507`**: a 235B-total/~22B-active MoE model with 94 transformer layers and hidden width 4,096. The drafters are custom-width **dense DSpark** models—not MoE—with a masked-block denoising backbone, Markov correction, and confidence head. All propose **seven draft tokens** per verification round; the target verifies and supplies the correction/bonus token.

The family follows the dimensions of the pinned official reference `deepseek-ai/dspark_qwen3_14b_block7` (revision `83207b416acf99f41c2184648923632fccea6dd0`). That reference supplies the architecture proportions; **our target is the 235B model, not Qwen3-14B**.

## 1. Data generation, cleaning, and split

Prompts come from `nvidia/OpenCodeInstruct`, revision `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`. Source selection requires `average_test_score >= 1.0`; this score applies to the source dataset's answer, **not to our newly generated Qwen answers**. Source IDs and normalized prompts were used to exclude prior selections and isolate the fresh holdout.

Generation used the frozen Qwen target through SGLang, BF16, **TP8 without EP**, on two eight-H200 machines for the new data. Generation concurrency was ultimately 128 after earlier lower-concurrency runs. Sampling was temperature **0.6**, top-p **0.95**, top-k **20**, min-p **0**. The `qwen3-thinking-nosystem` template preserves separate assistant `reasoning_content` and `content`, without inserting a generic system prompt.

The two-pass policy allows 2,048 initial output tokens. If thinking has not closed, the client appends an early-stop bridge ending in `</think>` and requests the final answer with a nominal 1,024-token answer budget. EOS is respected. Incomplete training answers are rejected/replaced, rather than accepted merely because the output limit was reached. Overlong prompts are rejected, not cut; the selection check reserves 3,100 tokens within an 8,192-token context.

This is **not an exact 2,048-reasoning/1,024-answer split in every saved row**: the initial request can contain both phases, injected text increases saved reasoning counts, and already-finished first-pass answers can exceed 1,024 tokens. The audit found 1,957 such answers (0.36%); maximum saved reasoning/answer counts were 2,088/1,844. Forced thinking termination occurred in 52.34% of training rows.

| Component | Raw/retained traces | Full-template tokens retained |
|---|---:|---:|
| Previously collected corpus | 196,816 retained | 434,839,110 |
| New worker 0 | 180,000 → 176,421 | 387,793,453 |
| New worker 1 | 180,000 → 176,389 | 387,557,415 |
| **Combined training file** | **549,626** | **1,210,189,978** |
| **Fresh validation set** | **5,000 separate prompts** | Generated independently during each evaluation |

Average full training sequence length is 2,201.84 tokens; maximum is 4,773, below the training limit of 8,192. Recorded generation metadata totals **750.28M reasoning tokens and 338.74M answer tokens**. Full-template tokens also include prompts and delimiters; 1.210B is therefore not the count of supervised answer tokens or sampled training anchors.

The new-data cleaner removed **7,190/360,000 rows (2.00%)** using normalized prompt/reasoning deduplication and repetitive-reasoning rules: a repeated 20-word shingle at least eight times (five for forced-stop traces), a long sentence at least three times, or a long paragraph at least three times. Answer repetition was audited separately because repeated code can be legitimate.

The combined corpus has zero duplicate IDs or empty user/reasoning/answer fields. Residual issues remain: **1,511 normalized-prompt duplicate extra rows and 2,188 repetitive-reasoning flags, all in the inherited component**, which was not re-pruned with the new policy. There are also 156 answer-repetition heuristic flags. These are a small minority, not evidence of perfect quality. The audit additionally found 131 rows without a recorded final server `stop` status despite the generation pipeline's textual end-marker check; these warrant review, not automatic classification as truncated.

The **fresh 5K has zero ID and normalized-prompt overlap with training**. We did not use the older 5K set, which had overlaps with the enlarged corpus. Exact matching does not rule out semantic near-duplicates or target pretraining contamination. Sources: [token inventory](evidence/corpus-token-audit.json), [data audit](evidence/quality-audit-20260912.json), [selection manifest](evidence/prepared.json).

## 2. Actual training examples

Rows store `id`, `conversations: [user, assistant]`, source metadata, and generation metadata. The assistant has separate `reasoning_content` and final `content`. Below are **abridged excerpts from real rows**, not synthetic replacements. They illustrate coherent task/answer structure, not a dataset-wide correctness guarantee.

**Combined-file line 1**, ID `36c38d2e15f7d562d892ced683450912`, inherited data: user asks for the longest consecutive integer sequence, with `[100, 4, 200, 1, 3, 2] → 4`. Recorded lengths: 557 reasoning + 632 answer tokens; natural completion. Reasoning begins: “We are going to solve the problem by using a set for O(1) lookups.” The answer's code is:

```python
def longest_consecutive(nums):
    num_set = set(nums)
    max_length = 0
    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            max_length = max(max_length, current_length)
    return max_length
```

**Combined-file line 196,818**, ID `feb4a542a2c8a70ae99ca1c7da1989e6`, new worker 0: user asks to aggregate `(account_id, amount)` transactions into a balance dictionary, returning `{}` for empty input. Recorded lengths: 284 reasoning + 461 answer tokens; natural completion. The final answer includes:

```python
def calculate_balance(transactions):
    balance_dict = {}
    for account_id, amount in transactions:
        if account_id in balance_dict:
            balance_dict[account_id] += amount
        else:
            balance_dict[account_id] = amount
    return balance_dict
```

Whitespace-only blank lines above are omitted. [Six full source rows](evidence/training-examples.jsonl) include the first natural and forced-stop example from each component; [line-number metadata](evidence/training-examples.json) identifies their origins. These are illustrative, deterministically selected examples, not a random correctness audit.

## 3. Proportional architecture scaling

| Label | Trainable parameters | Target taps K | Draft layers L | d_model | d_ff | Q/KV heads |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 530,429,057 | 3 | 3 | 3,200 | 10,880 | 25 / 5 |
| 1B | 915,384,833 | 4 | 4 | 3,840 | 13,056 | 30 / 6 |
| 2B | 1,876,167,169 | 5 | 5 | 5,120 | 17,408 | 40 / 8 |
| 4B | 3,926,503,681 | 7 | 7 | 6,400 | 21,760 | 50 / 10 |
| 8B | 8,302,654,593 | 9 | 9 | 8,320 | 28,288 | 65 / 13 |

The reference has K=L=5, d_model=5,120, d_ff=17,408. We scale depth, width, FFN, and taps together, keeping **d_ff/d_model=3.4**, head dimension **128**, and Q/KV ratio **5:1**. Width/depth stays within 10.72% of the reference aspect ratio after integer rounding. The seven-token draft block and Markov rank 256 stay fixed.

K denotes target hidden-state taps; L denotes draft transformer blocks. **Neither denotes diffusion iterations. K=L is our scaling choice**, not a requirement that each block consume one distinct target layer. Taps span target indices 1–92, and a learned projection fuses their concatenated features into draft-width context. Target width stays **4,096** for every model. Input/output adapters bridge target and draft widths; each draft block contains RMSNorm, GQA with RoPE and the block/context attention mask, residual connections, and a SwiGLU FFN.

Parameter counts include draft adapters and Markov/confidence heads but exclude **1,244,659,712 frozen target embedding/output-head parameters** shared with the target. Exact dimensions and tap indices: [architecture manifest](evidence/architecture-plan.json).

## 4. Training setup, steps, and loss history

All five models trained from scratch for **one epoch / 68,703 optimizer steps**, consuming 549,624 rows; two tail rows were dropped for global-batch alignment. Each run used eight GPUs, BF16/FSDP, microbatch 1 per rank, no gradient accumulation, and global optimizer batch **8**. Two eight-H200 machines supported parallel runs, with later resumptions/migrations.

SpecForge captures frozen-target hidden features and distribution supervision by prefilling the saved traces through SGLang; it does **not generate new answers during training**. Capture batch was increased during the campaign, with 64 selected after tests including 96; capture batching is separate from optimizer batch size.

Optimization: AdamW with FP32 master weights, peak LR **1e-4**, **1.5% warmup (1,030 steps)** followed by cosine decay to zero, weight decay 0, gradient clipping 1, seed 42. User/problem tokens are masked out; assistant reasoning and answer spans supply supervision. The objective is **0.1 × CE + 0.9 × distribution L1 + confidence loss**, so it is not ordinary language-model NLL/perplexity.

| Model | Loss near step 1K | 10K | 30K | 50K | Final window, through 68,703 |
|---|---:|---:|---:|---:|---:|
| 0.5B | 2.307 | 1.101 | 0.875 | 0.819 | 0.776 |
| 1B | 2.296 | 1.048 | 0.820 | 0.760 | 0.717 |
| 2B | 2.322 | 1.013 | 0.783 | 0.720 | 0.675 |
| 4B | 2.353 | 0.974 | 0.745 | 0.675 | 0.630 |
| 8B | 2.411 | 0.957 | 0.726 | 0.655 | 0.608 |

Each cell averages the 50 logged minibatch measurements in the preceding 1,000-step window; logging is every 20 steps. Resume histories are stitched from the selected original prefix and final resumed suffix, without duplicate steps. These are **training losses**, not held-out losses. All models improve through the final window, but this does not prove that more data or another epoch would improve validation. [Window summaries/provenance](evidence/loss-history-summary.json); full CSV histories: [0.5B](evidence/0p5b-loss-history.csv), [1B](evidence/1b-loss-history.csv), [2B](evidence/2b-loss-history.csv), [4B](evidence/4b-loss-history.csv), [8B](evidence/8b-loss-history.csv).

The results below evaluate final step-68,703 checkpoints, **not a 10%-interval validation curve**. Training used rolling 1,000-step saves plus final checkpoints.

## 5. Generated results and quality on the fresh 5K

Every model evaluated the same 5,000 full prompts: mean **223.13**, maximum **1,114** prompt tokens, with no prompt truncation. Evaluation used the same template/sampling and bounded-thinking policy as generation: nominal 2,048 thinking + 1,024 answer budget, TP8 target, concurrency **32**, seven draft positions, and radix cache disabled. Outputs can differ across drafters despite matching seed policy; these are not identical teacher-forced continuations.

Acceptance length here is **total generated completion tokens / total target verification calls**, including the bonus token. Draft acceptance is **accepted draft tokens / proposed draft tokens**, excluding that bonus. Both phases are aggregated, weighted by verification calls rather than equally by prompt.

| Model | Acceptance length τ | Draft acceptance | Mean generated tokens | Truncated answers | Repetition-flagged outputs |
|---|---:|---:|---:|---:|---:|
| 0.5B | **4.226** | 46.12% | 2,023.8 | 244 / 5,000 | 102 / 5,000 |
| 1B | **4.439** | 49.15% | 2,024.2 | 245 / 5,000 | 103 / 5,000 |
| 2B | **4.581** | 51.20% | 2,027.7 | 251 / 5,000 | 108 / 5,000 |
| 4B | **4.765** | 53.81% | 2,028.9 | 255 / 5,000 | 110 / 5,000 |
| 8B | **4.854** | 55.09% | 2,033.7 | 250 / 5,000 | 103 / 5,000 |

The audit read **all 25,000 saved responses**. Every response has nonempty reasoning and answer fields and a closing think marker; there are no exact normalized duplicate answers within each model's 5K. Approximately 55% require forced thinking termination. Around **5% hit the answer cap**, and 73–89 answers/model have an unmatched code fence, a heuristic often associated with truncation. Nonempty does not mean correct or complete.

Repetition heuristics flag **2.04–2.20%** of responses. Their acceptance lengths are **3.874–4.545**, lower than each model's overall result. Excluding flagged responses increases τ to **4.237, 4.447, 4.590, 4.774, 4.863**, respectively. Thus detected repetition is **not inflating the aggregate in this sensitivity check**; undetected repetition, domain ease, or other biases remain possible. Headline scores retain every response.

For a concrete generated result, holdout ID `5f7a12685115cddcdca070b005a1e952` asks to double a list recursively, without loops, for up to 1,000 elements. The 0.5B-assisted target output contains:

```python
def calc_new_positions(positions):
    if not positions:
        return []
    if len(positions) == 1:
        return [positions[0] * 2]
    mid = len(positions) // 2
    left = calc_new_positions(positions[:mid])
    right = calc_new_positions(positions[mid:])
    return left + right
```

This trace required forced thinking termination but produced an untruncated answer. Its divide-and-conquer structure avoids linear recursion depth. Two holdout IDs were selected by a deterministic hash, independent of their scores; [all five models' full outputs for those IDs](evidence/holdout-examples.json) are preserved. **Generated code was not executed/graded in this report: no pass@1 or correctness percentage is established.** Full numerical audit: [evaluation evidence](evidence/evaluation-audit.json).

## 6. Acceptance by draft position on the fresh 5K

We recovered these values from the saved per-request accepted-prefix histograms, validating their sums against verification counts and accepted-token totals. Let A be the number of accepted draft tokens in a round. The table reports **P(A ≥ j)**: the fraction of all verification rounds in which draft position j is accepted along with its entire preceding prefix. The bonus token is excluded.

| Model | Position 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | 83.42% | 66.74% | 52.84% | 41.60% | 32.68% | 25.60% | 19.95% |
| 1B | 84.40% | 68.89% | 55.82% | 45.11% | 36.53% | 29.53% | 23.80% |
| 2B | 85.20% | 70.41% | 57.87% | 47.50% | 39.05% | 32.08% | 26.30% |
| 4B | 86.11% | 72.40% | 60.55% | 50.52% | 42.26% | 35.35% | 29.50% |
| 8B | 86.65% | 73.42% | 61.85% | 52.00% | 43.78% | 36.89% | 31.03% |

Larger drafters especially improve prefix survival at later positions: full-block acceptance rises from **19.95% to 31.03%**. The separate *conditional* probability of accepting position 7 given acceptance through position 6 rises from **77.94% to 84.11%**; both definitions are saved in the evidence. Prefix survival is not independent per-position prediction accuracy.

The engine histogram counts acceptance before final EOS/output-budget clipping. Consequently, `1 + sum_j P(A ≥ j)` differs slightly (~0.002 tokens) from the headline τ computed from actual completion tokens.

## 7. Final BigCodeBench and LiveCodeBench results

All **10 external evaluations completed successfully at 02:27 UTC on September 18, 2026**, using the same final step-68,703 checkpoints. We evaluated all **1,140 BigCodeBench-Instruct v0.1.4** problems and all **1,055 LiveCodeBench code-generation release_v6** problems for each model. These are external coding-domain sets, distinct from the OpenCode training source—not guarantees of target pretraining non-contamination.

The contract matches the fresh-5K run: Qwen3-thinking-nosystem, full prompts without truncation, temperature 0.6, top-p 0.95, top-k 20, min-p 0, the same two-pass **2,048-token thinking phase / nominal 1,024-token answer budget**, TP8, seven draft positions, and radix cache disabled. Both client concurrency and server `max-running-requests` are **32**; continuous batching makes the instantaneous active batch variable. External seeds use the previously established `external-coding-validation-v1` namespace. The target, serving adapters, and TP8 attention-head padding are unchanged.

| DSpark | OpenCode fresh 5K τ | BigCodeBench τ | LiveCodeBench τ |
|---|---:|---:|---:|
| 0.5B | 4.226 | 3.260 | 3.389 |
| 1B | 4.439 | 3.442 | 3.567 |
| 2B | 4.581 | 3.578 | 3.715 |
| 4B | 4.765 | 3.770 | 3.890 |
| 8B | **4.854** | **3.831** | **3.988** |

Here τ is total generated completion tokens divided by total verification calls, **including the bonus token**, aggregated across both generation phases. It is not pass@1. Larger drafters improve acceptance on all three datasets, while external acceptance is consistently below the in-domain holdout. Doubling from the 4B to 8B size adds only **0.060 tokens on BigCodeBench** and **0.098 on LiveCodeBench**, compared with **0.089 on OpenCode**.

### Output quality and budget caveat

| DSpark | BigCodeBench answers hitting cap | LiveCodeBench answers hitting cap |
|---|---:|---:|
| 0.5B | 6 / 1,140 (0.53%) | 362 / 1,055 (34.31%) |
| 1B | 2 / 1,140 (0.18%) | 347 / 1,055 (32.89%) |
| 2B | 7 / 1,140 (0.61%) | 353 / 1,055 (33.46%) |
| 4B | 5 / 1,140 (0.44%) | 352 / 1,055 (33.36%) |
| 8B | 7 / 1,140 (0.61%) | 336 / 1,055 (31.85%) |

The **roughly one-third LiveCodeBench truncation rate is material**. These scores describe a budget-limited generation workload, not acceptance over unrestricted complete solutions. All answers, including truncated ones, remain in the headline measurement. A larger-answer-budget evaluation should be reported separately rather than mixed into this sweep.

An output audit of the completed **0.5B and 1B runs (4,390 responses)** found relevant, readable code but also concrete mistakes: one BigCodeBench answer returned a dictionary where bytes were requested; one naturally completed LiveCodeBench answer contained invalid Python; some truncated answers ended mid-code or continued deliberation inside comments. Of 3,673 untruncated responses, 3,670 extracted code blocks parsed as Python. **Parsing is not correctness; official benchmark tests were not executed.** This detailed quality audit covers the first two models, not all five; the cap counts above cover all five. See [output inspection and examples](external-output-audit/README.md).

A fresh scan against all 549,626 training rows found zero exact ID or normalized-problem overlap with either external benchmark. Prompt lengths were checked before generation; maximum rendered lengths were 1,348 tokens for BigCodeBench and 1,909 for LiveCodeBench. This scan did not rule out semantic near-duplicates. Full generations remain in NFS `metrics.json` files. [Final external results](evidence/external-benchmark-summary.json) include aggregate metrics and seven-position prefix-acceptance arrays for every model/benchmark pair.

## 8. Measured end-to-end throughput at concurrency 32

Both client concurrency and server `max-running-requests` were **32** for all three datasets. This is a continuous-batching concurrency cap, not a fixed active batch of exactly 32 at every instant.

The metric is **aggregate generated completion tokens / evaluator wall-clock seconds**, summed across initial generation and answer continuation. It includes prompt/continuation prefill, drafting, target verification, request scheduling, and the initial cache flush. It excludes server/model startup, warmup, tokenizer/prompt preparation before the timed workload, and final result-file serialization. Injected think-closing bridge tokens are not counted as generated output. This is workload-level throughput, not per-request decode tokens/s or a latency percentile.

| DSpark | OpenCode fresh 5K (tok/s) | BigCodeBench (tok/s) | LiveCodeBench (tok/s) |
|---|---:|---:|---:|
| 0.5B | 3,274.6 | 2,608.1 | 2,624.0 |
| 1B | 3,414.2 | 2,727.0 | 2,740.9 |
| 2B | 3,487.4 | 2,803.3 | 2,818.8 |
| 4B | **3,555.4** | **2,892.1** | 2,889.6 |
| 8B | 3,548.7 | 2,877.9 | **2,900.4** |

Corresponding complete-workload wall times, excluding server startup:

| DSpark | OpenCode fresh 5K (s) | BigCodeBench (s) | LiveCodeBench (s) |
|---|---:|---:|---:|
| 0.5B | 3,090.16 | 903.04 | 1,077.38 |
| 1B | 2,964.46 | 861.99 | 1,031.96 |
| 2B | 2,907.21 | 836.20 | 1,004.18 |
| 4B | 2,853.34 | 809.13 | 978.69 |
| 8B | 2,865.42 | 819.34 | 975.63 |

**Interpretation:** 4B has the highest observed throughput on OpenCode and BigCodeBench; 8B leads slightly on LiveCodeBench. The 4B/8B throughput differences are below 0.5% on each dataset. With only one timed run per configuration and slightly different generated token counts, we cannot establish a statistically reliable winner. The observed plateau is an inference-cost/acceptance tradeoff, not proof of training saturation.

**These are measured throughputs, not speedup ratios against the target alone.** This sweep has no matched target-only baseline on the same pod/runtime. Older target-only measurements are not used as denominators. A defensible target-relative speedup requires that matched baseline and preferably repeated timing runs.

The timings include the actual TP8 zero-head-padding overhead described below, so they characterize this serving implementation rather than an optimized native-width implementation. All results also retain the existing bounded-output policy, including the substantial LiveCodeBench truncation rate.

Sources: [OpenCode timing and token counters](evidence/evaluation-audit.json), [external benchmark timing and token counters](evidence/external-benchmark-summary.json).

## 9. Interpretation and limitations

This is a **fixed-data, one-epoch model-size sweep**, not an isoFLOPs study. Bigger models attain lower training loss and longer accepted prefixes, with diminishing gains: going from 4B to 8B adds only **0.089 tokens/verification**. That is not proof of saturation, and longer acceptance is not automatically faster inference.

Custom-width evaluation used learned width adapters plus zero attention-head padding for TP8 compatibility: native Q/KV heads 25/5 and 30/6 → 40/8; 50/10 and 65/13 → 80/16; the 2B model stays 40/8. This preserves the intended attention mapping mathematically but adds execution overhead. Therefore these results should **not be treated as a native-cost speedup ranking**. A target-only baseline and matched wall-clock measurements are needed to select the deployment winner.

Other boundaries: no repeated-seed uncertainty or held-out loss curve is included; generated-code correctness is ungraded; and truncation must accompany any quality claim—roughly 5% on the OpenCode holdout, below 1% on BigCodeBench, and 32–34% on LiveCodeBench. The external scores in section 7 are new evaluations of this five-model family, not reused scores from earlier DSpark/Eagle3 experiments.

## 10. Smaller drafters and 4B WSD results — September 20 update

### Models and training branches

The additional runs use the same frozen Qwen3-235B target, 549,626-row / 1.210B-full-template-token corpus, global optimizer batch 8, and final step 68,703 (549,624 rows consumed). The small models extend the same proportional DSpark family; parameter counts exclude frozen target embeddings and output head.

| Model | Trainable parameters | K=L | d_model | d_ff | Q/KV heads | Schedule |
|---|---:|---:|---:|---:|---:|---|
| 0.125B | 114,170,625 | 1 | 1,280 | 4,352 | 10 / 2 | Cosine |
| 0.25B | 202,160,001 | 2 | 1,920 | 6,528 | 15 / 3 | Cosine |
| 4B stable | 3,926,503,681 | 7 | 6,400 | 21,760 | 50 / 10 | WSD, no decay |
| 4B cooldown | 3,926,503,681 | 7 | 6,400 | 21,760 | 50 / 10 | WSD + final cooldown |

FFN/width remains 3.4, head dimension 128, and Q/KV ratio 5:1. Integer depth/width constraints matter at the small end: the one-layer model's width/depth ratio is 25% above the reference, while the two-layer model is 6.25% below. Thus proportional scaling is approximate, not an exact aspect-ratio match at every size.

**Both final 4B WSD branches are preserved and evaluated.** The stable branch warms up for 1,030 updates, then holds LR at **1e-4** through step 68,703. A separate cooldown branch forks from stable step **61,833 (90%)**, then decays LR with cosine from 1e-4 toward zero over the same final **6,870 updates**. This is not a second epoch: both endpoints see the same amount of training data. Running both branches adds 10% extra 4B training work. The original stable trajectory and its tenth checkpoints remain intact.

The new campaign used shared target prefill and Mooncake feature transfer, eventually with two prefill nodes feeding one trainer node. This differs from the original cosine execution layout, so old-cosine versus new-WSD results are not a perfectly isolated LR-only ablation.

### Final acceptance results, side by side

All rows use the same full-prompt evaluation contract: fresh OpenCode **5,000**, BigCodeBench **1,140**, LiveCodeBench **1,055**; concurrency **32**, TP8 BF16, seven draft positions, temperature 0.6, top-p 0.95, top-k 20, and the existing two-pass 2,048-initial-output / nominal 1,024-answer budget. Acceptance length **includes the bonus token** and is aggregated across both phases; it is not code correctness.

| Model / schedule | OpenCode fresh 5K τ | BigCodeBench τ | LiveCodeBench τ |
|---|---:|---:|---:|
| 0.125B / cosine — new | 2.067 | 1.721 | 1.721 |
| 0.25B / cosine — new | 3.541 | 2.652 | 2.760 |
| 0.5B / cosine | 4.226 | 3.260 | 3.389 |
| 1B / cosine | 4.439 | 3.442 | 3.567 |
| 2B / cosine | 4.581 | 3.578 | 3.715 |
| 4B / cosine | 4.765 | 3.770 | 3.890 |
| 4B / WSD stable — new | 4.640 | 3.713 | 3.850 |
| 4B / WSD cooldown — new | 4.802 | 3.821 | 3.950 |
| 8B / cosine | 4.854 | 3.831 | 3.988 |

Cooldown improves over the final stable checkpoint by **+0.162 / +0.107 / +0.100** tokens on OpenCode / BigCodeBench / LiveCodeBench. It exceeds the original 4B cosine result by **+0.037 / +0.050 / +0.060**. These are single-run comparisons without seed-repeat confidence intervals.

### All ten stable WSD checkpoints: fresh-5K validation

Each checkpoint below was evaluated on all **5,000 unique holdout prompts, without cooldown**. Intermediate checkpoints were **not** evaluated on BigCodeBench or LiveCodeBench; those datasets have the final stable and final cooldown results above.

| Training fraction | Step | Traces consumed | OpenCode fresh 5K τ |
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

Acceptance continues to rise, with diminishing gains: the final 20% adds **0.093** tokens. This is an unannealed learning curve, not the best cooled result at each data budget and not proof of saturation. The final cooldown result **4.802** is a separate branch, not an eleventh point on the stable trajectory.

### Throughput and output-budget caveats for the new runs

| Model / schedule | OpenCode tok/s | BigCodeBench tok/s | LiveCodeBench tok/s | Answer-cap hits: OpenCode / BCB / LCB |
|---|---:|---:|---:|---|
| 0.125B / cosine | 1,697.7 | 1,458.7 | 1,413.8 | 245 / 7 / 339 |
| 0.25B / cosine | 2,794.3 | 2,165.7 | 2,184.7 | 247 / 9 / 338 |
| 4B / WSD stable | 3,449.2 | 2,840.0 | 2,843.0 | 245 / 6 / 349 |
| 4B / WSD cooldown | 3,568.4 | 2,925.2 | 2,919.9 | 236 / 7 / 349 |

Cap-hit denominators are 5,000 / 1,140 / 1,055 respectively. All responses, including truncated answers, remain in acceptance results. Roughly one-third of LiveCodeBench answers hit the answer limit; generated code was not execution-graded. The original five-model repetition audit does not establish repetition rates for these additional runs.

Throughputs describe the padded TP8 implementation, **not target-relative speedup or native architecture cost**. Small-model Q/KV heads are padded from 10/2 and 15/3 to 40/8; 4B uses 80/16. No matched target-only baseline or repeated timing runs were added, and evaluations ran at different times/on different pods.

### Validation audit and provenance

All ten retained checkpoint outputs were revalidated: **5,000 unique prompts each and zero summary/raw-metric mismatches**. Cross-node queue locking accidentally duplicated evaluations at steps 6,870, 48,092, and 61,833. Earlier duplicate full outputs may have been overwritten; scalar logs survive. No best-of-repeat selection was applied. The queue was subsequently changed to atomic directory claims and tested across all three nodes.

Sources: [combined comparison and full-precision metrics](../dspark-comparison-20260920/comparison.json), [validated stable curve](../dspark-comparison-20260920/evidence/stable_curve.json), [duplicate-work audit](../dspark-comparison-20260920/evidence/curve_audit.json), and [source paths/checksums](../dspark-comparison-20260920/evidence/source-manifest.json).

NFS run root: `/shared/user/H200/specforge/dspark-shared-prefill-wsd-20260918/dual-prefill-224/`. Final evaluations are under `eval/final-three-datasets/`; the ten-point curve and its audit are under `eval/4b-stable-trajectory-5k/`. Unannealed checkpoints are preserved separately under `checkpoints/4b-wsd/stable/stepN/`.

## Artifact locations

All paths below are on the lor1 shared NFS. This report and its compact evidence are also copied there.

```text
Shared data root:
/shared/public/data/opencode-qwen3-235b-thinking-scaling-1p2b-20260910/
  data/opencode_scaling.combined.cleaned.jsonl
  data/fresh_holdout.source.jsonl

Experiment root:
/shared/user/H200/specforge/dspark-proportional-1p2b-20260912/
  checkpoints/{0p5b,1b,2b,4b,8b}/dspark-proportional-{size}-step68703/
  eval/fresh5k-20260916/{0p5b,1b,2b,4b,8b}/metrics.json
  eval/external-coding-20260917/summary.json
  eval/external-coding-20260917/{size}/{benchmark}/metrics.json
  eval/external-coding-20260917/output-audit-20260918/
  reports/dspark-scaling-20260917/

Frozen target checkpoint:
/shared/public/elr-models/Qwen/Qwen3-235B-A22B-Thinking-2507/
  ddee1c5fb2c5406e8315f8a6d47214c74349e15b/
```

Training-file SHA256: `70048dfd036935a9e0567fd1354142bb81b739f2b427c7201064afcf7ebaf0c6`.
Ordered validation-ID SHA256: `f65949dcaac72ea8fd132e7b1fe40047fa67d9b67f63d8e9f9114a4beec8ffc7`.
SpecForge base revision: `7d5a69386f9cef8240094b96a7b93c1401e889b8`; evaluation SGLang base revision: `acaab22d098594579dc3d2e04ea78474b0a60d7f` with custom-width compatibility changes. [Evidence collector](collect_evidence.py) documents the saved-response audit and loss-history reconstruction.
