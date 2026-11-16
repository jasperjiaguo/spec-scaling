# External generated-output inspection — 2026-09-18

Scope: all 4,390 saved responses from the completed 0.5B and 1B evaluations, plus manual inspection of selected complete/truncated outputs. Inputs, checkpoints, sampling, and the ongoing queue were not modified. No generated code was executed.

| Model | BigCodeBench answer-cap hits | LiveCodeBench answer-cap hits |
|---|---:|---:|
| 0.5B | 6/1,140 (0.53%) | 362/1,055 (34.31%) |
| 1B | 2/1,140 (0.18%) | 347/1,055 (32.89%) |

All responses have nonempty final answers and a closing thinking marker. Of 3,673 untruncated responses, 3,670 extracted code blocks parse as Python; this is syntax only, not runtime correctness. Among truncated LiveCodeBench responses, 107 (0.5B) and 95 (1B) fail the extraction/parse check. Syntactically valid truncated code can still be incomplete—for example, a function ending in comments without a return.

Manual findings:

- **Relevant, plausible outputs:** 0.5B LiveCodeBench `abc336_a` constructs `"L" + "o" * N + "n" + "g"`, matching the Dragon String question. BigCodeBench/428 implements the requested dataframe outer join, feature scaling, and pair plot; no claim that all edge cases pass.
- **Concrete semantic error despite valid syntax:** 0.5B BigCodeBench/544 requests bytes but the answer returns `{'bytes': compressed}`, a dictionary.
- **Concrete syntax error despite natural completion:** 1B LiveCodeBench `3091` contains `del freq[0] if 0 in freq else None`, which Python rejects.
- **Real truncation:** 0.5B LiveCodeBench `abc379_f` ends at `if __name__ == "__`. Other samples spend the final-answer budget continuing deliberation or extensive comments, sometimes never reaching usable code.

Repetition flags: 36/1,140 and 39/1,140 BigCodeBench responses; 2/1,055 and 5/1,055 LiveCodeBench responses. Excluding flagged responses changes acceptance by approximately 0.003–0.004 tokens or less. These heuristics do not capture all rambling or incomplete reasoning.

Interpretation: the outputs are generally on-task, but not universally correct or complete. LiveCodeBench acceptance is a measurement under a restrictive answer budget, not an unrestricted coding-quality result. Preserve the current contract across all five sizes; any larger-budget run should be separate. A code-execution grader is needed for pass@1, and a matched target-only control is needed to investigate whether any errors are specific to speculative decoding.

The JSON summary contains complete counts and aggregate acceptance split by truncation/repetition. Example files contain two deterministically hash-selected responses per populated completion/syntax bucket—not a representative random correctness sample.
