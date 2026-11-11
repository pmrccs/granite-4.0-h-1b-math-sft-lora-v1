# granite-4.0-h-1b-math-sft-lora-v1: evaluation outputs

Raw generations and scores from benchmarking [`pmrccs/granite-4.0-h-1b-math-sft-lora-v1`](https://huggingface.co/pmrccs/granite-4.0-h-1b-math-sft-lora-v1), a LoRA adapter for [`ibm-granite/granite-4.0-h-1b-base`](https://huggingface.co/ibm-granite/granite-4.0-h-1b-base) fine-tuned on short step-by-step math solutions. Every output behind the numbers in the model card is here, for both the fine-tuned model and the unmodified base model, so the results can be inspected, re-scored and challenged.

This repository holds data only. The training and evaluation notebooks are not published; the protocol is described below.

## Links

- Model (adapter): https://huggingface.co/pmrccs/granite-4.0-h-1b-math-sft-lora-v1
- Training data: https://huggingface.co/datasets/pmrccs/math-sft-mix-1024tok-v1
- GGUF files: https://huggingface.co/pmrccs/granite-4.0-h-1b-math-sft-lora-v1-gguf (not benchmarked; the results here are for the unquantized model)

## Results

Zero-shot, greedy decoding, at most 1024 new tokens, one run, using the template the adapter was trained on. The base model got the same template and no examples, so this is a like-for-like comparison, not the base model's best case.

| Benchmark | Problems | Fine-tuned (95% CI) | Base (95% CI) | Difference | Paired sign test |
|---|---|---|---|---|---|
| GSM8K | 1319 | 74.45% (72.03-76.73) | 65.50% (62.90-68.02) | +8.95 | not computed (the intervals do not overlap) |
| SVAMP | 300 | 83.67% (79.06-87.42) | 78.00% (72.97-82.32) | +5.67 | p = 0.033 |
| Minerva Math | 272 | 20.59% (16.21-25.79) | 11.76% (8.46-16.14) | +8.82 | p = 0.0022 |
| MATH-500 | 500 | 45.20% (40.89-49.58) | 43.00% (38.73-47.38) | +2.20 | p = 0.36 (not significant) |

Intervals are 95% Wilson intervals. The sign test is an exact paired test on the problems where exactly one of the two models is right. Outputs that reached the 1024-token limit are scored as wrong (fine-tuned / base): GSM8K 18 / 25, SVAMP 4 / 4, MATH-500 61 / 59, Minerva Math 52 / 99. Limitations and caveats are in the model card.

## Repository layout

```
results/run1/
  gsm8k/          finetuned/  base/    raw generations        scores/
  math500/        finetuned/  base/    raw generations        scores/
  svamp/          finetuned/  base/    raw generations        scores/
  minerva_math/   finetuned/  base/    raw generations        scores/
  sessions/                            settings of each generation session (JSON)
test_runs/                             small smoke tests of the pipeline, not results
```

- `finetuned` is the adapter on the base model; `base` is `ibm-granite/granite-4.0-h-1b-base`.
- Generations are stored as `part-*.jsonl` files (up to 100 results each). Concatenate all parts of a folder to get the full benchmark; every problem appears exactly once per folder.
- `scores/` holds, per variant, a per-problem file (`<variant>-scored-<session>.jsonl`), a summary (`<variant>-summary-<session>.json`) and, where both variants were scored together, a comparison file. Some folders contain more than one scoring session. Use the latest one; the numbers above come from the latest session in each folder. If an earlier GSM8K base score of 2.65% is present, it used a boxed-only rule that was replaced by the rule described below.

### Generation file fields

| Field | Meaning |
|---|---|
| `id` | Problem index within the benchmark's test set (0-based, unique within a folder) |
| `prompt` | The exact prompt sent to the model |
| `gold` | Reference answer from the benchmark (GSM8K: the full reference solution ending in `#### answer`) |
| `output` | The raw model output (cut at a new `### Problem:` section if the model started one) |
| `stop_reason` | `eos` (model finished), `max_tokens` (hit the 1024-token limit), or `stop_string` (started a new problem) |
| `prompt_tokens`, `gen_tokens` | Token counts |
| `n_boxed`, `boxed_nonempty`, `structure_ok` | Number of `\boxed{}` in the output, whether one is non-empty, and whether the output finished normally with a box |
| `variant`, `run`, `gpu`, `batch` | Which model, which generation launch, which GPU and batch produced it |
| `batch_size`, `batch_prefill_s`, `batch_decode_s`, `batch_steps`, `batch_prompt_tps`, `batch_gen_tps` | Timing of the batch the problem was in (see the note below) |
| `subject`, `level`, `unique_id` | MATH-500 only |
| `bench` | SVAMP and Minerva Math only |

Timing note: speeds are per batch, not per problem. The first batch each GPU processed in a session includes a one-time GPU kernel compile and is much slower (one MATH-500 batch shows a prefill time of 431 s); exclude each GPU's first batch when averaging speeds.

### Scored file fields

Each row has `id`, `gold`, `pred` (the extracted answer), `source` (where the answer was taken from: `boxed`, `answer_line`, `last_number` or `none`) and `correct`, plus `stop_reason`, `n_boxed`, `structure_ok` and `gen_tokens`. MATH-500 and Minerva Math files also record `correct_mv` (Math-Verify) and `correct_str` (string match); the exact set of columns differs slightly between benchmarks.

## Evaluation protocol

- **Data:** GSM8K test (1319; `openai/gsm8k`, config `main`), MATH-500 (`HuggingFaceH4/MATH-500`), SVAMP test split (300; `ChilleD/SVAMP`; the problem is the Body plus the Question), Minerva Math (272; `zkshan2002/minerva_math`, which per its dataset card is a copy of `math-ai/minervamath` with extra columns removed).
- **Prompt:** `### Problem:\n{problem}\n\n### Solution:\n`, zero-shot, for both models.
- **Generation:** greedy, at most 1024 new tokens, fp16, batched (left-padded), 2 x Tesla T4. Generation also stopped if the model began a new `### Problem:` section. Software: transformers 5.17.0, peft 0.21.2, torch 2.11.0, mamba-ssm 2.3.1, causal-conv1d 1.7.0. The adapter was merged into the base weights in memory before generation.
- **Answer extraction:** the content of the last closed `\boxed{}`. If an output has no box, the first number after an "Answer:" line is used, then the last number in the text. Outputs cut off at the token limit count only if they contain a closed box.
- **Grading:** GSM8K and SVAMP compare numbers (tolerance 1e-6). MATH-500 and Minerva Math count an answer as correct if Math-Verify 0.9.0 judges it equivalent to the reference or a conservative normalized string match agrees (the normalization strips `\left`/`\right`, `\text{}`, spaces, `$`, `%`, degree signs and thousands commas, and rewrites `\dfrac` as `\frac`).
- **Note on grading sensitivity:** on Minerva Math the fine-tuned model scores 20.59% with this rule, but plain normalized string matching alone credits 27 of 272 problems (about 10%). Compare numbers only with results graded the same way.

## Using the data

The snippets below are minimal and were not run verbatim.

```python
import glob, json
import pandas as pd

def load(folder):
    rows = [json.loads(line) for f in sorted(glob.glob(f"{folder}/part-*.jsonl")) for line in open(f)]
    return pd.DataFrame(rows)

df = load("results/run1/gsm8k/finetuned")
print(len(df), df["id"].is_unique)                     # 1319 True
print(df["stop_reason"].value_counts())

latest = sorted(glob.glob("results/run1/gsm8k/scores/finetuned-scored-*.jsonl"))[-1]
scored = pd.read_json(latest, lines=True)
print(scored["correct"].mean())                        # accuracy as reported above
```

To re-score with your own rules, take `output` and `gold` from the generation files. To compare the two models problem by problem, join the `finetuned` and `base` files on `id`.

## Caveats

- Single greedy run per model and benchmark, no repeated seeds. Results may differ slightly with other hardware, precision, batching or library versions.
- Zero-shot with the training template only. Published figures for these benchmarks usually use few-shot or chat prompts and longer token limits, so they are not directly comparable.
- The training data was decontaminated against these test sets by 10-word n-gram overlap, which does not detect paraphrases, and roughly a quarter of the training mix is derived from the GSM8K training set. See the model card.

## License

The generated outputs and scores are released under CC-BY-NC-4.0 (non-commercial), matching the model. The `prompt` and `gold` fields contain problems and reference answers taken from the benchmark datasets listed above; those remain under their original licenses and terms, which you should check before reuse.

## Citation

```bibtex
@misc{pmrccs2026granite_math_lora,
  title        = {granite-4.0-h-1b-math-sft-lora-v1: a LoRA adapter for short chain-of-thought math},
  author       = {pmrccs},
  year         = {2026},
  howpublished = {https://huggingface.co/pmrccs/granite-4.0-h-1b-math-sft-lora-v1}
}
```
