# SWIFT KV-SSD Experimental Branch

This repository is based on SWIFT: On-the-Fly Self-Speculative Decoding for
LLM Inference Acceleration.  The current branch adds experiments around draft
KV retention, masked KV selection, reused draft attention statistics, and
dynamic local controllers.

## What This Branch Adds

- Draft KV compression with configurable retain ratio.
- KV selection by sink + recent + attention-heavy tokens.
- Reuse of previous draft attention statistics to avoid an extra observation
  forward pass.
- Mask-based draft KV mode that hides unselected old context tokens instead of
  physically copying selected KV into a compact cache.
- Local adaptive controllers for retain ratio and skipped attention layers.
- Dynamic-6 cold start: a one-time, task-agnostic repeat-passage warmup that
  seeds the final2 config table before the benchmark stream.

## Main Entry Point

The current evaluation script is:

```bash
bash eval_llama_cosine_mmlu.sh
```

The script supports MMLU, GSM8K, SAMSum, and MT-Bench through `TASK_NAME`:

```bash
TASK_NAME=mmlu bash eval_llama_cosine_mmlu.sh
TASK_NAME=gsm8k bash eval_llama_cosine_mmlu.sh
TASK_NAME=mt_bench bash eval_llama_cosine_mmlu.sh
```

Useful flags:

```bash
RUN_BASELINE=0                # skip vanilla baseline
WRITE_LOG=0                   # do not tee stdout to a .log file
ADAPTIVE_COLD_START=1         # enable dynamic-6 cold start
DRAFT_KV_CACHE_MODE=mask      # default: use attention mask instead of KV copy
DRAFT_KV_SCORE_SOURCE=reuse   # default: reuse previous draft attention scores
```

Example:

```bash
TASK_NAME=gsm8k \
ADAPTIVE_COLD_START=1 \
RUN_BASELINE=0 \
bash eval_llama_cosine_mmlu.sh
```

## KV Selection

The current default is:

```bash
DRAFT_KV_SCORE_SOURCE=reuse
DRAFT_KV_CACHE_MODE=mask
```

For a context of length `full_len`, the draft KV budget is:

```text
keep_len = max(16, int(full_len * retain_ratio))
```

The selected KV tokens are:

1. Sink tokens from the beginning of the context.
2. Recent tokens from the end of the context.
3. Remaining middle tokens ranked by attention score.

When `DRAFT_KV_SCORE_SOURCE=reuse`, the attention score comes from the previous
draft step.  If no score exists yet, such as the first decode step of each
question, the middle-token budget falls back to deterministic even sampling.

When `DRAFT_KV_CACHE_MODE=mask`, the full KV layout is kept and unselected old
context tokens are hidden by an attention mask.  This reduces KV-copy overhead,
but it is not a true sparse-attention kernel by itself.

## Dynamic-6 Cold Start

Dynamic-6 cold start is enabled with:

```bash
ADAPTIVE_COLD_START=1
```

The default mode is:

```bash
ADAPTIVE_COLD_START_MODE=dynamic
ADAPTIVE_COLD_START_PROMPT=auto
```

In dynamic mode, the code runs one task-agnostic repeat-passage prompt before
the real benchmark.  The prompt asks the model to repeat a neutral passage while
the final2 controller is already active, so the warmup seeds the ratio/layer
table from KV retention and layer-skip behavior rather than benchmark-specific
content.

The warmup uses an isolated statistics dictionary.  Only the low-weight
adaptive config table is merged back into the real run.  The warmup KV cache,
global acceptance reference, and generated tokens are not reused by the actual
benchmark stream.

The older fixed-config warmup is still available as an ablation:

```bash
ADAPTIVE_COLD_START_MODE=probe
```

## Important Output Fields

Each `.jsonl` answer file ends with a `__summary__` record.  Useful fields:

- `Mean accepted tokens`
- `Token acceptance rate`
- `Accuracy`
- `Draft KV Cache Mode`
- `Draft KV Score Source`
- `Adaptive Step Config Count`
- `Adaptive Total Switches`
- `Final2 Controller Action Counts`
- `Adaptive Cold Start Mode`
- `Adaptive Cold Start Time Sec`
- `Adaptive Cold Start Mean Accepted Tokens`
- `Adaptive Cold Start Token Acceptance Rate`

For timing analysis, per-sample records include `wall_time`, `new_tokens`, and
`decoding_steps`.

## Environment

Basic setup:

```bash
conda create -n swift python=3.9
conda activate swift
pip install -r requirements.txt
```

Environment audit against the current `swift` conda environment:

- No package listed in `requirements.txt` is missing from `swift`.
- One exact pin differs:
  - `accelerate==0.21.0` is required, but `swift` has `accelerate==1.10.1`.
- `maturin==0.12` is satisfied by installed `maturin==0.12.0`.
- Unpinned requirements currently resolve in `swift` as:
  - `torch==2.8.0`
  - `datasets==3.6.0`
  - `rouge_score==0.1.2`
  - `human_eval==1.0.3`
- The `swift` environment has many extra packages not listed directly in
  `requirements.txt`, mostly transitive dependencies and evaluation tools.
  Notable extras include `evaluate`, `lm_eval`, `peft`, `pandas`,
  `matplotlib`, `tiktoken`, `sacrebleu`, `scikit-learn`, `nltk`, `fire`,
  and `jsonlines`.

To reproduce the audit:

```bash
conda run -n swift python -m pip freeze
conda list -n swift
```

## Original SWIFT Reference

If you use the original SWIFT method, cite:

```bibtex
@inproceedings{xia2025swift,
  title={{SWIFT}: On-the-Fly Self-Speculative Decoding for {LLM} Inference Acceleration},
  author={Heming Xia and Yongqi Li and Jun Zhang and Cunxiao Du and Wenjie Li},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=EKJhH5D5wA}
}
```
