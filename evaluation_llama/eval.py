"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import logging
import os
import time
import torch
import random
import numpy as np
import re
from decimal import Decimal, InvalidOperation

from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from human_eval.data import read_problems


def seed_everything(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_adaptive_step_config_stats(config_stats):
    rows = []
    for key, entry in (config_stats or {}).items():
        step_count = int(entry.get("step_count", 0))
        accepted_tokens = int(entry.get("accepted_tokens", 0))
        drafted_tokens = int(entry.get("drafted_tokens", 0))
        mean_accepted_std = 0.0
        if step_count > 1:
            mean_accepted_std = (max(0.0, float(entry.get("mean_accepted_step_m2", 0.0)) / (step_count - 1))) ** 0.5
        rows.append({
            "config_key": key,
            "current_ratio": float(entry.get("current_ratio", 0.0)),
            "current_attn_skip_count": int(entry.get("current_attn_skip_count", 0)),
            "step_count": step_count,
            "accepted_tokens": accepted_tokens,
            "drafted_tokens": drafted_tokens,
            "mean_accepted_tokens": (
                (accepted_tokens + step_count) / step_count if step_count > 0 else 0.0
            ),
            "mean_accepted_tokens_std": mean_accepted_std,
            "mean_accepted_draft_tokens": (
                accepted_tokens / step_count if step_count > 0 else 0.0
            ),
            "token_acceptance_rate": (
                accepted_tokens / drafted_tokens if drafted_tokens > 0 else 0.0
            ),
        })
    rows.sort(key=lambda row: (row["current_ratio"], row["current_attn_skip_count"]))
    return rows


def set_flops_trace_context(runtime_statistics, question_idx, question_id=None, turn_idx=None):
    if not runtime_statistics or not runtime_statistics.get("flops_trace_enabled", False):
        return
    runtime_statistics["flops_trace_phase"] = "eval"
    runtime_statistics["flops_trace_question_index"] = int(question_idx)
    runtime_statistics["flops_trace_question_id"] = (
        int(question_id)
        if isinstance(question_id, (int, np.integer))
        else (str(question_id) if question_id is not None else int(question_idx))
    )
    runtime_statistics["flops_trace_turn_index"] = (
        int(turn_idx)
        if turn_idx is not None
        else None
    )


def add_flops_trace_summary(summary, runtime_statistics):
    if not runtime_statistics or not runtime_statistics.get("flops_trace_enabled", False):
        return

    components = ("prefill", "draft", "verify", "safe_commit")
    generated_tokens = int(runtime_statistics.get("flops_trace_generated_token_count", 0))
    summary["FLOPs Trace Enabled"] = True
    summary["FLOPs Trace File"] = runtime_statistics.get("flops_trace_file")
    summary["FLOPs Trace Samples"] = int(runtime_statistics.get("flops_trace_sample_count", 0))
    summary["FLOPs Trace Steps"] = int(runtime_statistics.get("flops_trace_step_count", 0))
    summary["FLOPs Trace Generated Tokens"] = generated_tokens
    summary["FLOPs Trace Draft Tokens"] = int(runtime_statistics.get("flops_trace_draft_token_count", 0))
    summary["FLOPs Trace Accepted Draft Tokens"] = int(runtime_statistics.get("flops_trace_accepted_draft_token_count", 0))
    summary["FLOPs Trace Verify Approx Steps"] = int(runtime_statistics.get("flops_trace_verify_approx_steps", 0))

    for mode in ("logical", "physical"):
        core_total = 0
        core_plus_lm_head_total = 0
        cold_core_total = 0
        cold_core_plus_lm_head_total = 0
        breakdown = {}
        cold_breakdown = {}
        for component in components:
            core_key = f"flops_estimated_{component}_{mode}_core_sum"
            total_key = f"flops_estimated_{component}_{mode}_core_plus_lm_head_sum"
            core_value = int(runtime_statistics.get(core_key, 0))
            total_value = int(runtime_statistics.get(total_key, core_value))
            core_total += core_value
            core_plus_lm_head_total += total_value
            breakdown[component] = {
                "core": core_value,
                "core_plus_lm_head": total_value,
            }
            cold_core_key = f"flops_cold_start_estimated_{component}_{mode}_core_sum"
            cold_total_key = f"flops_cold_start_estimated_{component}_{mode}_core_plus_lm_head_sum"
            cold_core_value = int(runtime_statistics.get(cold_core_key, 0))
            cold_total_value = int(runtime_statistics.get(cold_total_key, cold_core_value))
            cold_core_total += cold_core_value
            cold_core_plus_lm_head_total += cold_total_value
            cold_breakdown[component] = {
                "core": cold_core_value,
                "core_plus_lm_head": cold_total_value,
            }
        title_mode = mode.capitalize()
        summary[f"Estimated {title_mode} Core FLOPs"] = core_total
        summary[f"Estimated {title_mode} Core+LMHead FLOPs"] = core_plus_lm_head_total
        summary[f"Estimated {title_mode} FLOPs Breakdown"] = breakdown
        if cold_core_total > 0 or cold_core_plus_lm_head_total > 0:
            summary[f"Cold Start Estimated {title_mode} Core FLOPs"] = cold_core_total
            summary[f"Cold Start Estimated {title_mode} Core+LMHead FLOPs"] = cold_core_plus_lm_head_total
            summary[f"Cold Start Estimated {title_mode} FLOPs Breakdown"] = cold_breakdown
            summary[f"Total Estimated {title_mode} Core FLOPs Including Cold Start"] = (
                core_total + cold_core_total
            )
            summary[f"Total Estimated {title_mode} Core+LMHead FLOPs Including Cold Start"] = (
                core_plus_lm_head_total + cold_core_plus_lm_head_total
            )
        if generated_tokens > 0:
            summary[f"Estimated {title_mode} Core FLOPs / Output Token"] = core_total / generated_tokens
            summary[f"Estimated {title_mode} Core+LMHead FLOPs / Output Token"] = (
                core_plus_lm_head_total / generated_tokens
            )

    step_count = int(runtime_statistics.get("flops_trace_step_count", 0))
    if step_count > 0:
        summary["FLOPs Trace Avg Draft KV Visible Len"] = (
            float(runtime_statistics.get("flops_trace_draft_kv_visible_len_sum", 0)) / step_count
        )
        summary["FLOPs Trace Avg Draft KV Physical Len"] = (
            float(runtime_statistics.get("flops_trace_draft_kv_physical_len_sum", 0)) / step_count
        )
        summary["FLOPs Trace Avg Verify KV Visible Len"] = (
            float(runtime_statistics.get("flops_trace_verify_kv_visible_len_sum", 0)) / step_count
        )
        summary["FLOPs Trace Avg Verify KV Physical Len"] = (
            float(runtime_statistics.get("flops_trace_verify_kv_physical_len_sum", 0)) / step_count
        )
        summary["FLOPs Trace Avg Draft Attn Skip Count"] = (
            float(runtime_statistics.get("flops_trace_draft_attn_skip_count_sum", 0)) / step_count
        )
        summary["FLOPs Trace Avg Draft MLP Skip Count"] = (
            float(runtime_statistics.get("flops_trace_draft_mlp_skip_count_sum", 0)) / step_count
        )


def normalize_task_name(task_name):
    name = task_name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "trivia_qa": "triviaqa",
        "naturalquestions": "natural_questions",
        "natural_question": "natural_questions",
        "nq": "natural_questions",
        "nq_open": "natural_questions",
        "sam_sum": "samsum",
        "sam_sum_dialogue": "samsum",
        "samsum_dialogue": "samsum",
        "longgsm8k": "long_gsm8k",
        "longmmlu": "long_mmlu",
    }
    return aliases.get(name, name)


def is_qa_task(task_name):
    return normalize_task_name(task_name) in {"triviaqa", "natural_questions"}


LONGGEN_BASE_TASKS = {
    "long_gsm8k": "gsm8k",
    "long_mmlu": "mmlu",
}

LONGGEN_DEFAULT_BATCH_SIZE = {
    "long_gsm8k": 30,
    "long_mmlu": 30,
}

LONGGEN_DEFAULT_NUM_SHOTS = {
    "long_gsm8k": 8,
    "long_mmlu": 5,
}

LONGGEN_SYSTEM_PROMPT = (
    "Answer each question step by step, adhering to the format shown in the examples provided. "
    "Start each response with 'Answer_' followed by the question number, and introduce the final "
    "response in each block with 'The answer is'. Do not repeat the question. Ensure that you "
    "respond to all the questions presented, regardless of their number."
)


def is_longgen_task(task_name):
    return normalize_task_name(task_name) in LONGGEN_BASE_TASKS


def get_longgen_base_task(task_name):
    return LONGGEN_BASE_TASKS[normalize_task_name(task_name)]


def get_longgen_env_int(task_name, setting_name, default_value):
    base_task = get_longgen_base_task(task_name).upper()
    env_names = [
        f"LONGGEN_{base_task}_{setting_name}",
        f"LONGGEN_{setting_name}",
    ]
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value is None or value == "":
            continue
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{env_name} must be a positive integer.")
        return parsed
    return default_value


def extract_gsm8k_gold(answer_text):
    # Standard GSM8K answer format usually ends with "#### 72"
    if "####" in answer_text:
        answer_text = answer_text.split("####")[-1]
    answer_text = answer_text.strip().replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    return matches[-1] if matches else None


GSM8K_FINAL_ANSWER_RE = re.compile(
    r"\bfinal\s+answer\s*[:：]\s*"
    r"(?:the\s+final\s+answer\s+is\s*)?"
    r"(?:the\s+answer\s+is\s*)?"
    r"\$?\s*(?P<number>-?\d[\d,]*(?:\.\d+)?)",
    flags=re.IGNORECASE,
)

GSM8K_ANSWER_RE = re.compile(
    r"\bthe\s+answer\s+is\s*\$?\s*(?P<number>-?\d[\d,]*(?:\.\d+)?)",
    flags=re.IGNORECASE,
)

GSM8K_STOP_PATTERNS = [
    # Do not stop on end-of-buffer after a number during incremental decoding:
    # a SWIFT step can end mid-answer, e.g. "$143" before the next token "0".
    r"\bthe\s+answer\s+is\s+\$?-?\d[\d,]*(?:\.\d+)?\s*(?:\.(?!\d)|\n)",
    r"\bfinal\s+answer\s*[:：]\s*(?:the\s+answer\s+is\s*)?\$?-?\d[\d,]*(?:\.\d+)?\s*(?:\.(?!\d)|\n)",
    r"<\|(?:eot_id|end_of_text|start_header_id|end_header_id)\|>",
]

GSM8K_TRIM_PATTERNS = [
    r"<\|[^>]+\|>",
]

GSM8K_FINAL_SENTENCE_RE = re.compile(
    r"(?:\bthe\s+answer\s+is\s+\$?-?\d[\d,]*(?:\.\d+)?\s*\.|"
    r"\bfinal\s+answer\s*[:：]\s*(?:the\s+answer\s+is\s*)?\$?-?\d[\d,]*(?:\.\d+)?\s*\.)",
    flags=re.IGNORECASE,
)


def clean_gsm8k_output(output_text):
    cleaned = output_text.strip()

    first_match = None
    for pattern in GSM8K_TRIM_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
        if match and (first_match is None or match.start() < first_match.start()):
            first_match = match
    if first_match:
        cleaned = cleaned[: first_match.start()].strip()

    final_match = GSM8K_FINAL_SENTENCE_RE.search(cleaned)
    if final_match:
        cleaned = cleaned[: final_match.end()].strip()

    cleaned = re.sub(r"<\|[^>]+\|>", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def normalize_number_text(number_text):
    return number_text.replace(",", "")


def normalize_gsm8k_answer_number(number_text):
    if number_text is None:
        return None
    normalized = normalize_number_text(str(number_text).strip())
    try:
        canonical = format(Decimal(normalized).normalize(), "f")
    except InvalidOperation:
        return normalized
    return "0" if canonical == "-0" else canonical


def extract_gsm8k_pred(output_text):
    text = output_text.strip().replace(",", "")
    if "####" in text:
        text = text.split("####")[-1]

    answer_match = GSM8K_ANSWER_RE.search(output_text)
    if answer_match:
        return normalize_number_text(answer_match.group("number"))

    final_match = GSM8K_FINAL_ANSWER_RE.search(output_text)
    if final_match:
        return normalize_number_text(final_match.group("number"))

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else None


def extract_mmlu_gold(example):
    ans = example["answer"]
    if isinstance(ans, str):
        ans = ans.strip().upper()
        if ans in {"A", "B", "C", "D"}:
            return ans
        if ans.isdigit():
            idx = int(ans)
            if 0 <= idx <= 3:
                return "ABCD"[idx]
        return None
    if isinstance(ans, (int, np.integer)):
        if 0 <= int(ans) <= 3:
            return "ABCD"[int(ans)]
    return None


def extract_mmlu_pred(output_text):
    text = output_text.strip().upper()

    first_answer = re.search(r"^\s*(?:ANSWER\s*[:：]\s*)?([ABCD])(?:\b|[\.\):])", text)
    if first_answer:
        return first_answer.group(1)

    patterns = [
        r"ANSWER\s*[:：]\s*([ABCD])\b",
        r"THE ANSWER IS\s*([ABCD])\b",
        r"\bOPTION\s*([ABCD])\b",
        r"\bCHOICE\s*([ABCD])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]

    matches = re.findall(r"\b([ABCD])\b", text)
    return matches[0] if matches else None


MMLU_TRIM_PATTERNS = [
    r"\bthe following is a multiple choice question\b",
    r"\n\s*question\s*[:：]",
    r"\.{2,}\s*(?:more|less)\b",
    r"<\|[^>]*header_id\|>",
]


def clean_mmlu_output(output_text):
    cleaned = output_text.strip()
    first_match = None
    for pattern in MMLU_TRIM_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
        if match and (first_match is None or match.start() < first_match.start()):
            first_match = match
    if first_match:
        cleaned = cleaned[: first_match.start()].strip()
    return cleaned


LONGGEN_ANSWER_MARKER_RE = re.compile(
    r"(?:^|\n)\s*Answer[_\s-]*(\d+)\s*[:：]?\s*",
    flags=re.IGNORECASE,
)


def format_longgen_gsm8k_answer(answer_text):
    rationale = str(answer_text)
    if "####" in rationale:
        rationale = rationale.split("####")[0]
    rationale = rationale.strip()
    gold = extract_gsm8k_gold(answer_text)
    if gold is not None:
        if rationale:
            return f"{rationale}\nThe answer is {gold}."
        return f"The answer is {gold}."
    return rationale


def format_longgen_mmlu_question(example, question_number):
    choices = example["choices"]
    return (
        f"Question_{question_number}:\n"
        f"{str(example['question']).strip()}\n"
        f"(A) {choices[0]}\n"
        f"(B) {choices[1]}\n"
        f"(C) {choices[2]}\n"
        f"(D) {choices[3]}"
    )


def format_longgen_gsm8k_question(example, question_number):
    return f"Question_{question_number}:\n{str(example['question']).strip()}"


def format_longgen_examples(base_task, examples):
    question_chunks = []
    answer_chunks = []
    for idx, example in enumerate(examples, start=1):
        if base_task == "gsm8k":
            question_chunks.append(format_longgen_gsm8k_question(example, idx))
            answer_text = format_longgen_gsm8k_answer(example["answer"])
        elif base_task == "mmlu":
            question_chunks.append(format_longgen_mmlu_question(example, idx))
            answer_text = f"The answer is {extract_mmlu_gold(example)}."
        else:
            raise ValueError(f"Unsupported LongGenBench base task: {base_task}")
        answer_chunks.append(f"Answer_{idx}:\n{answer_text}")
    return "\n\n".join(question_chunks + answer_chunks).strip()


def format_longgen_questions(base_task, examples):
    question_chunks = []
    for idx, example in enumerate(examples, start=1):
        if base_task == "gsm8k":
            question_chunks.append(format_longgen_gsm8k_question(example, idx))
        elif base_task == "mmlu":
            question_chunks.append(format_longgen_mmlu_question(example, idx))
        else:
            raise ValueError(f"Unsupported LongGenBench base task: {base_task}")
    return "\n\n".join(question_chunks).strip()


def make_longgen_batches(task_name, examples):
    batch_size = get_longgen_env_int(
        task_name,
        "BATCH_SIZE",
        LONGGEN_DEFAULT_BATCH_SIZE[normalize_task_name(task_name)],
    )
    batches = []
    for start in range(0, len(examples), batch_size):
        items = examples[start:start + batch_size]
        if not items:
            continue
        batches.append({
            "longgen_task": normalize_task_name(task_name),
            "base_task": get_longgen_base_task(task_name),
            "batch_index": len(batches),
            "batch_start": start,
            "items": items,
        })
    return batches


def split_longgen_answer_blocks(output_text, expected_count):
    blocks = [""] * expected_count
    matches = list(LONGGEN_ANSWER_MARKER_RE.finditer(output_text))
    if matches:
        numbered_blocks = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(output_text)
            numbered_blocks.append((int(match.group(1)), output_text[start:end].strip()))

        numbers = [number for number, _content in numbered_blocks]
        use_marker_numbers = numbers and min(numbers) <= 2
        if use_marker_numbers:
            overflow_blocks = []
            for number, content in numbered_blocks:
                if 1 <= number <= expected_count and not blocks[number - 1]:
                    blocks[number - 1] = content
                else:
                    overflow_blocks.append(content)
            overflow_iter = iter(overflow_blocks)
            for idx, block in enumerate(blocks):
                if not block:
                    blocks[idx] = next(overflow_iter, "")
        else:
            for idx, (_number, content) in enumerate(numbered_blocks[:expected_count]):
                blocks[idx] = content
        return blocks

    chunks = [chunk.strip() for chunk in re.split(r"\n{2,}", output_text.strip()) if chunk.strip()]
    for idx, chunk in enumerate(chunks[:expected_count]):
        blocks[idx] = chunk
    return blocks


def extract_longgen_mmlu_pred(output_text):
    text = output_text.strip().upper()
    patterns = [
        r"THE\s+ANSWER\s+IS\s*\(?\s*([ABCD])\s*\)?(?:\s*[\.\),:]|\b|$)",
        r"FINAL\s+ANSWER\s*[:：]\s*\(?\s*([ABCD])\s*\)?(?:\s*[\.\),:]|\b|$)",
        r"ANSWER\s*[:：]\s*\(?\s*([ABCD])\s*\)?(?:\s*[\.\),:]|\b|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    return extract_mmlu_pred(output_text)


def score_longgen_batch(task_name, batch, output_text):
    base_task = get_longgen_base_task(task_name)
    items = batch["items"]
    blocks = split_longgen_answer_blocks(output_text, len(items))
    results = []
    correct_count = 0

    for idx, (item, block) in enumerate(zip(items, blocks), start=1):
        if base_task == "gsm8k":
            gold = extract_gsm8k_gold(item["answer"])
            pred = extract_gsm8k_pred(block)
            normalized_gold = normalize_gsm8k_answer_number(gold)
            normalized_pred = normalize_gsm8k_answer_number(pred)
            correct = (
                normalized_pred is not None
                and normalized_gold is not None
                and normalized_pred == normalized_gold
            )
            result = {
                "question_number": idx,
                "question": item["question"],
                "gold_answer": gold,
                "pred_answer": pred,
                "answer_text": block,
                "correct": correct,
            }
        elif base_task == "mmlu":
            gold = extract_mmlu_gold(item)
            pred = extract_longgen_mmlu_pred(block)
            correct = pred is not None and gold is not None and pred == gold
            result = {
                "question_number": idx,
                "question": item["question"],
                "choices_text": item["choices"],
                "subject": item.get("subject", None),
                "gold_answer": gold,
                "pred_answer": pred,
                "answer_text": block,
                "correct": correct,
            }
        else:
            raise ValueError(f"Unsupported LongGenBench base task: {base_task}")

        correct_count += int(correct)
        results.append(result)

    return results, correct_count


SAMSUM_STOP_PATTERNS = [
    r"<\|(?:eot_id|end_of_text|start_header_id|end_header_id)\|>",
]

MT_BENCH_STOP_PATTERNS = [
    r"<\|(?:eot_id|end_of_text|start_header_id|end_header_id)\|>",
]

MT_BENCH_TRIM_PATTERNS = MT_BENCH_STOP_PATTERNS + [
    r"<\|[^>]+\|>",
]

SAMSUM_TRIM_PATTERNS = SAMSUM_STOP_PATTERNS + [
    r"<\|[^>]+\|>",
    r"\n\s*(?:dialogue|user|assistant)\s*[:：]",
]

SAMSUM_PREFIX_PATTERNS = [
    r"^(?:sure[,.]?\s*)?here\s+(?:is|is\s+a|is\s+the|is\s+an)\s+(?:concise\s+)?summary(?:\s+of\s+the\s+dialogue)?\s*[:：-]\s*",
    r"^(?:sure[,.]?\s*)?(?:a\s+)?(?:concise\s+)?summary(?:\s+of\s+the\s+dialogue)?\s*[:：-]\s*",
]


def clean_samsum_output(output_text):
    cleaned = output_text.strip()
    first_match = None
    for pattern in SAMSUM_TRIM_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
        if match and (first_match is None or match.start() < first_match.start()):
            first_match = match
    if first_match:
        cleaned = cleaned[: first_match.start()].strip()

    cleaned = re.sub(r"<\|[^>]+\|>", " ", cleaned)

    previous = None
    while previous != cleaned:
        previous = cleaned
        for pattern in SAMSUM_PREFIX_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_mt_bench_output(output_text):
    cleaned = output_text.strip()
    first_match = None
    for pattern in MT_BENCH_TRIM_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
        if match and (first_match is None or match.start() < first_match.start()):
            first_match = match
    if first_match:
        cleaned = cleaned[: first_match.start()].strip()
    cleaned = re.sub(r"<\|[^>]+\|>", " ", cleaned)
    return cleaned.strip()


def get_generation_stop_config(task_name):
    task_name = normalize_task_name(task_name)
    if task_name == "mmlu":
        return {
            "patterns": MMLU_TRIM_PATTERNS,
            "min_chars_before_match": 0,
        }
    if task_name == "samsum":
        return {
            "patterns": SAMSUM_STOP_PATTERNS,
            "min_chars_before_match": 0,
        }
    if task_name == "gsm8k":
        return {
            "patterns": GSM8K_STOP_PATTERNS,
            "min_chars_before_match": 0,
        }
    if task_name == "mt_bench":
        return {
            "patterns": MT_BENCH_STOP_PATTERNS,
            "min_chars_before_match": 0,
        }
    if is_qa_task(task_name):
        return {
            "patterns": [
                r"\bquestion\s*[:：]",
                r"\.{2,}\s*(?:more|less)\b",
                r"\b(?:read|show)\s+more\b",
                r"\bshow\s+less\b",
                r"\bback\s+to\s+the\s+list\b",
                r"<\|[^>]*header_id\|>",
            ],
        }

    return None

def normalize_qa_answer(text):
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def clean_qa_candidate(text):
    text = text.strip()
    if not text:
        return None

    text = re.split(r"<\|", text, maxsplit=1)[0].strip()
    text = re.split(r"\n+", text, maxsplit=1)[0].strip()
    text = re.split(
        r"\b(?:answer\s+the\s+following\s+question|question\s*[:：]|"
        r"explanation\s*[:：]|because\s*[:：])",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    text = re.sub(r"\.{2,}\s*(?:more|less)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:read\s+more|more|less)\b", " ", text, flags=re.IGNORECASE)
    text = re.split(r"\s+#", text, maxsplit=1)[0].strip()

    prefix_pattern = re.compile(
        r"^(?:final\s+answer\s*[:：-]\s*|answer\s*[:：-]\s*|"
        r"the\s+answer\s+is\s+|answer\s+is\s+|it\s+is\s+|it's\s+)",
        flags=re.IGNORECASE,
    )
    previous = None
    while previous != text:
        previous = text
        text = prefix_pattern.sub("", text).strip()

    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\"'`*_-.。,:;!?()[]{}")
    return text if text else None


def is_placeholder_qa_candidate(text):
    if not text:
        return True

    lowered = text.lower()
    if "_" in text:
        return True
    if re.fullmatch(r"(?:one|\d+)\s+words?", lowered):
        return True
    if re.fullmatch(r"[\W_]+", text):
        return True
    placeholder_markers = (
        "short phrase",
        "no explanation",
        "following question",
        "fill in the blank",
        "read more",
        "show more",
        "show less",
        "back to the list",
    )
    return any(marker in lowered for marker in placeholder_markers)


def extract_qa_pred(output_text):
    text = output_text.strip()
    if not text:
        return None

    candidates = re.split(r"\banswer\s*[:：]", text, flags=re.IGNORECASE)
    for candidate in candidates:
        candidate = clean_qa_candidate(candidate)
        if not is_placeholder_qa_candidate(candidate):
            return candidate

    fallback = clean_qa_candidate(text)
    if is_placeholder_qa_candidate(fallback):
        return None
    return fallback


def clean_qa_output(output_text):
    pred = extract_qa_pred(output_text)
    if pred is not None:
        return pred

    cleaned = clean_qa_candidate(output_text)
    if is_placeholder_qa_candidate(cleaned):
        return ""
    return cleaned or output_text.strip()


def qa_f1_score(prediction, gold_answer):
    pred_tokens = normalize_qa_answer(prediction).split()
    gold_tokens = normalize_qa_answer(gold_answer).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_exact_match_score(prediction, gold_answer):
    return float(normalize_qa_answer(prediction) == normalize_qa_answer(gold_answer))


def score_qa_prediction(prediction, gold_answers):
    gold_answers = [answer for answer in gold_answers if str(answer).strip()]
    if prediction is None or not gold_answers:
        return 0.0, 0.0

    exact_match = max(qa_exact_match_score(prediction, answer) for answer in gold_answers)
    f1 = max(qa_f1_score(prediction, answer) for answer in gold_answers)
    return exact_match, f1


def build_rouge_scorer():
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def score_rouge_prediction(scorer, prediction, reference):
    prediction = str(prediction).replace("\n", " ")
    reference = str(reference).replace("\n", " ")
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def extract_qa_gold_answers(example, task_name):
    task_name = normalize_task_name(task_name)
    gold_answers = []

    if task_name == "triviaqa":
        answer = example.get("answer", {})
        if isinstance(answer, dict):
            for key in ("value", "aliases", "normalized_value", "normalized_aliases"):
                value = answer.get(key)
                if isinstance(value, list):
                    gold_answers.extend(value)
                elif value:
                    gold_answers.append(value)
        elif isinstance(answer, list):
            gold_answers.extend(answer)
        elif answer:
            gold_answers.append(answer)

    elif task_name == "natural_questions":
        answer = example.get("answer", [])
        if isinstance(answer, list):
            gold_answers.extend(answer)
        elif answer:
            gold_answers.append(answer)

    deduped = []
    seen = set()
    for answer in gold_answers:
        answer = str(answer).strip()
        if answer and answer not in seen:
            deduped.append(answer)
            seen.add(answer)

    return deduped


def clip_input(
    tokenizer,
    prompt,
    task_name,
    device="cuda",
    max_new_tokens=512,
    tree_length=250,
    max_output_length=4096,
    prompt_shots=None,
    model_id="",
):
    task_name = normalize_task_name(task_name)
    prompt_shots = prompt_shots or ""
    end_prompt = ""
    is_instruct = any(kw in model_id.lower() for kw in ["instruct", "chat", "it"])

    if task_name == "cnndm":
        prompt_text = prompt_shots + "Article: " + prompt["article"] + "\nSummary:"
        end_prompt = "\nSummary:"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    elif task_name == "samsum":
        dialogue = prompt["dialogue"].strip()
        if is_instruct and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes conversations.",
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize the following dialogue in one concise paragraph. "
                        "Start directly with the summary. Do not include labels, preambles, "
                        "or another assistant turn.\n\n"
                        f"Dialogue:\n{dialogue}"
                    ),
                },
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
        else:
            prompt_text = (
                prompt_shots
                + "Summarize the following dialogue in one concise paragraph. "
                + "Start directly with the summary.\n\n"
                + f"Dialogue:\n{dialogue}\nSummary:"
            )
            end_prompt = "\nSummary:"
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    elif task_name == "humaneval":
        raw_prompt = prompt["prompt"]
        if is_instruct:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that writes Python code."},
                {
                    "role": "user",
                    "content": (
                        "Please complete the following Python function. "
                        "Provide only the implementation of the function, following the signature "
                        "and docstring provided. Do not repeat the signature or docstring in your response.\n\n"
                        f"```python\n{raw_prompt}\n```"
                    ),
                },
            ]
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = raw_prompt
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
        else:
            inputs = tokenizer(raw_prompt, return_tensors="pt").to(device)

    elif is_longgen_task(task_name):
        base_task = get_longgen_base_task(task_name)
        examples_text = prompt_shots.strip()
        questions_text = format_longgen_questions(base_task, prompt["items"])
        if base_task == "gsm8k":
            task_instruction = (
                "Solve the following grade school math problems. "
                "For each answer block, show concise step-by-step reasoning and end with "
                "'The answer is <number>'."
            )
        elif base_task == "mmlu":
            task_instruction = (
                "Answer the following multiple choice questions. "
                "For each answer block, give concise reasoning and end with "
                "'The answer is <A, B, C, or D>'."
            )
        else:
            raise ValueError(f"Unsupported LongGenBench base task: {base_task}")

        user_content = task_instruction + "\n\n"
        if examples_text:
            user_content += "Examples:\n" + examples_text + "\n\n"
        user_content += "Following Questions:\n" + questions_text

        if is_instruct and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": LONGGEN_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
        else:
            prompt_text = LONGGEN_SYSTEM_PROMPT + "\n\n" + user_content + "\n\n"
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    elif task_name == "gsm8k":
        if is_instruct and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            user_content = (
                prompt_shots
                + "Solve the following grade school math problem. "
                + "Show your reasoning step by step, and finish with a concise final sentence "
                + "in the form: The answer is <number>.\n\n"
                + "Question: "
                + prompt["question"].strip()
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant that solves math word problems."},
                {"role": "user", "content": user_content},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
        else:
            prompt_text = (
                prompt_shots
                + "Question: "
                + prompt["question"].strip()
                + "\nAnswer: Let's think step by step."
            )
            end_prompt = "\nAnswer: Let's think step by step."
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    elif task_name == "mmlu":
        choices = prompt["choices"]
        user_content = (
            prompt_shots
            + "The following is a multiple choice question. "
            + "Choose the best answer and provide a concise explanation. "
            + "Respond in exactly this format:\n"
            + "Answer: <A, B, C, or D>\n"
            + "Explanation: <2-3 concise sentences. Do not repeat the question or choices.>\n\n"
            + f"Question: {prompt['question'].strip()}\n"
            + f"A. {choices[0]}\n"
            + f"B. {choices[1]}\n"
            + f"C. {choices[2]}\n"
            + f"D. {choices[3]}"
        )
        if is_instruct and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": "You answer multiple choice questions with concise reasoning."},
                {"role": "user", "content": user_content},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
        else:
            prompt_text = user_content + "\nAnswer:"
            end_prompt = "\nAnswer:"
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    elif task_name == "mt_bench":
        turns = [turn.strip() for turn in prompt.get("turns", []) if str(turn).strip()]
        current_turn_idx = prompt.get("_current_turn_idx", 0)
        current_turn = turns[current_turn_idx] if current_turn_idx < len(turns) else (turns[0] if turns else "")
        previous_outputs = prompt.get("_previous_outputs", [])

        if is_instruct and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for prev_idx in range(min(current_turn_idx, len(previous_outputs), len(turns))):
                messages.append({"role": "user", "content": turns[prev_idx]})
                messages.append({"role": "assistant", "content": str(previous_outputs[prev_idx]).strip()})
            messages.append({"role": "user", "content": current_turn})
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
        else:
            prompt_text = prompt_shots + "System: You are a helpful assistant.\n"
            for prev_idx in range(min(current_turn_idx, len(previous_outputs), len(turns))):
                prompt_text += (
                    "User: "
                    + turns[prev_idx]
                    + "\nAssistant: "
                    + str(previous_outputs[prev_idx]).strip()
                    + "\n"
                )
            prompt_text += "User: " + current_turn + "\nAssistant:"
            end_prompt = "\nAssistant:"
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    elif is_qa_task(task_name):
        prompt_text = (
            prompt_shots
            + "Answer the following question with only the final short phrase. "
            + "Do not include an explanation, repeated answers, or another question.\n\n"
            + f"Question: {prompt['question'].strip()}\n"
            + "Answer:"
        )
        end_prompt = "\nAnswer:"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    else:
        raise ValueError(f"Unsupported task: {task_name}")

    input_ids = inputs.input_ids
    end_prompt_length = 0
    if end_prompt:
        end_prompt_length = len(tokenizer(end_prompt, return_tensors="pt").input_ids[0])

    if len(input_ids[0]) + max_new_tokens + tree_length >= max_output_length:
        sample_num = len(input_ids[0]) + max_new_tokens + tree_length - max_output_length

        # Important fix:
        # when end_prompt_length == 0, do not use input_ids[0][-0:] because that equals input_ids[0][0:]
        if end_prompt_length > 0:
            head_keep = len(input_ids[0]) - (end_prompt_length + sample_num)
            head_keep = max(head_keep, 0)
            input_ids = torch.cat(
                (input_ids[0][:head_keep], input_ids[0][-end_prompt_length:]),
                dim=0,
            ).unsqueeze(0)
        else:
            keep_len = max_output_length - max_new_tokens - tree_length
            keep_len = max(1, keep_len)
            input_ids = input_ids[:, -keep_len:]

    return input_ids


def load_data(task_name, seed, data_num=10):
    task_name = normalize_task_name(task_name)
    data = []
    prompt_shots = ""

    if task_name == "cnndm":
        n_shot = 1
        data = load_dataset("cnn_dailymail", name="3.0.0", split="test").shuffle(seed=seed).select(range(data_num))
        shots = load_dataset("cnn_dailymail", name="3.0.0", split="train").shuffle(seed=seed).select(range(n_shot))
        prompt_keys = ["article", "highlights"]
        instructions = ["Article: ", "\nSummary: "]
        for i in range(n_shot):
            prompt = (
                instructions[0]
                + shots[i][prompt_keys[0]]
                + instructions[1]
                + shots[i][prompt_keys[1]].replace("\n", "")
                + "\n"
            )
            prompt_shots += prompt

    elif task_name == "samsum":
        dataset = load_dataset("knkarthick/samsum", split="test").shuffle(seed=seed)
        if data_num is not None and data_num < len(dataset):
            dataset = dataset.select(range(data_num))
        data = dataset

    elif task_name == "humaneval":
        original_data = read_problems()
        for i, task_id in enumerate(original_data):
            if i >= data_num:
                break
            data.append(original_data[task_id])

    elif task_name == "long_gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        if data_num is not None and data_num < len(dataset):
            dataset = dataset.select(range(data_num))
        examples = [dict(dataset[i]) for i in range(len(dataset))]
        num_shots = get_longgen_env_int(
            task_name,
            "NUM_SHOTS",
            LONGGEN_DEFAULT_NUM_SHOTS[task_name],
        )
        if num_shots > 0:
            shot_dataset = load_dataset("openai/gsm8k", "main", split="train").select(range(num_shots))
            prompt_shots = format_longgen_examples(
                "gsm8k",
                [dict(shot_dataset[i]) for i in range(len(shot_dataset))],
            )
        data = make_longgen_batches(task_name, examples)

    elif task_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        if data_num is not None and data_num < len(dataset):
            dataset = dataset.select(range(data_num))
        data = dataset

    elif task_name == "long_mmlu":
        dataset = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed)
        if data_num is not None and data_num < len(dataset):
            dataset = dataset.select(range(data_num))
        examples = [dict(dataset[i]) for i in range(len(dataset))]
        num_shots = get_longgen_env_int(
            task_name,
            "NUM_SHOTS",
            LONGGEN_DEFAULT_NUM_SHOTS[task_name],
        )
        if num_shots > 0:
            try:
                shot_dataset = load_dataset("cais/mmlu", "all", split="dev").shuffle(seed=seed)
            except ValueError:
                shot_dataset = load_dataset("cais/mmlu", "all", split="validation").shuffle(seed=seed)
            shot_dataset = shot_dataset.select(range(min(num_shots, len(shot_dataset))))
            prompt_shots = format_longgen_examples(
                "mmlu",
                [dict(shot_dataset[i]) for i in range(len(shot_dataset))],
            )
        data = make_longgen_batches(task_name, examples)

    elif task_name == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed)
        if data_num is not None and data_num < len(dataset):
            dataset = dataset.select(range(data_num))
        data = dataset

    elif task_name == "triviaqa":
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").shuffle(seed=seed)
        if data_num is not None and data_num < len(dataset):
            dataset = dataset.select(range(data_num))
        data = dataset

    elif task_name == "natural_questions":
        dataset = load_dataset("google-research-datasets/nq_open", "nq_open", split="validation").shuffle(seed=seed)
        if data_num is not None and data_num < len(dataset):
            dataset = dataset.select(range(data_num))
        data = dataset

    elif task_name == "mt_bench":
        # Load MT-Bench from local FastChat data
        import os
        fastchat_mt_bench_path = None
        
        # Try to find FastChat MT-Bench data in common locations
        possible_paths = [
            "./FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl",
            "../FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl",
            "../../FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl",
            "/home/tjlin/KV_SSD/SWIFT/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                fastchat_mt_bench_path = path
                break
        
        if fastchat_mt_bench_path is None:
            raise ValueError(
                "MT-Bench dataset not found. Please ensure FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl exists."
            )
        
        # Load and parse JSONL file
        data = []
        with open(fastchat_mt_bench_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    data.append(sample)
        
        if data_num is not None and data_num < len(data):
            data = data[:data_num]

    else:
        raise ValueError(f"Unsupported task: {task_name}")

    return data, prompt_shots


def run_eval(
    model,
    tokenizer,
    forward_func,
    model_id,
    answer_file,
    max_new_tokens,
    num_gpus_per_model,
    num_gpus_total,
    task_name,
    data_num,
    seed,
    **kwargs,
):
    assert num_gpus_total % num_gpus_per_model == 0

    seed_everything(seed)

    data, prompt_shots = load_data(task_name, seed, data_num=data_num)
    get_answers_func = get_model_answers

    get_answers_func(
        model,
        tokenizer,
        forward_func,
        model_id,
        data,
        prompt_shots,
        answer_file,
        max_new_tokens,
        task_name,
        **kwargs,
    )


@torch.inference_mode()
def get_model_answers(
    model,
    tokenizer,
    forward_func,
    model_id,
    data,
    prompt_shots,
    answer_file,
    max_new_tokens,
    task_name,
    **kwargs,
):
    task_name = normalize_task_name(task_name)
    model.eval()
    print("Check model training state:", model.training)

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print("CUDA VISIBLE DEVICES:", cuda_visible_devices)

    generation_stop_config = get_generation_stop_config(task_name)
    forward_kwargs = dict(kwargs)
    if generation_stop_config is not None:
        forward_kwargs["stop_config"] = generation_stop_config
    rouge_metric_scorer = build_rouge_scorer() if task_name == "samsum" else None

    accept_lengths_tree = []
    total_draft_num = 0
    total_correct = 0
    total_scored = 0
    total_qa_exact_match = 0.0
    total_qa_f1 = 0.0
    total_qa_scored = 0
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    total_rougeL = 0.0
    total_rouge_scored = 0

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(os.path.expanduser(answer_file), "w", encoding="utf-8"):
        pass

    for question_idx, question in enumerate(tqdm(data)):
        choices = []

        # MT-Bench needs one FastChat-compatible answer row per question.
        # There is no label-based accuracy here; quality is judged later with
        # the FastChat MT-Bench judge pipeline using the generated turns.
        if task_name == "mt_bench":
            turns_outputs = []
            turns = [turn.strip() for turn in question.get("turns", []) if str(turn).strip()]
            question_for_inference = dict(question)
            cur_accept_lengths_tree = []
            cur_draft_num = 0
            steps = []
            new_tokens = []
            wall_time = []
            adaptive_samples = []

            for turn_idx in range(len(turns)):
                question_for_inference["_current_turn_idx"] = turn_idx
                question_for_inference["_previous_outputs"] = turns_outputs

                input_ids = clip_input(
                    tokenizer,
                    question_for_inference,
                    task_name,
                    device=model.model.embed_tokens.weight.device,
                    max_new_tokens=max_new_tokens,
                    prompt_shots=prompt_shots,
                    max_output_length=model.config.max_position_embeddings,
                    model_id=model_id,
                )

                set_flops_trace_context(
                    forward_kwargs.get("statistics"),
                    question_idx,
                    question_id=question.get("question_id", question_idx),
                    turn_idx=turn_idx,
                )
                safe_cuda_synchronize()
                start_time = time.time()
                output_ids, new_token_num, step, accept_length_tree, draft_token_num = forward_func(
                    input_ids,
                    model,
                    tokenizer,
                    max_new_tokens,
                    **forward_kwargs,
                )
                safe_cuda_synchronize()
                total_time = time.time() - start_time

                cur_accept_lengths_tree.extend(accept_length_tree)
                cur_draft_num += int(draft_token_num)

                if output_ids.dim() == 2:
                    generated_ids = output_ids[0][len(input_ids[0]):]
                elif output_ids.dim() == 1:
                    generated_ids = output_ids[len(input_ids[0]):]
                else:
                    raise ValueError(f"Unexpected output_ids shape: {output_ids.shape}")

                output = tokenizer.decode(
                    generated_ids,
                    spaces_between_special_tokens=False,
                )
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = clean_mt_bench_output(output)

                turns_outputs.append(output)
                steps.append(int(step))
                new_tokens.append(int(new_token_num))
                wall_time.append(total_time)

                runtime_statistics = forward_kwargs.get("statistics")
                if runtime_statistics and runtime_statistics.get("local_adaptive_controller"):
                    adaptive_sample = runtime_statistics.get("adaptive_last_sample")
                    if adaptive_sample is not None:
                        adaptive_samples.append(adaptive_sample)

            accept_lengths_tree.extend(cur_accept_lengths_tree)
            total_draft_num += cur_draft_num

            if cur_draft_num > 0:
                sample_acceptance_rate = (sum(cur_accept_lengths_tree) - len(cur_accept_lengths_tree)) / cur_draft_num
            else:
                sample_acceptance_rate = 0.0

            metrics = {
                "question_index": question_idx,
                "task_name": task_name,
                "prompt_turns": question.get("turns", []),
                "decoding_steps": steps,
                "new_tokens": new_tokens,
                "wall_time": wall_time,
                "accept_lengths": cur_accept_lengths_tree,
                "acceptance_rate": sample_acceptance_rate,
            }
            if adaptive_samples:
                metrics["adaptive"] = adaptive_samples

            mt_question_id = question.get("question_id", question_idx)
            ans_json = {
                "question_index": question_idx,
                "question_id": mt_question_id,
                "answer_id": f"{model_id}-{mt_question_id}",
                "model_id": model_id,
                "choices": [{"index": 0, "turns": turns_outputs}],
                "tstamp": time.time(),
                "metrics": metrics,
            }
            if "category" in question:
                ans_json["category"] = question["category"]

            with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
                fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")
            continue

        else:
            input_ids = clip_input(
                tokenizer,
                question,
                task_name,
                device=model.model.embed_tokens.weight.device,
                max_new_tokens=max_new_tokens,
                prompt_shots=prompt_shots,
                max_output_length=model.config.max_position_embeddings,
                model_id=model_id,
            )

            set_flops_trace_context(
                forward_kwargs.get("statistics"),
                question_idx,
                question_id=question.get("question_id", question_idx) if isinstance(question, dict) else question_idx,
                turn_idx=None,
            )
            cur_accept_lengths_tree = []
            cur_draft_num = 0
            steps = []
            new_tokens = []
            wall_time = []

            safe_cuda_synchronize()
            start_time = time.time()
            output_ids, new_token_num, step, accept_length_tree, draft_token_num = forward_func(
                input_ids,
                model,
                tokenizer,
                max_new_tokens,
                **forward_kwargs,
            )
            safe_cuda_synchronize()
            total_time = time.time() - start_time

            cur_accept_lengths_tree.extend(accept_length_tree)
            cur_draft_num += draft_token_num
            
            # Handle both 1D and 2D output_ids tensors
            if output_ids.dim() == 2:
                output_ids = output_ids[0][len(input_ids[0]):]
            elif output_ids.dim() == 1:
                output_ids = output_ids[len(input_ids[0]):]
            else:
                raise ValueError(f"Unexpected output_ids shape: {output_ids.shape}")

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()
            raw_output = output
            if task_name == "mmlu":
                output = clean_mmlu_output(output)
            elif task_name == "gsm8k":
                output = clean_gsm8k_output(output)
            elif task_name == "samsum":
                output = clean_samsum_output(output)
            elif is_qa_task(task_name):
                output = clean_qa_output(output)

            steps.append(int(step))
            new_tokens.append(int(new_token_num))
            wall_time.append(total_time)

        accept_lengths_tree.extend(cur_accept_lengths_tree)
        total_draft_num += cur_draft_num

        if cur_draft_num > 0:
            sample_acceptance_rate = (sum(cur_accept_lengths_tree) - len(cur_accept_lengths_tree)) / cur_draft_num
        else:
            sample_acceptance_rate = 0.0

        sample_record = {
            "turns": output,
            "decoding_steps": steps,
            "new_tokens": new_tokens,
            "wall_time": wall_time,
            "accept_lengths": cur_accept_lengths_tree,
            "acceptance_rate": sample_acceptance_rate,
        }
        if task_name in {"mmlu", "gsm8k", "samsum"} and raw_output != output:
            sample_record["raw_turns"] = raw_output
        runtime_statistics = forward_kwargs.get("statistics")
        if runtime_statistics and runtime_statistics.get("local_adaptive_controller"):
            adaptive_sample = runtime_statistics.get("adaptive_last_sample")
            if adaptive_sample is not None:
                sample_record["adaptive"] = adaptive_sample

        sample_correct = None

        if task_name == "humaneval":
            if isinstance(question, dict) and "task_id" in question:
                sample_record["task_id"] = question["task_id"]

        elif is_longgen_task(task_name):
            longgen_results, correct_count = score_longgen_batch(task_name, question, output)
            question_count = len(longgen_results)
            total_correct += correct_count
            total_scored += question_count

            sample_record["longgen_task"] = question.get("base_task", get_longgen_base_task(task_name))
            sample_record["batch_index"] = question.get("batch_index", question_idx)
            sample_record["batch_start"] = question.get("batch_start", None)
            sample_record["question_count"] = question_count
            sample_record["correct_count"] = correct_count
            sample_record["accuracy"] = correct_count / question_count if question_count > 0 else 0.0
            sample_record["longgen_results"] = longgen_results

        elif task_name == "gsm8k":
            gold = extract_gsm8k_gold(question["answer"])
            pred = extract_gsm8k_pred(output)
            normalized_gold = normalize_gsm8k_answer_number(gold)
            normalized_pred = normalize_gsm8k_answer_number(pred)
            sample_correct = (
                normalized_pred is not None
                and normalized_gold is not None
                and normalized_pred == normalized_gold
            )

            sample_record["question"] = question["question"]
            sample_record["gold_answer"] = gold
            sample_record["pred_answer"] = pred
            sample_record["correct"] = sample_correct

        elif task_name == "mmlu":
            gold = extract_mmlu_gold(question)
            pred = extract_mmlu_pred(output)
            sample_correct = (pred is not None and gold is not None and pred == gold)

            sample_record["question"] = question["question"]
            sample_record["choices_text"] = question["choices"]
            sample_record["subject"] = question.get("subject", None)
            sample_record["gold_answer"] = gold
            sample_record["pred_answer"] = pred
            sample_record["correct"] = sample_correct

        elif is_qa_task(task_name):
            gold_answers = extract_qa_gold_answers(question, task_name)
            pred = extract_qa_pred(output)
            exact_match, f1 = score_qa_prediction(pred, gold_answers)
            sample_correct = exact_match == 1.0

            sample_record["question"] = question["question"]
            if "question_id" in question:
                sample_record["question_id"] = question["question_id"]
            sample_record["gold_answers"] = gold_answers
            sample_record["pred_answer"] = pred
            sample_record["exact_match"] = exact_match
            sample_record["f1"] = f1
            sample_record["correct"] = sample_correct

            total_qa_exact_match += exact_match
            total_qa_f1 += f1
            total_qa_scored += 1

        elif task_name == "samsum":
            rouge_scores = score_rouge_prediction(
                rouge_metric_scorer,
                output,
                question["summary"],
            )
            sample_record["dialogue"] = question["dialogue"]
            sample_record["gold_summary"] = question["summary"]
            sample_record["rouge1"] = rouge_scores["rouge1"]
            sample_record["rouge2"] = rouge_scores["rouge2"]
            sample_record["rougeL"] = rouge_scores["rougeL"]

            total_rouge1 += rouge_scores["rouge1"]
            total_rouge2 += rouge_scores["rouge2"]
            total_rougeL += rouge_scores["rougeL"]
            total_rouge_scored += 1

        if sample_correct is not None:
            total_scored += 1
            total_correct += int(sample_correct)

        choices.append(sample_record)

        with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
            ans_json = {
                "question_index": question_idx,
                "model_id": model_id,
                "task_name": task_name,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")

    summary = {}

    if len(accept_lengths_tree) > 0:
        summary["Mean accepted tokens"] = float(np.mean(accept_lengths_tree))

    if total_draft_num > 0:
        summary["Token acceptance rate"] = float(
            (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num
        )
    else:
        summary["Token acceptance rate"] = 0.0

    # Keep skip-layer summary only when it is meaningful.
    if (
        hasattr(model, "get_skip_layers")
        and len(accept_lengths_tree) > 0
        and np.mean(accept_lengths_tree) > 1
    ):
        best_attn_skip_layer_id_set, best_mlp_skip_layer_id_set = model.get_skip_layers()
        best_skip_ratio = (
            len(best_mlp_skip_layer_id_set) + len(best_attn_skip_layer_id_set)
        ) / ((model.config.num_hidden_layers - 2) * 2)

        summary["Best Skip Ratio"] = best_skip_ratio
        summary["Best Attn Layer Set"] = [int(x) for x in list(best_attn_skip_layer_id_set)]
        summary["Best MLP Layer Set"] = [int(x) for x in list(best_mlp_skip_layer_id_set)]

    runtime_statistics = forward_kwargs.get("statistics")
    add_flops_trace_summary(summary, runtime_statistics)
    if runtime_statistics and runtime_statistics.get("draft_kv_score_source"):
        summary["Draft KV Cache Mode"] = runtime_statistics.get("draft_kv_cache_mode", "copy")
        summary["Draft KV Score Source"] = runtime_statistics.get("draft_kv_score_source")
        summary["Draft KV Copy Cache Rebuilds"] = int(runtime_statistics.get("draft_kv_copy_cache_rebuilds", 0))
        summary["Draft KV Mask Cache Rebuilds"] = int(runtime_statistics.get("draft_kv_mask_cache_rebuilds", 0))
        summary["Draft KV Reuse EMA"] = float(runtime_statistics.get("draft_kv_reuse_ema", 0.7))
        summary["Draft KV Reuse Score Hits"] = int(runtime_statistics.get("draft_kv_reuse_score_hits", 0))
        summary["Draft KV Reuse Score Misses"] = int(runtime_statistics.get("draft_kv_reuse_score_misses", 0))
        summary["Draft KV Reuse Score Updates"] = int(runtime_statistics.get("draft_kv_reuse_score_updates", 0))
        summary["Draft KV Reuse Empty Updates"] = int(runtime_statistics.get("draft_kv_reuse_score_empty_updates", 0))
        summary["Verify KV Compress"] = bool(runtime_statistics.get("verify_kv_compress", False))
        if runtime_statistics.get("verify_kv_compress", False):
            summary["Verify KV Cache Mode"] = runtime_statistics.get("verify_kv_cache_mode", "copy")
            summary["Verify KV Score Source"] = runtime_statistics.get("verify_kv_score_source", "reuse")
            summary["Verify KV Retain Ratio"] = runtime_statistics.get("verify_kv_retain_ratio")
            summary["Verify KV Bootstrap Full Steps"] = int(runtime_statistics.get("verify_kv_bootstrap_full_steps", 0))
            summary["Verify KV Bootstrap Full Uses"] = int(runtime_statistics.get("verify_kv_bootstrap_full_uses", 0))
            summary["Verify KV Copy Cache Rebuilds"] = int(runtime_statistics.get("verify_kv_copy_cache_rebuilds", 0))
            summary["Verify KV Mask Cache Rebuilds"] = int(runtime_statistics.get("verify_kv_mask_cache_rebuilds", 0))
            summary["Verify KV Reuse Score Hits"] = int(runtime_statistics.get("verify_kv_reuse_score_hits", 0))
            summary["Verify KV Reuse Score Misses"] = int(runtime_statistics.get("verify_kv_reuse_score_misses", 0))
            summary["Verify KV Semantic Score Hits"] = int(runtime_statistics.get("verify_kv_semantic_score_hits", 0))
            summary["Verify KV Semantic Score Misses"] = int(runtime_statistics.get("verify_kv_semantic_score_misses", 0))
            summary["Verify KV Semantic Score Updates"] = int(runtime_statistics.get("verify_kv_semantic_score_updates", 0))
            summary["Verify KV Scope Recent Size"] = int(runtime_statistics.get("verify_kv_scope_recent_size", 0))
            summary["Verify KV Scope Beta1"] = int(runtime_statistics.get("verify_kv_scope_beta1", 0))
            summary["Verify KV Scope Beta2"] = int(runtime_statistics.get("verify_kv_scope_beta2", 0))
            summary["Verify KV Scope Score Full Only"] = bool(runtime_statistics.get("verify_kv_scope_score_full_only", True))
            summary["Verify KV Scope Full Decode Uses"] = int(runtime_statistics.get("verify_kv_scope_full_decode_uses", 0))
            summary["Verify KV Safe Commit"] = bool(runtime_statistics.get("verify_kv_safe_commit", False))
            summary["Verify KV Safe Commit Uses"] = int(runtime_statistics.get("verify_kv_safe_commit_uses", 0))
            summary["Verify KV Dynamic"] = bool(runtime_statistics.get("verify_kv_dynamic", False))
            if runtime_statistics.get("verify_kv_dynamic", False):
                summary["Verify KV Dynamic Initial Beta1"] = int(runtime_statistics.get("verify_kv_dynamic_initial_beta1", 0))
                summary["Verify KV Dynamic Current Beta1"] = int(runtime_statistics.get("verify_kv_dynamic_current_beta1", runtime_statistics.get("verify_kv_scope_beta1", 0)))
                summary["Verify KV Dynamic Min Beta1"] = int(runtime_statistics.get("verify_kv_dynamic_min_beta1", 0))
                summary["Verify KV Dynamic Max Beta1"] = int(runtime_statistics.get("verify_kv_dynamic_max_beta1", 0))
                summary["Verify KV Dynamic Step"] = int(runtime_statistics.get("verify_kv_dynamic_step", 0))
                summary["Verify KV Dynamic Window"] = int(runtime_statistics.get("verify_kv_dynamic_window", 0))
                summary["Verify KV Dynamic Acceptance Floor"] = float(runtime_statistics.get("verify_kv_dynamic_acceptance_floor", 0.88))
                summary["Verify KV Dynamic Mean Floor"] = float(runtime_statistics.get("verify_kv_dynamic_mean_floor", 3.0))
                summary["Verify KV Dynamic Confidence Floor"] = float(runtime_statistics.get("verify_kv_dynamic_confidence_floor", 0.5))
                summary["Verify KV Dynamic Confidence Low"] = float(runtime_statistics.get("verify_kv_dynamic_confidence_low", 0.25))
                summary["Verify KV Dynamic Decisions"] = int(runtime_statistics.get("verify_kv_dynamic_decision_count", 0))
                summary["Verify KV Dynamic Switches"] = int(runtime_statistics.get("verify_kv_dynamic_switches", 0))
                summary["Verify KV Dynamic More-Compress Switches"] = int(runtime_statistics.get("verify_kv_dynamic_more_compress_switches", 0))
                summary["Verify KV Dynamic Less-Compress Switches"] = int(runtime_statistics.get("verify_kv_dynamic_less_compress_switches", 0))
                summary["Verify KV Dynamic Action Counts"] = runtime_statistics.get("verify_kv_dynamic_action_counts", {})
                dyn_obs = int(runtime_statistics.get("verify_kv_dynamic_observations", 0))
                summary["Verify KV Dynamic Observations"] = dyn_obs
                if dyn_obs > 0:
                    summary["Verify KV Dynamic Avg Acceptance"] = (
                        float(runtime_statistics.get("verify_kv_dynamic_acceptance_sum", 0.0)) / dyn_obs
                    )
                    summary["Verify KV Dynamic Avg Accepted Tokens"] = (
                        float(runtime_statistics.get("verify_kv_dynamic_mean_sum", 0.0)) / dyn_obs
                    )
                    summary["Verify KV Dynamic Avg Confidence Margin"] = (
                        float(runtime_statistics.get("verify_kv_dynamic_confidence_margin_sum", 0.0)) / dyn_obs
                    )
                    summary["Verify KV Dynamic Avg Min Confidence Margin"] = (
                        float(runtime_statistics.get("verify_kv_dynamic_confidence_min_sum", 0.0)) / dyn_obs
                    )
                summary["Verify KV Dynamic Last Window Acceptance"] = runtime_statistics.get("verify_kv_dynamic_last_window_acceptance")
                summary["Verify KV Dynamic Last Window Mean"] = runtime_statistics.get("verify_kv_dynamic_last_window_mean")
                summary["Verify KV Dynamic Last Window Margin"] = runtime_statistics.get("verify_kv_dynamic_last_window_margin")
            scope_count = int(runtime_statistics.get("verify_kv_scope_selection_count", 0))
            summary["Verify KV Scope Selection Count"] = scope_count
            if scope_count > 0:
                summary["Verify KV Scope Avg Prefill Kept"] = (
                    float(runtime_statistics.get("verify_kv_scope_prefill_kept_sum", 0)) / scope_count
                )
                summary["Verify KV Scope Avg Decode Len"] = (
                    float(runtime_statistics.get("verify_kv_scope_decode_len_sum", 0)) / scope_count
                )
                summary["Verify KV Scope Avg Decode Kept"] = (
                    float(runtime_statistics.get("verify_kv_scope_decode_kept_sum", 0)) / scope_count
                )
    if runtime_statistics and runtime_statistics.get("cosine_prefill_skip_layers"):
        summary["Cosine Prefill Skip Layers"] = True
        summary["Cosine Skip Mode"] = runtime_statistics.get("cosine_skip_mode", "topk")
        summary["Cosine Attn Alpha"] = float(runtime_statistics.get("cosine_attn_alpha", 0.985))
        summary["Cosine Max Skip Layers"] = runtime_statistics.get("cosine_max_skip_layers")
        summary["Cosine Keep First Layers"] = int(runtime_statistics.get("cosine_keep_first_layers", 1))
        summary["Cosine Keep Last Layers"] = int(runtime_statistics.get("cosine_keep_last_layers", 2))
        summary["Cosine MLP Interval"] = int(runtime_statistics.get("cosine_mlp_interval", 0))
        summary["Cosine Prefill Count"] = int(runtime_statistics.get("cosine_prefill_count", 0))
        prefill_count = int(runtime_statistics.get("cosine_prefill_count", 0))
        summary["Cosine Avg Attn Skip Count"] = (
            float(runtime_statistics.get("cosine_attn_skip_count_sum", 0)) / prefill_count
            if prefill_count > 0
            else None
        )
        summary["Cosine Avg MLP Skip Count"] = (
            float(runtime_statistics.get("cosine_mlp_skip_count_sum", 0)) / prefill_count
            if prefill_count > 0
            else None
        )
        summary["Last Cosine Attn Layer Set"] = runtime_statistics.get("cosine_attn_skip_layers", [])
        summary["Last Cosine MLP Layer Set"] = runtime_statistics.get("cosine_mlp_skip_layers", [])
    if runtime_statistics and runtime_statistics.get("dynamic_retain_ratio"):
        summary["Dynamic Retain Ratio"] = True
        summary["Best Draft KV Retain Ratio"] = float(
            runtime_statistics.get(
                "best_retain_ratio",
                runtime_statistics.get("draft_kv_retain_ratio", getattr(model, "draft_kv_retain_ratio", 1.0)),
            )
        )
        if "best_retain_score" in runtime_statistics:
            summary["Best Retain Matchness"] = float(runtime_statistics["best_retain_score"])
        if "best_retain_utility" in runtime_statistics:
            summary["Best Retain Utility"] = float(runtime_statistics["best_retain_utility"])
        summary["Retain Utility Mode"] = runtime_statistics.get("retain_utility_mode", "relative")
        summary["Retain Compression Weight"] = float(runtime_statistics.get("retain_compression_weight", 0.5))
        summary["Retain Score Tolerance"] = float(runtime_statistics.get("retain_score_tolerance", 0.05))
        summary["Retain Utility Lambda"] = float(runtime_statistics.get("retain_utility_lambda", 1.0))
        summary["Retain UCB C"] = float(runtime_statistics.get("retain_ucb_c", 0.3))
        summary["Retain Warmup Rounds"] = int(runtime_statistics.get("retain_warmup_rounds", 50))
        summary["Retain Filter Top K"] = int(runtime_statistics.get("retain_filter_top_k", 3))
        summary["Retain Refine Rounds"] = int(runtime_statistics.get("retain_refine_rounds", 100))
        summary["Retain Final Tolerance"] = float(runtime_statistics.get("retain_final_tolerance", 0.05))
        summary["Final Layer Refine Rounds"] = int(runtime_statistics.get("final_layer_refine_rounds", 100))
        summary["Retain Stage"] = runtime_statistics.get("retain_stage")
        summary["Retain Candidate Ratios"] = runtime_statistics.get("retain_candidate_ratios")
        summary["Retain Final Ratio"] = runtime_statistics.get("retain_final_ratio")
        summary["Retain Ratio Search State"] = runtime_statistics.get("retain_ratio_state", {})
    if runtime_statistics and runtime_statistics.get("local_adaptive_controller"):
        summary["Local Adaptive Controller"] = True
        summary["Draft KV Retain Ratio"] = float(runtime_statistics.get("retain_final_ratio", runtime_statistics.get("adaptive_initial_retain_ratio", 1.0)))
        summary["Adaptive Starts After Dynamic"] = bool(runtime_statistics.get("dynamic_retain_ratio", False))
        summary["Adaptive Ratio Ladder"] = runtime_statistics.get("adaptive_ratio_ladder", [])
        summary["Adaptive Window"] = int(runtime_statistics.get("adaptive_window", 16))
        summary["Adaptive Min Observations"] = int(runtime_statistics.get("adaptive_min_observations", 24))
        summary["Adaptive Reference Mode"] = runtime_statistics.get("adaptive_reference_mode", "global")
        summary["Adaptive Std K"] = float(runtime_statistics.get("adaptive_std_k", 0.5))
        summary["Adaptive Up Std K"] = float(runtime_statistics.get("adaptive_up_std_k", runtime_statistics.get("adaptive_std_k", 0.5)))
        summary["Adaptive Down Std K"] = float(runtime_statistics.get("adaptive_down_std_k", runtime_statistics.get("adaptive_std_k", 0.5)))
        summary["Adaptive Std Floor"] = float(runtime_statistics.get("adaptive_std_floor", 0.05))
        summary["Adaptive Patience"] = int(runtime_statistics.get("adaptive_patience", 1))
        summary["Adaptive Cooldown"] = int(runtime_statistics.get("adaptive_cooldown", 8))
        adaptive_global_draft = int(runtime_statistics.get("adaptive_global_draft_tokens", 0))
        summary["Adaptive Global Observations"] = len(runtime_statistics.get("adaptive_global_step_acceptance_history", []))
        summary["Adaptive Global Acceptance Rate"] = (
            float(runtime_statistics.get("adaptive_global_accepted_tokens", 0)) / adaptive_global_draft
            if adaptive_global_draft > 0
            else None
        )
        summary["Adaptive Total Switches"] = int(runtime_statistics.get("adaptive_total_switches", 0))
        summary["Adaptive Layer Controller"] = bool(runtime_statistics.get("adaptive_layer_controller", False))
        summary["Adaptive Layer Fallback Window"] = int(runtime_statistics.get("adaptive_layer_fallback_window", 16))
        summary["Adaptive Layer Improvement Delta"] = float(runtime_statistics.get("adaptive_layer_improvement_delta", 0.0))
        summary["Adaptive Layer Total Switches"] = int(runtime_statistics.get("adaptive_layer_total_switches", 0))
        summary["Adaptive Less-Skip Switches"] = int(runtime_statistics.get("adaptive_less_skip_total_switches", 0))
        summary["Adaptive More-Skip Switches"] = int(runtime_statistics.get("adaptive_more_skip_total_switches", 0))
        summary["Adaptive Layer Questions With Switch"] = int(runtime_statistics.get("adaptive_layer_questions_with_switch", 0))
        summary["Adaptive Aggressive Controller"] = bool(runtime_statistics.get("adaptive_aggressive_controller", False))
        summary["Adaptive Min Retain Ratio"] = float(runtime_statistics.get("adaptive_min_retain_ratio", 0.1))
        summary["Adaptive Ratio Step"] = float(runtime_statistics.get("adaptive_ratio_step", 0.1))
        summary["Adaptive Aggressive Tolerance"] = float(runtime_statistics.get("adaptive_aggressive_tolerance", 0.02))
        summary["Adaptive Aggressive Std K"] = float(runtime_statistics.get("adaptive_aggressive_std_k", 0.5))
        summary["Adaptive Aggressive Patience"] = int(runtime_statistics.get("adaptive_aggressive_patience", 1))
        summary["Adaptive Max Extra Skip Layers"] = runtime_statistics.get("adaptive_max_extra_skip_layers")
        summary["Adaptive Max Skip Layers"] = runtime_statistics.get("adaptive_max_skip_layers")
        summary["Adaptive Aggressive Ratio Down Switches"] = int(runtime_statistics.get("adaptive_aggressive_ratio_down_switches", 0))
        summary["Adaptive Final Controller"] = bool(runtime_statistics.get("adaptive_final_controller", False))
        summary["Final Target Mean Accepted"] = float(runtime_statistics.get("final_target_mean_accepted", 3.0))
        summary["Final Bad Mean Accepted"] = float(runtime_statistics.get("final_bad_mean_accepted", 2.5))
        summary["Final Severe Mean Accepted"] = float(runtime_statistics.get("final_severe_mean_accepted", 2.1))
        summary["Final Token Acceptance Floor"] = float(runtime_statistics.get("final_token_acceptance_floor", 0.85))
        summary["Final More-Skip Token Acceptance Floor"] = float(runtime_statistics.get("final_more_skip_token_acceptance_floor", 0.90))
        summary["Final Draft Len Floor"] = float(runtime_statistics.get("final_draft_len_floor", 2.0))
        summary["Final More-Skip Draft Len Floor"] = float(runtime_statistics.get("final_more_skip_draft_len_floor", 2.2))
        summary["Final Stable Mean Margin"] = float(runtime_statistics.get("final_stable_mean_margin", 0.1))
        summary["Final Soft Max Skip Layers"] = int(runtime_statistics.get("final_soft_max_skip_layers", 17))
        summary["Final Hard Max Skip Layers"] = int(runtime_statistics.get("final_hard_max_skip_layers", 18))
        summary["Final Min Ratio For More-Skip"] = float(runtime_statistics.get("final_min_ratio_for_more_skip", 0.4))
        summary["Final Low Ratio Guard"] = float(runtime_statistics.get("final_low_ratio_guard", 0.2))
        summary["Final Low Ratio Guard Skip Layers"] = int(runtime_statistics.get("final_low_ratio_guard_skip_layers", 17))
        summary["Final Hard Layer Mean Floor"] = float(runtime_statistics.get("final_hard_layer_mean_floor", 2.8))
        summary["Final Hard Probe Mean Margin"] = float(runtime_statistics.get("final_hard_probe_mean_margin", 0.5))
        summary["Final Hard Probe Token Acceptance Floor"] = float(runtime_statistics.get("final_hard_probe_token_acceptance_floor", 0.92))
        summary["Final Hard Probe Draft Len Margin"] = float(runtime_statistics.get("final_hard_probe_draft_len_margin", 0.3))
        summary["Final Ratio Down Gain Weight"] = float(runtime_statistics.get("final_ratio_down_gain_weight", 1.0))
        summary["Final Layer Skip Gain Weight"] = float(runtime_statistics.get("final_layer_skip_gain_weight", 3.0))
        summary["Final Controller Action Counts"] = runtime_statistics.get("adaptive_final_controller_action_counts", {})
        summary["Adaptive Final2 Controller"] = bool(runtime_statistics.get("adaptive_final2_controller", False))
        summary["Final2 Low Std K"] = float(runtime_statistics.get("final2_low_std_k", 0.5))
        summary["Final2 High Std K"] = float(runtime_statistics.get("final2_high_std_k", 0.5))
        summary["Final2 Mean Std Floor"] = float(runtime_statistics.get("final2_mean_std_floor", 0.10))
        summary["Final2 Min Config Observations"] = int(runtime_statistics.get("final2_min_config_observations", 16))
        summary["Final2 Prediction Beta"] = float(runtime_statistics.get("final2_prediction_beta", 0.5))
        summary["Final2 Ratio Mean Slope"] = float(runtime_statistics.get("final2_ratio_mean_slope", 1.0))
        summary["Final2 Layer Mean Slope"] = float(runtime_statistics.get("final2_layer_mean_slope", 0.45))
        summary["Final2 Cold Start Penalty"] = float(runtime_statistics.get("final2_cold_start_penalty", 0.15))
        summary["Final2 Token Acceptance Floor"] = float(runtime_statistics.get("final2_token_acceptance_floor", 0.85))
        summary["Final2 More Aggressive Token Acceptance Floor"] = float(runtime_statistics.get("final2_more_aggressive_token_acceptance_floor", 0.90))
        summary["Final2 Draft Len Floor"] = float(runtime_statistics.get("final2_draft_len_floor", 2.0))
        summary["Final2 More Aggressive Draft Len Floor"] = float(runtime_statistics.get("final2_more_aggressive_draft_len_floor", 2.2))
        summary["Final2 Switch Cost"] = float(runtime_statistics.get("final2_switch_cost", 0.02))
        summary["Final2 Layer Switch Cost"] = float(runtime_statistics.get("final2_layer_switch_cost", runtime_statistics.get("final2_switch_cost", 0.02)))
        summary["Final2 Ratio Down Gain Weight"] = float(runtime_statistics.get("final2_ratio_down_gain_weight", 1.0))
        summary["Final2 Layer Skip Gain Weight"] = float(runtime_statistics.get("final2_layer_skip_gain_weight", 2.0))
        summary["Final2 Decision Count"] = int(runtime_statistics.get("final2_decision_count", 0))
        summary["Final2 Keep Decisions"] = int(runtime_statistics.get("final2_keep_decisions", 0))
        summary["Final2 Controller Action Counts"] = runtime_statistics.get("adaptive_final2_controller_action_counts", {})
        summary["Final2 Prediction Source Counts"] = runtime_statistics.get("final2_prediction_source_counts", {})
        if runtime_statistics.get("adaptive_cold_start", False):
            summary["Adaptive Cold Start"] = True
            summary["Adaptive Cold Start Version"] = runtime_statistics.get("adaptive_cold_start_version", "dynamic-6")
            summary["Adaptive Cold Start Mode"] = runtime_statistics.get("adaptive_cold_start_mode", "dynamic")
            summary["Adaptive Cold Start Time Sec"] = float(runtime_statistics.get("adaptive_cold_start_time_sec", 0.0))
            summary["Adaptive Cold Start Requested Configs"] = int(runtime_statistics.get("adaptive_cold_start_requested_config_count", 0))
            summary["Adaptive Cold Start Merged Configs"] = int(runtime_statistics.get("adaptive_cold_start_merged_config_count", 0))
            summary["Adaptive Cold Start Raw Steps"] = int(runtime_statistics.get("adaptive_cold_start_raw_step_count", 0))
            summary["Adaptive Cold Start Effective Steps"] = int(runtime_statistics.get("adaptive_cold_start_effective_step_count", 0))
            summary["Adaptive Cold Start Raw New Tokens"] = int(runtime_statistics.get("adaptive_cold_start_raw_new_tokens", 0))
            summary["Adaptive Cold Start Raw Draft Tokens"] = int(runtime_statistics.get("adaptive_cold_start_raw_draft_tokens", 0))
            if "adaptive_cold_start_mean_accepted_tokens" in runtime_statistics:
                summary["Adaptive Cold Start Mean Accepted Tokens"] = float(runtime_statistics["adaptive_cold_start_mean_accepted_tokens"])
            if "adaptive_cold_start_token_acceptance_rate" in runtime_statistics:
                summary["Adaptive Cold Start Token Acceptance Rate"] = float(runtime_statistics["adaptive_cold_start_token_acceptance_rate"])
            if "adaptive_cold_start_dynamic_switches" in runtime_statistics:
                summary["Adaptive Cold Start Dynamic Switches"] = int(runtime_statistics["adaptive_cold_start_dynamic_switches"])
            if "adaptive_cold_start_dynamic_layer_switches" in runtime_statistics:
                summary["Adaptive Cold Start Dynamic Layer Switches"] = int(runtime_statistics["adaptive_cold_start_dynamic_layer_switches"])
            if "adaptive_cold_start_final2_action_counts" in runtime_statistics:
                summary["Adaptive Cold Start Final2 Action Counts"] = runtime_statistics["adaptive_cold_start_final2_action_counts"]
            if "adaptive_cold_start_ratio_step_counts" in runtime_statistics:
                summary["Adaptive Cold Start Ratio Step Counts"] = runtime_statistics["adaptive_cold_start_ratio_step_counts"]
            if "adaptive_cold_start_final_ratio_counts" in runtime_statistics:
                summary["Adaptive Cold Start Final Ratio Counts"] = runtime_statistics["adaptive_cold_start_final_ratio_counts"]
        summary["Lyapunov Adaptive Controller"] = bool(runtime_statistics.get("lyapunov_adaptive_controller", False))
        summary["Lyapunov Acceptance Target"] = float(runtime_statistics.get("lyapunov_acceptance_target", 0.92))
        summary["Lyapunov V"] = float(runtime_statistics.get("lyapunov_v", 0.1))
        summary["Lyapunov Switch Cost"] = float(runtime_statistics.get("lyapunov_switch_cost", 0.01))
        summary["Lyapunov Layer Switch Cost"] = float(runtime_statistics.get("lyapunov_layer_switch_cost", runtime_statistics.get("lyapunov_switch_cost", 0.01)))
        summary["Lyapunov Layer Penalty Weight"] = float(runtime_statistics.get("lyapunov_layer_penalty_weight", 0.02))
        summary["Lyapunov Prediction Beta"] = float(runtime_statistics.get("lyapunov_prediction_beta", 0.5))
        summary["Lyapunov Ratio Acceptance Slope"] = float(runtime_statistics.get("lyapunov_ratio_acceptance_slope", 0.2))
        summary["Lyapunov Layer Acceptance Slope"] = float(runtime_statistics.get("lyapunov_layer_acceptance_slope", 0.015))
        summary["Lyapunov Cold Start Penalty"] = float(runtime_statistics.get("lyapunov_cold_start_penalty", 0.03))
        summary["Lyapunov Virtual Queue Final"] = float(runtime_statistics.get("lyapunov_virtual_queue", 0.0))
        summary["Lyapunov Virtual Queue Max"] = float(runtime_statistics.get("lyapunov_virtual_queue_max", 0.0))
        lyapunov_decisions = int(runtime_statistics.get("lyapunov_decision_count", 0))
        summary["Lyapunov Decision Count"] = lyapunov_decisions
        summary["Lyapunov Virtual Queue Mean"] = (
            float(runtime_statistics.get("lyapunov_virtual_queue_sum", 0.0)) / lyapunov_decisions
            if lyapunov_decisions > 0
            else 0.0
        )
        summary["Lyapunov Ratio Up Switches"] = int(runtime_statistics.get("lyapunov_ratio_up_switches", 0))
        summary["Lyapunov Ratio Down Switches"] = int(runtime_statistics.get("lyapunov_ratio_down_switches", 0))
        summary["Lyapunov Keep Decisions"] = int(runtime_statistics.get("lyapunov_keep_decisions", 0))
        summary["Lyapunov Action Counts"] = runtime_statistics.get("lyapunov_action_counts", {})
        summary["Adaptive Question Count"] = int(runtime_statistics.get("adaptive_question_count", 0))
        summary["Adaptive Questions With Switch"] = int(runtime_statistics.get("adaptive_questions_with_switch", 0))
        summary["Adaptive Ratio Step Counts"] = runtime_statistics.get("adaptive_ratio_step_counts", {})
        summary["Adaptive Final Ratio Counts"] = runtime_statistics.get("adaptive_final_ratio_counts", {})
        adaptive_step_config_stats = summarize_adaptive_step_config_stats(
            runtime_statistics.get("adaptive_step_config_stats", {})
        )
        summary["Adaptive Step Trace Count"] = int(runtime_statistics.get("adaptive_step_trace_count", 0))
        summary["Adaptive Step Config Count"] = len(adaptive_step_config_stats)
        summary["Adaptive Step Config Stats"] = adaptive_step_config_stats
    elif (not runtime_statistics or not runtime_statistics.get("dynamic_retain_ratio")) and hasattr(model, "draft_kv_retain_ratio"):
        summary["Draft KV Retain Ratio"] = float(model.draft_kv_retain_ratio)

    if total_scored > 0:
        summary["Accuracy"] = total_correct / total_scored
        summary["Total Correct"] = total_correct
        summary["Total Scored"] = total_scored

    if total_qa_scored > 0:
        summary["Exact Match"] = total_qa_exact_match / total_qa_scored
        summary["F1"] = total_qa_f1 / total_qa_scored

    if total_rouge_scored > 0:
        summary["ROUGE-1"] = total_rouge1 / total_rouge_scored
        summary["ROUGE-2"] = total_rouge2 / total_rouge_scored
        summary["ROUGE-L"] = total_rougeL / total_rouge_scored
        summary["Total ROUGE Scored"] = total_rouge_scored

    summary = {
        "question_index": "__summary__",
        "question_id": "__summary__",
        "task_name": task_name,
        **summary,
    }

    with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
        fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

    if "Mean accepted tokens" in summary:
        print("#Mean accepted tokens:", summary["Mean accepted tokens"])
    if "Token acceptance rate" in summary:
        print("Token acceptance rate:", summary["Token acceptance rate"])
    if "Best Draft KV Retain Ratio" in summary:
        print("Best Draft KV Retain Ratio:", summary["Best Draft KV Retain Ratio"])
    elif "Draft KV Retain Ratio" in summary:
        print("Draft KV Retain Ratio:", summary["Draft KV Retain Ratio"])
    if "Adaptive Total Switches" in summary:
        print("Adaptive Total Switches:", summary["Adaptive Total Switches"])
    if "Adaptive Step Config Count" in summary:
        print("Adaptive Step Config Count:", summary["Adaptive Step Config Count"])
    if "Adaptive Cold Start Time Sec" in summary:
        print("Adaptive Cold Start Time Sec:", summary["Adaptive Cold Start Time Sec"])
    if "FLOPs Trace File" in summary:
        print("FLOPs Trace File:", summary["FLOPs Trace File"])
    if "Estimated Physical Core FLOPs / Output Token" in summary:
        print("Estimated Physical Core FLOPs / Output Token:", summary["Estimated Physical Core FLOPs / Output Token"])
    if "Estimated Logical Core FLOPs / Output Token" in summary:
        print("Estimated Logical Core FLOPs / Output Token:", summary["Estimated Logical Core FLOPs / Output Token"])
    if "Accuracy" in summary:
        print("Accuracy:", summary["Accuracy"])
    if "Exact Match" in summary:
        print("Exact Match:", summary["Exact Match"])
    if "F1" in summary:
        print("F1:", summary["F1"])
    if "ROUGE-1" in summary:
        print("ROUGE-1:", summary["ROUGE-1"])
    if "ROUGE-2" in summary:
        print("ROUGE-2:", summary["ROUGE-2"])
    if "ROUGE-L" in summary:
        print("ROUGE-L:", summary["ROUGE-L"])
