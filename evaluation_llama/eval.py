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
    }
    return aliases.get(name, name)


def is_qa_task(task_name):
    return normalize_task_name(task_name) in {"triviaqa", "natural_questions"}


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

def normalize_number_text(number_text):
    return number_text.replace(",", "")


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


def get_generation_stop_config(task_name):
    task_name = normalize_task_name(task_name)
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
                        "Provide only the summary.\n\n"
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
                + "Summarize the following dialogue in one concise paragraph.\n\n"
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
        prompt_text = (
            prompt_shots
            + "The following is a multiple choice question. "
            + "Answer with only the letter A, B, C, or D.\n\n"
            + f"Question: {prompt['question'].strip()}\n"
            + f"A. {choices[0]}\n"
            + f"B. {choices[1]}\n"
            + f"C. {choices[2]}\n"
            + f"D. {choices[3]}\n"
            + "Answer:"
        )
        end_prompt = "\nAnswer:"
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

    elif task_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        if data_num is not None and data_num < len(dataset):
            dataset = dataset.select(range(data_num))
        data = dataset

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
        output_ids = output_ids[0][len(input_ids[0]):]

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
        if is_qa_task(task_name):
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

        sample_correct = None

        if task_name == "humaneval":
            if isinstance(question, dict) and "task_id" in question:
                sample_record["task_id"] = question["task_id"]

        elif task_name == "gsm8k":
            gold = extract_gsm8k_gold(question["answer"])
            pred = extract_gsm8k_pred(output)
            sample_correct = (pred is not None and gold is not None and pred == gold)

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

    with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
        fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

    if "Mean accepted tokens" in summary:
        print("#Mean accepted tokens:", summary["Mean accepted tokens"])
    if "Token acceptance rate" in summary:
        print("Token acceptance rate:", summary["Token acceptance rate"])
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
