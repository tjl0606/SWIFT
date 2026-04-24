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
    return task_name.strip().lower()


def extract_gsm8k_gold(answer_text):
    # Standard GSM8K answer format usually ends with "#### 72"
    if "####" in answer_text:
        answer_text = answer_text.split("####")[-1]
    answer_text = answer_text.strip().replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    return matches[-1] if matches else None


def extract_gsm8k_pred(output_text):
    text = output_text.strip().replace(",", "")
    if "####" in text:
        text = text.split("####")[-1]
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

    patterns = [
        r"ANSWER\s*[:：]\s*([ABCD])\b",
        r"THE ANSWER IS\s*([ABCD])\b",
        r"\bOPTION\s*([ABCD])\b",
        r"\bCHOICE\s*([ABCD])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]

    matches = re.findall(r"\b([ABCD])\b", text)
    return matches[-1] if matches else None


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

    accept_lengths_tree = []
    total_draft_num = 0
    total_correct = 0
    total_scored = 0

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

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
            **kwargs,
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
        summary["Best Attn Layer Set"] = best_attn_skip_layer_id_set
        summary["Best MLP Layer Set"] = best_mlp_skip_layer_id_set

    if total_scored > 0:
        summary["Accuracy"] = total_correct / total_scored
        summary["Total Correct"] = total_correct
        summary["Total Scored"] = total_scored

    with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
        fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

    if "Mean accepted tokens" in summary:
        print("#Mean accepted tokens:", summary["Mean accepted tokens"])
    if "Token acceptance rate" in summary:
        print("Token acceptance rate:", summary["Token acceptance rate"])
    if "Accuracy" in summary:
        print("Accuracy:", summary["Accuracy"])