import copy
import math
import os.path
import random
import json
import logging
import torch
import numpy as np
from .kv_cache import clone_past_key_values

TOPK = 10  # topk for sparse tree
SMART_KV_SINK_LEN = 4
SMART_KV_OBSERVATION_WINDOW = 32
SMART_KV_RECENT_FRACTION = 0.5
SMART_KV_POOL_KERNEL = 7

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def set_logger(log_path=None):
    """Set the logger to log info in terminal and file `log_path`."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if log_path:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def _clear_swift_tree_state(model):
    """
    Clear tree-specific runtime state before entering a normal verify / draft /
    cache-rebuild path.

    This is the key fix for stale swift_mask causing shape mismatches like:
    old swift_len != rebuilt KV cache length.
    """
    if hasattr(model, "model") and hasattr(model.model, "swift_mask"):
        model.model.swift_mask = None


def get_cache_configuration(file_name='skip_layers.json', model_name='llama-2-13b', task_name='cnndm'):
    """
    Get the cached SWIFT configuration for LLM acceleration.
    """
    if not os.path.exists(file_name):
        print("Cache file not found.")
        return None
    with open(file_name) as f:
        data = json.load(f)
        if f'{model_name}_{task_name}' not in data.keys():
            print("Configuration not found in cache.")
            return None
        else:
            print(f"Use cached configuration in {file_name}.")
            return data[f'{model_name}_{task_name}']['attention'], data[f'{model_name}_{task_name}']['mlp']


def _skip_layer_list(values):
    return [int(x) for x in list(values)]


def get_skip_layer_cache(file_name, cache_key):
    """
    Load a benchmark-specific skip-layer configuration.
    """
    if not os.path.exists(file_name):
        print(f"Skip-layer cache file not found: {file_name}")
        return None

    with open(file_name, encoding="utf-8") as f:
        data = json.load(f)

    if cache_key not in data:
        print(f"Skip-layer cache key not found: {cache_key}")
        return None

    entry = data[cache_key]
    print(f"Use skip-layer cache '{cache_key}' from {file_name}.")
    return entry["attention"], entry["mlp"]


def get_selected_swift_config(file_name, model_name, task_name):
    """
    Load one benchmark-level SWIFT config containing retain ratio and skip layers.
    """
    if not os.path.exists(file_name):
        print(f"Selected SWIFT config file not found: {file_name}")
        return None

    with open(file_name, encoding="utf-8") as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", data)
    entry = benchmarks.get(task_name)
    if entry is None:
        entry = benchmarks.get(f"{model_name}_{task_name}")
    if entry is None:
        print(f"Selected SWIFT config not found for task '{task_name}'.")
        return None

    config_model = entry.get("model_id")
    if config_model is not None and config_model != model_name:
        print(
            f"Selected SWIFT config model mismatch: expected {model_name}, "
            f"found {config_model}."
        )
        return None

    print(f"Use selected SWIFT config for '{task_name}' from {file_name}.")
    return entry


def save_skip_layer_cache(file_name, cache_key, attn_skip_layers, mlp_skip_layers, metadata=None):
    """
    Save a benchmark-specific skip-layer configuration.
    """
    cache_dir = os.path.dirname(file_name)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(file_name):
        with open(file_name, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data[cache_key] = {
        "attention": _skip_layer_list(attn_skip_layers),
        "mlp": _skip_layer_list(mlp_skip_layers),
        "metadata": metadata or {},
    }

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Saved skip-layer cache '{cache_key}' to {file_name}.")


def get_choices_list(prob_list, logits_processor=None):
    """
    Generate tree choices list based on the provided confidence.
    """
    choices_list = []
    if logits_processor is not None:
        candidate_set = [1, 3, 5, 10]
    else:
        candidate_set = [3, 5, 8, 10]
    for idx, item in enumerate(prob_list):
        if item > 0.95:
            candidate_num = candidate_set[0]
        elif item > 0.8:
            candidate_num = candidate_set[1]
        elif item > 0.5:
            candidate_num = candidate_set[2]
        else:
            candidate_num = candidate_set[3]
        choices_list.extend([[0] * idx + [i] for i in range(candidate_num)])
    return choices_list


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    """
    Prepare the logits processor based on the provided parameters.
    """
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    """
    return path + [pad_value] * (length - len(path))


def generate_swift_buffers(swift_choices, device="cuda"):
    """
    Generate buffers for the swift structure based on the provided choices.
    """
    sorted_swift_choices = sorted(swift_choices, key=lambda x: (len(x), x))
    swift_len = len(sorted_swift_choices) + 1

    depth_counts = []
    prev_depth = 0
    for path in sorted_swift_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    swift_attn_mask = torch.eye(swift_len, swift_len)
    swift_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_swift_choice = sorted_swift_choices[start + j]
            if len(cur_swift_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_swift_choice) - 1):
                ancestor_idx.append(sorted_swift_choices.index(cur_swift_choice[:c + 1]) + 1)
            swift_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    swift_tree_indices = torch.zeros(swift_len, dtype=torch.long)
    swift_p_indices = [0 for _ in range(swift_len - 1)]
    swift_b_indices = [[] for _ in range(swift_len - 1)]
    swift_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        b = []
        for j in range(depth_counts[i]):
            cur_swift_choice = sorted_swift_choices[start + j]
            swift_tree_indices[start + j + 1] = cur_swift_choice[-1] + TOPK * i + 1
            swift_p_indices[start + j] = 0
            if len(b) > 0:
                swift_b_indices[start + j] = copy.deepcopy(b)
            else:
                swift_b_indices[start + j] = []
            b.append(cur_swift_choice[-1] + TOPK * i + 1)
        start += depth_counts[i]

    swift_p_indices = [-1] + swift_p_indices
    swift_position_ids = torch.zeros(swift_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        swift_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_swift_choices)):
        cur_swift_choice = sorted_swift_choices[-i - 1]
        retrieve_indice = []
        if cur_swift_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_swift_choice)):
                retrieve_indice.append(sorted_swift_choices.index(cur_swift_choice[:c + 1]))
                retrieve_paths.append(cur_swift_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat(
        [torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
        dim=1
    )

    maxitem = retrieve_indices.max().item() + 5

    def custom_sort(lst):
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys

    retrieve_indices = retrieve_indices.tolist()
    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

    swift_p_indices = torch.tensor(swift_p_indices)
    swift_p_indices_new = swift_p_indices[retrieve_indices]
    swift_p_indices_new = swift_p_indices_new.tolist()

    swift_b_indices = [[]] + swift_b_indices
    swift_b_indices_new = []
    for ib in range(retrieve_indices.shape[0]):
        iblist = []
        for jb in range(retrieve_indices.shape[1]):
            index = retrieve_indices[ib, jb]
            if index == -1:
                iblist.append([])
            else:
                b = swift_b_indices[index]
                if len(b) > 0:
                    bt = []
                    for bi in b:
                        bt.append(torch.where(swift_tree_indices == bi)[0].item())
                    iblist.append(torch.tensor(bt, device=device))
                else:
                    iblist.append(b)
        swift_b_indices_new.append(iblist)

    swift_buffers = {
        "swift_attn_mask": swift_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": swift_tree_indices,
        "swift_position_ids": swift_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    swift_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in swift_buffers.items()
    }
    swift_buffers["p_indices"] = swift_p_indices_new
    swift_buffers["b_indices"] = swift_b_indices_new
    return swift_buffers


def initialize_swift(input_ids, model, max_new_tokens, past_key_values, past_key_values_data, current_length_data,
                     logits_processor=None, draft_token_num=None):
    """
    Initializes the swift structure for a given model.
    """
    with torch.inference_mode():
        # Normal verify path: must not reuse a stale tree mask
        _clear_swift_tree_state(model)

        outputs, logits = swift_verify(model, input_ids, past_key_values=past_key_values)

        if logits_processor is not None:
            last_logits = logits[:, -1]
            last_logits = logits_processor(None, last_logits)
            probabilities = torch.nn.functional.softmax(last_logits, dim=1)
            sample_token = torch.multinomial(probabilities, 1)
        else:
            sample_token = torch.argmax(logits[:, -1])
            sample_token = sample_token[None, None]

        swift_logits, top1_prob = swift_draft(
            model,
            input_ids=sample_token,
            full_input_ids=input_ids,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            draft_token_num=draft_token_num,
        )
    return swift_logits, sample_token, top1_prob


def swift_verify(
        model,
        input_ids=None,
        past_key_values=None,
        position_ids=None,
):
    """
    Verify the swift structure using the provided model and input.
    """
    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        orig = model.lm_head(outputs[0])

    return outputs, orig


def sample(logits, logits_processor, k=1):
    """
    Sample from the provided logits using the specified processor.
    """
    logits = logits.view(-1, logits.size(-1))
    logits = logits_processor(None, logits)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    sampled_indices = torch.multinomial(probabilities, k, replacement=False)
    sampled_probs = torch.gather(probabilities, -1, sampled_indices)

    cumulative_sum = torch.cumsum(sampled_probs, dim=-1)
    cumulative_sum = torch.cat(
        (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]),
        dim=-1
    )

    sampled_probs = sampled_probs / (1 - cumulative_sum)
    sampled_probs[torch.isinf(sampled_probs)] = -1
    sampled_probs[torch.isnan(sampled_probs)] = -1

    sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

    return sampled_indices, sampled_probs, probabilities


def _clone_past_key_values_data(past_key_values_data):
    return [x.clone() for x in past_key_values_data]


def _copy_past_key_values(model, past_key_values_data, current_length_data):
    draft_past_key_values_data = _clone_past_key_values_data(past_key_values_data)
    draft_current_length_data = current_length_data.clone()
    draft_past_key_values = clone_past_key_values(
        model, draft_past_key_values_data, draft_current_length_data
    )
    return draft_past_key_values_data, draft_current_length_data, draft_past_key_values


def _pool_token_scores(scores, kernel_size=SMART_KV_POOL_KERNEL):
    if scores is None or scores.numel() == 0 or kernel_size <= 1:
        return scores
    padding = kernel_size // 2
    pooled = torch.nn.functional.max_pool1d(
        scores.view(1, 1, -1),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    ).view(-1)
    return pooled[: scores.numel()]


def _get_observation_attention_scores(
        model,
        full_input_ids,
        past_key_values_data,
        current_length_data,
        observation_window=SMART_KV_OBSERVATION_WINDOW,
):
    """
    Estimate token importance from the draft model's attention over a short
    observation window at the end of the current sequence.
    """
    if full_input_ids is None or full_input_ids.shape[1] <= 1:
        return None

    full_len = full_input_ids.shape[1]
    obs_len = min(observation_window, full_len)
    prefix_len = full_len - obs_len
    device = past_key_values_data[0].device if len(past_key_values_data) > 0 else full_input_ids.device
    obs_input_ids = full_input_ids[:, prefix_len:].to(device)
    obs_position_ids = torch.arange(prefix_len, full_len, device=device).unsqueeze(0)

    score_past_key_values = None
    if prefix_len > 0:
        score_past_key_values_data = _clone_past_key_values_data(past_key_values_data)
        score_current_length_data = current_length_data.clone()
        score_current_length_data.fill_(prefix_len)
        score_past_key_values = clone_past_key_values(
            model, score_past_key_values_data, score_current_length_data
        )

    _clear_swift_tree_state(model)

    try:
        with torch.inference_mode():
            with model.self_draft():
                outputs = model.model(
                    input_ids=obs_input_ids,
                    attention_mask=None,
                    past_key_values=score_past_key_values,
                    position_ids=obs_position_ids,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
    except Exception as exc:
        logging.warning("Smart draft KV scoring failed; falling back to heuristic compression: %s", exc)
        _clear_swift_tree_state(model)
        return None

    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        return None

    scores = torch.zeros(full_len, device=device, dtype=torch.float32)
    layer_count = 0
    for attn in attentions:
        if attn is None:
            continue
        attn = attn.detach().float()
        if attn.shape[-1] < full_len:
            continue
        scores += attn[0, :, :, :full_len].mean(dim=(0, 1))
        layer_count += 1

    if layer_count == 0:
        return None

    scores = scores / layer_count
    return _pool_token_scores(scores)


def _fill_indices_evenly(candidate_indices, num_to_fill):
    if num_to_fill <= 0 or candidate_indices.numel() == 0:
        return candidate_indices.new_empty(0)
    if candidate_indices.numel() <= num_to_fill:
        return candidate_indices
    pick = torch.linspace(
        0,
        candidate_indices.numel() - 1,
        steps=num_to_fill,
        device=candidate_indices.device,
    ).round().long()
    return candidate_indices[pick].unique(sorted=True)


def _select_smart_kv_indices(
        full_input_ids,
        keep_len,
        scores=None,
        sink_len=SMART_KV_SINK_LEN,
        recent_fraction=SMART_KV_RECENT_FRACTION,
):
    """
    Select a fixed-size KV budget using sink + recent + attention-heavy tokens.
    The final indices are sorted so the compressed cache keeps causal order.
    """
    full_len = full_input_ids.shape[1]
    device = full_input_ids.device

    if keep_len >= full_len:
        return torch.arange(full_len, device=device, dtype=torch.long)

    keep_mask = torch.zeros(full_len, device=device, dtype=torch.bool)

    sink_count = min(sink_len, keep_len, full_len)
    if sink_count > 0:
        keep_mask[:sink_count] = True

    remaining = keep_len - int(keep_mask.sum().item())
    if remaining > 0:
        recent_count = max(1, int(keep_len * recent_fraction))
        recent_count = min(recent_count, remaining, full_len)
        keep_mask[full_len - recent_count:] = True

    remaining = keep_len - int(keep_mask.sum().item())
    if remaining > 0:
        candidate_indices = torch.nonzero(~keep_mask, as_tuple=True)[0]
        if scores is not None and scores.numel() == full_len:
            candidate_scores = scores.to(device=device)[candidate_indices]
            topk = min(remaining, candidate_indices.numel())
            selected = candidate_indices[torch.topk(candidate_scores, k=topk).indices]
        else:
            selected = _fill_indices_evenly(candidate_indices, remaining)
        keep_mask[selected] = True

    remaining = keep_len - int(keep_mask.sum().item())
    if remaining > 0:
        candidate_indices = torch.nonzero(~keep_mask, as_tuple=True)[0]
        keep_mask[candidate_indices[-remaining:]] = True

    return torch.nonzero(keep_mask, as_tuple=True)[0]


def _copy_selected_kv_cache(model, past_key_values_data, current_length_data, keep_indices):
    draft_past_key_values_data = []
    for past_key_values_data_item in past_key_values_data:
        compressed_data = past_key_values_data_item.clone()
        compressed_data.zero_()
        local_indices = keep_indices.to(past_key_values_data_item.device)
        selected_data = past_key_values_data_item.index_select(-2, local_indices)
        compressed_data[..., : keep_indices.numel(), :].copy_(selected_data, non_blocking=True)
        draft_past_key_values_data.append(compressed_data)

    draft_current_length_data = current_length_data.clone()
    draft_current_length_data.fill_(keep_indices.numel())
    draft_past_key_values = clone_past_key_values(
        model, draft_past_key_values_data, draft_current_length_data
    )
    return draft_past_key_values_data, draft_current_length_data, draft_past_key_values


@torch.no_grad()
def swift_draft(
        model,
        input_ids=None,
        full_input_ids=None,
        new_token_num=0,
        past_key_values_data=None,
        current_length_data=None,
        max_new_tokens=1024,
        position_ids=None,
        max_step_draft=25,
        logits_processor=None,
        stop_threshold=0.8,
        draft_token_num=None,
):
    """
    Draft new tokens using the swift structure.
    """
    # Normal draft path must not reuse a stale tree mask from the previous tree decoding round
    _clear_swift_tree_state(model)

    if hasattr(model, "draft_kv_compress") and model.draft_kv_compress and model.draft_kv_retain_ratio < 0.9999:
        draft_past_key_values_data, draft_current_length_data, draft_past_key_values = rebuild_compressed_draft_cache(
            model=model,
            full_input_ids=full_input_ids,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            retain_ratio=model.draft_kv_retain_ratio,
            min_retain_tokens=16,
            # sink_tokens=4,
        )
    else:
        draft_past_key_values_data, draft_current_length_data, draft_past_key_values = _copy_past_key_values(
            model, past_key_values_data, current_length_data
        )

    # Be explicit again after rebuilding / cloning cache
    _clear_swift_tree_state(model)

    ss_token, ss_prob, ss_op, top1_prob = [], [], [], []
    draft_position = full_input_ids.shape[1] if full_input_ids is not None else None
    use_fixed_draft_tokens = draft_token_num is not None
    if use_fixed_draft_tokens:
        if draft_token_num <= 0:
            raise ValueError("draft_token_num must be a positive integer when set.")
        max_step_draft = min(max_step_draft, draft_token_num)

    with torch.inference_mode():
        for step_draft in range(max_step_draft):
            step_position_ids = position_ids
            if step_position_ids is None and draft_position is not None:
                step_position_ids = torch.arange(
                    draft_position,
                    draft_position + input_ids.shape[1],
                    dtype=torch.long,
                    device=input_ids.device,
                ).unsqueeze(0)

            with model.self_draft():
                draft_outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    past_key_values=draft_past_key_values,
                    position_ids=step_position_ids,
                )
            current_draft_logits = model.lm_head(draft_outputs[0])

            if logits_processor is not None:
                topk_index, topk_prob, op = sample(current_draft_logits, logits_processor, k=TOPK)
                input_ids = topk_index[:, 0].unsqueeze(0)
            else:
                top = torch.topk(current_draft_logits, TOPK, dim=-1)
                topk_index, topk_prob = top.indices, top.values
                input_ids = topk_index[:, :, 0]
                op = None

            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)

            origin_draft_probs = current_draft_logits.softmax(-1)
            argmax_prob = torch.gather(origin_draft_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)
            current_threshold = argmax_prob.item()
            top1_prob.append(current_threshold)

            reached_generation_limit = new_token_num + step_draft + 2 >= max_new_tokens
            if reached_generation_limit or (not use_fixed_draft_tokens and current_threshold < stop_threshold):
                break

            if draft_position is not None:
                draft_position += input_ids.shape[1]

    return (torch.cat(ss_token), torch.cat(ss_prob), ss_op), top1_prob


def reset_swift_mode(model):
    """
    Resets the swift settings to their initial state.
    """
    _clear_swift_tree_state(model)


def rebuild_compressed_draft_cache(
        model,
        full_input_ids,
        past_key_values_data,
        current_length_data,
        retain_ratio=1.0,
        min_retain_tokens=16,
):
    """
    Rebuild a draft cache by retaining sink, recent, and attention-heavy tokens.
    """
    if full_input_ids is None:
        return _copy_past_key_values(model, past_key_values_data, current_length_data)

    full_len = full_input_ids.shape[1]
    keep_len = max(min_retain_tokens, int(full_len * retain_ratio))
    keep_len = min(keep_len, full_len)

    if keep_len == full_len:
        return _copy_past_key_values(model, past_key_values_data, current_length_data)

    _clear_swift_tree_state(model)
    scores = _get_observation_attention_scores(
        model=model,
        full_input_ids=full_input_ids,
        past_key_values_data=past_key_values_data,
        current_length_data=current_length_data,
    )
    keep_indices = _select_smart_kv_indices(
        full_input_ids=full_input_ids,
        keep_len=keep_len,
        scores=scores,
    )
    return _copy_selected_kv_cache(
        model=model,
        past_key_values_data=past_key_values_data,
        current_length_data=current_length_data,
        keep_indices=keep_indices,
    )


def reset_past_key_values(past_key_values):
    """
    Resets the current lengths in the past key-values to zero.
    """
    for i in range(len(past_key_values)):
        for j in range(2):
            past_key_values[i][j].current_length.fill_(0)
    return past_key_values


def generate_candidates(swift_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    """
    Generate candidates based on provided logits and indices.
    """
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]
    candidates_swift_logits = swift_logits[0]

    candidates = torch.cat([candidates_logit, candidates_swift_logits.view(-1)], dim=-1)
    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)],
        dim=0
    )

    cart_candidates = tree_candidates_ext[retrieve_indices]

    if logits_processor is not None:
        candidates_tree_prob = swift_logits[1]
        candidates_prob = torch.cat(
            [torch.ones(1, device=candidates_tree_prob.device, dtype=torch.float32), candidates_tree_prob.view(-1)],
            dim=-1
        )

        tree_candidates_prob = candidates_prob[tree_indices]
        tree_candidates_prob_ext = torch.cat(
            [tree_candidates_prob, torch.ones((1), dtype=torch.float32, device=tree_candidates_prob.device)],
            dim=0
        )
        cart_candidates_prob = tree_candidates_prob_ext[retrieve_indices]
    else:
        cart_candidates_prob = None

    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, cart_candidates_prob, tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        swift_position_ids,
        input_ids,
        retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    """
    position_ids = swift_position_ids + input_ids.shape[1]

    outputs, tree_logits = swift_verify(
        model,
        tree_candidates,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    logits = tree_logits[0, retrieve_indices]

    return logits, outputs


def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        cart_candidates_prob,
        op,
        p_indices,
        tree_candidates,
        b_indices
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits
    and choose the best candidate.
    """
    if logits_processor is None:
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()

        if accept_length == 0:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]
    else:
        cart_candidates_prob = cart_candidates_prob.to(logits.device)
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        adjustflag = False
        gtp = None

        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []

            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = cart_candidates_prob[j, i]
                    if qx <= 0:
                        continue
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        q = op[i - 1][p_indices[j][i]].clone()
                        b = b_indices[j][i]
                        if len(b) > 0:
                            mask = tree_candidates[0][b]
                            q[mask] = 0
                            q = q / q.sum()
                        max_id = gtp.argmax()
                        gtp = gtp - q
                        gtp[gtp < 0] = 0
                        if torch.equal(gtp.cpu(), torch.zeros(gtp.shape)):
                            gtp[max_id] = 1
                        gtp = gtp / (gtp.sum() + 1e-6)
                        adjustflag = True

        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)

        return best_candidate, accept_length - 1, sample_p


def get_next_point_to_probe(attn_skip_layers, mlp_skip_layers, num_hidden_layers=40):
    """
    Get the next point to probe of Bayes Optimization based on the skip layers.
    """
    next_point_to_probe = {}
    for i in range(num_hidden_layers - 2):
        if (i + 1) in attn_skip_layers:
            next_point_to_probe[f"x{i}"] = 1.0
        else:
            next_point_to_probe[f"x{i}"] = 0.0

    for i in range(num_hidden_layers - 2, (num_hidden_layers - 2) * 2):
        if (i - (num_hidden_layers - 2) + 1) in mlp_skip_layers:
            next_point_to_probe[f"x{i}"] = 1.0
        else:
            next_point_to_probe[f"x{i}"] = 0.0

    return next_point_to_probe


def layer_bayes_search(optimizer, utility, num_skip_layers=34, num_hidden_layers=40):
    """
    Perform Bayesian optimization to select the next point to probe.
    """
    next_point_to_probe = optimizer.suggest(utility)
    sorted_point = sorted(next_point_to_probe.items(), reverse=True, key=lambda item: item[1])
    skip_layer_list = [k for (k, v) in sorted_point[:num_skip_layers]]
    attn_skip_layers = []
    mlp_skip_layers = []

    for i in range(num_hidden_layers - 2):
        if f"x{i}" in skip_layer_list:
            attn_skip_layers.append(i + 1)
    for i in range(num_hidden_layers - 2, (num_hidden_layers - 2) * 2):
        if f"x{i}" in skip_layer_list:
            mlp_skip_layers.append(i - (num_hidden_layers - 2) + 1)

    return next_point_to_probe, attn_skip_layers, mlp_skip_layers


def layer_random_search(num_skip_layers=34, num_hidden_layers=40):
    """
    Randomly select layers for skipping, always keep the first and last layer.
    """
    skip_layer_list = np.random.choice((num_hidden_layers - 2) * 2, num_skip_layers, replace=False)
    attn_skip_layers = []
    mlp_skip_layers = []

    for i in range(num_hidden_layers - 2):
        if i in skip_layer_list:
            attn_skip_layers.append(i + 1)
    for i in range(num_hidden_layers - 2, (num_hidden_layers - 2) * 2):
        if i in skip_layer_list:
            mlp_skip_layers.append(i - (num_hidden_layers - 2) + 1)

    return attn_skip_layers, mlp_skip_layers


def _retain_ratio_key(retain_ratio):
    return str(float(retain_ratio))


def _ensure_dynamic_retain_state(statistics):
    retain_ratio_grid = statistics.get("retain_ratio_grid", [statistics.get("draft_kv_retain_ratio", 1.0)])
    retain_ratio_state = statistics.setdefault("retain_ratio_state", {})
    for retain_ratio in retain_ratio_grid:
        key = _retain_ratio_key(retain_ratio)
        retain_ratio_state.setdefault(
            key,
            {
                "trials": 0,
                "score_sum": 0.0,
                "utility_sum": 0.0,
                "best_score": 0.0,
                "best_utility": -1e30,
                "best_attention": [],
                "best_mlp": [],
            },
        )
    statistics.setdefault("origin_utility", float("-inf"))
    return retain_ratio_state


def _select_dynamic_retain_ratio(statistics, model=None):
    retain_ratio_grid = statistics.get("retain_ratio_grid", [statistics.get("draft_kv_retain_ratio", 1.0)])
    retain_ratio_state = _ensure_dynamic_retain_state(statistics)
    statistics.setdefault("retain_stage", "warmup")

    while True:
        retain_stage = statistics.get("retain_stage", "warmup")
        if retain_stage == "warmup":
            warmup_rounds = int(statistics.get("retain_warmup_rounds", 1))
            under_warmup = [
                retain_ratio
                for retain_ratio in retain_ratio_grid
                if retain_ratio_state[_retain_ratio_key(retain_ratio)]["trials"] < warmup_rounds
            ]
            if under_warmup:
                return min(
                    under_warmup,
                    key=lambda retain_ratio: retain_ratio_state[_retain_ratio_key(retain_ratio)]["trials"],
                )

            _start_dynamic_retain_candidate_refine(statistics)
            continue

        if retain_stage == "candidate_refine":
            candidate_ratios = statistics.get("retain_candidate_ratios", retain_ratio_grid)
            refine_start_trials = statistics.get("retain_refine_start_trials", {})
            refine_rounds = int(statistics.get("retain_refine_rounds", 0))
            under_refine = [
                retain_ratio
                for retain_ratio in candidate_ratios
                if (
                    retain_ratio_state[_retain_ratio_key(retain_ratio)]["trials"]
                    - int(refine_start_trials.get(_retain_ratio_key(retain_ratio), 0))
                ) < refine_rounds
            ]
            if under_refine:
                return min(
                    under_refine,
                    key=lambda retain_ratio: (
                        retain_ratio_state[_retain_ratio_key(retain_ratio)]["trials"]
                        - int(refine_start_trials.get(_retain_ratio_key(retain_ratio), 0))
                    ),
                )

            _start_dynamic_retain_final_refine(statistics, model=model)
            continue

        if retain_stage == "final_refine":
            return statistics.get("retain_final_ratio", statistics.get("draft_kv_retain_ratio", retain_ratio_grid[0]))

        return statistics.get("retain_final_ratio", statistics.get("draft_kv_retain_ratio", retain_ratio_grid[0]))


def _rank_dynamic_retain_ratios(statistics, ratios=None):
    retain_ratio_grid = ratios or statistics.get("retain_ratio_grid", [statistics.get("draft_kv_retain_ratio", 1.0)])
    retain_ratio_state = _ensure_dynamic_retain_state(statistics)
    return sorted(
        retain_ratio_grid,
        key=lambda retain_ratio: (
            -float(retain_ratio_state[_retain_ratio_key(retain_ratio)].get("best_utility", -1e30)),
            float(retain_ratio),
        ),
    )


def _start_dynamic_retain_candidate_refine(statistics):
    retain_ratio_grid = statistics.get("retain_ratio_grid", [statistics.get("draft_kv_retain_ratio", 1.0)])
    retain_ratio_state = _ensure_dynamic_retain_state(statistics)
    top_k = max(1, int(statistics.get("retain_filter_top_k", 3)))
    candidate_ratios = _rank_dynamic_retain_ratios(statistics, retain_ratio_grid)[:min(top_k, len(retain_ratio_grid))]

    statistics["retain_candidate_ratios"] = [float(retain_ratio) for retain_ratio in candidate_ratios]
    statistics["retain_refine_start_trials"] = {
        _retain_ratio_key(retain_ratio): int(retain_ratio_state[_retain_ratio_key(retain_ratio)]["trials"])
        for retain_ratio in candidate_ratios
    }
    statistics["retain_stage"] = "candidate_refine"
    logging.info("Dynamic retain Stage 2 candidates: {}".format(statistics["retain_candidate_ratios"]))


def _start_dynamic_retain_final_refine(statistics, model=None):
    retain_ratio_state = _ensure_dynamic_retain_state(statistics)
    candidate_ratios = statistics.get("retain_candidate_ratios")
    if not candidate_ratios:
        candidate_ratios = statistics.get("retain_ratio_grid", [statistics.get("draft_kv_retain_ratio", 1.0)])

    best_utility = max(
        float(retain_ratio_state[_retain_ratio_key(retain_ratio)].get("best_utility", -1e30))
        for retain_ratio in candidate_ratios
    )
    final_tolerance = float(statistics.get("retain_final_tolerance", 0.05))
    eligible_ratios = [
        retain_ratio
        for retain_ratio in candidate_ratios
        if best_utility - float(retain_ratio_state[_retain_ratio_key(retain_ratio)].get("best_utility", -1e30))
        <= final_tolerance
    ]
    final_ratio = min(eligible_ratios)

    statistics["retain_final_ratio"] = float(final_ratio)
    statistics["retain_final_start_trials"] = int(retain_ratio_state[_retain_ratio_key(final_ratio)]["trials"])
    statistics["retain_stage"] = "final_refine"
    _set_dynamic_retain_best(statistics, final_ratio, model=model)
    logging.info("Dynamic retain Stage 3 final ratio: {}".format(final_ratio))


def _set_dynamic_retain_best(statistics, retain_ratio, model=None):
    retain_ratio_state = _ensure_dynamic_retain_state(statistics)
    state = retain_ratio_state[_retain_ratio_key(retain_ratio)]
    statistics["draft_kv_retain_ratio"] = float(retain_ratio)
    statistics["best_retain_ratio"] = float(retain_ratio)
    statistics["best_retain_score"] = float(state.get("best_score", 0.0))
    statistics["best_retain_utility"] = float(state.get("best_utility", -1e30))
    statistics["origin_score"] = float(state.get("best_score", 0.0))
    statistics["origin_utility"] = float(state.get("best_utility", -1e30))

    if model is not None:
        model.draft_kv_retain_ratio = float(retain_ratio)
        if state.get("best_attention") or state.get("best_mlp"):
            model.set_skip_layers(state.get("best_attention", []), state.get("best_mlp", []))


def _maybe_finish_dynamic_retain(statistics):
    if statistics.get("retain_stage") != "final_refine":
        return

    retain_ratio_state = _ensure_dynamic_retain_state(statistics)
    final_ratio = statistics.get("retain_final_ratio")
    if final_ratio is None:
        return

    final_rounds = int(statistics.get("final_layer_refine_rounds", 0))
    final_done = (
        retain_ratio_state[_retain_ratio_key(final_ratio)]["trials"]
        - int(statistics.get("retain_final_start_trials", 0))
    )
    if final_done >= final_rounds:
        statistics["retain_stage"] = "done"
        statistics["optimization"] = False
        logging.info("Dynamic retain search finished after final layer refinement.")


def _dynamic_retain_utility(score, retain_ratio, statistics):
    utility_mode = statistics.get("retain_utility_mode", "relative")
    target_score = float(statistics.get("retain_target_score", statistics.get("max_score", 0.93)))
    penalty = float(statistics.get("retain_utility_lambda", 1.0))
    compression_weight = float(statistics.get("retain_compression_weight", 0.5))
    score_tolerance = float(statistics.get("retain_score_tolerance", 0.05))
    compression_gain = max(0.0, 1.0 - float(retain_ratio))

    if utility_mode == "absolute":
        if score >= target_score:
            return compression_gain + 0.01 * (score - target_score)
        return -penalty * (target_score - score)

    if utility_mode == "additive":
        return score + compression_weight * compression_gain

    reference_score = _dynamic_retain_reference_score(statistics)
    score_deficit = max(0.0, reference_score - float(score) - score_tolerance)
    return float(score) + compression_weight * compression_gain - penalty * score_deficit


def _dynamic_retain_reference_score(statistics):
    retain_ratio_grid = statistics.get("retain_ratio_grid", [statistics.get("draft_kv_retain_ratio", 1.0)])
    retain_ratio_state = _ensure_dynamic_retain_state(statistics)
    baseline_ratio = max(retain_ratio_grid)
    baseline_state = retain_ratio_state.get(_retain_ratio_key(baseline_ratio))
    if baseline_state and baseline_state["trials"] > 0:
        return float(baseline_state["best_score"])

    best_scores = [
        float(state["best_score"])
        for state in retain_ratio_state.values()
        if state["trials"] > 0
    ]
    if best_scores:
        return max(best_scores)
    return 0.0


def _record_dynamic_retain_candidate(statistics, retain_ratio, score, candidate_utility,
                                     attn_skip_layers, mlp_skip_layers):
    retain_ratio_state = _ensure_dynamic_retain_state(statistics)
    state = retain_ratio_state[_retain_ratio_key(retain_ratio)]
    state["trials"] += 1
    state["score_sum"] += float(score)
    state["utility_sum"] += float(candidate_utility)
    if candidate_utility > state["best_utility"]:
        state["best_score"] = float(score)
        state["best_utility"] = float(candidate_utility)
        state["best_attention"] = _skip_layer_list(attn_skip_layers)
        state["best_mlp"] = _skip_layer_list(mlp_skip_layers)


def _get_retain_ratio_optimizer(optimizer, retain_ratio):
    if not isinstance(optimizer, dict):
        return optimizer
    if retain_ratio in optimizer:
        return optimizer[retain_ratio]
    for key, value in optimizer.items():
        if abs(float(key) - float(retain_ratio)) < 1e-9:
            return value
    return None


def _optimizer_observation_count(optimizer):
    if optimizer is None:
        return 0
    if hasattr(optimizer, "res"):
        return len(optimizer.res)
    if hasattr(optimizer, "space"):
        try:
            return len(optimizer.space)
        except TypeError:
            return 0
    return 0


def swift_optimization(model, output_ids, full_input_ids, input_past_key_values_data,
            input_current_length_data, new_token_num, statistics, optimizer=None, utility=None, position_ids=None):
    """
    Perform an optimization to find the optimal layer set for the model based on the draft matchness.
    """
    generate_ids = output_ids.clone()

    cur_past_key_values_data = []
    for i in range(len(input_past_key_values_data)):
        cur_past_key_values_data.append(input_past_key_values_data[i].clone())
    cur_current_length_data = input_current_length_data.clone()

    dynamic_retain_ratio = statistics.get("dynamic_retain_ratio", False)
    candidate_retain_ratio = statistics.get("draft_kv_retain_ratio", 1.0)
    candidate_optimizer = optimizer
    candidate_ratio_trials = statistics["opt_iter"]
    if dynamic_retain_ratio:
        candidate_retain_ratio = _select_dynamic_retain_ratio(statistics, model=model)
        candidate_optimizer = _get_retain_ratio_optimizer(optimizer, candidate_retain_ratio)
        candidate_ratio_state = _ensure_dynamic_retain_state(statistics)[_retain_ratio_key(candidate_retain_ratio)]
        candidate_ratio_trials = candidate_ratio_state["trials"]

    use_compressed_optimization_kv = (
        statistics.get("optimize_with_compressed_draft_kv", True)
        and statistics.get("draft_kv_compress", False)
        and candidate_retain_ratio < 0.9999
    )

    if use_compressed_optimization_kv:
        cur_past_key_values_data, cur_current_length_data, input_past_key_values = rebuild_compressed_draft_cache(
            model=model,
            full_input_ids=full_input_ids,
            past_key_values_data=cur_past_key_values_data,
            current_length_data=cur_current_length_data,
            retain_ratio=candidate_retain_ratio,
        )
    else:
        input_past_key_values = clone_past_key_values(model, cur_past_key_values_data, cur_current_length_data)

    origin_attn_skip_layer_id_set, origin_mlp_skip_layer_id_set = model.get_skip_layers()
    skip_layer_num = int((model.config.num_hidden_layers - 2) * 2 * statistics["skip_ratio"])

    if ((candidate_ratio_trials + 1) % statistics["bayes_interval"] == 0
            and statistics["bayes"]
            and _optimizer_observation_count(candidate_optimizer) > 0):
        logging.info("*" * 30 + "Bayes Search!" + "*" * 30)
        next_point_to_probe, _attn_skip_layer_id_set, _mlp_skip_layer_id_set = layer_bayes_search(
            candidate_optimizer, utility, num_skip_layers=skip_layer_num, num_hidden_layers=model.config.num_hidden_layers)
    else:
        _attn_skip_layer_id_set, _mlp_skip_layer_id_set = layer_random_search(
            num_skip_layers=skip_layer_num, num_hidden_layers=model.config.num_hidden_layers)
        next_point_to_probe = get_next_point_to_probe(
            _attn_skip_layer_id_set, _mlp_skip_layer_id_set, model.config.num_hidden_layers
        )

    model.set_skip_layers(_attn_skip_layer_id_set, _mlp_skip_layer_id_set)

    # Optimization path is also a normal draft forward, not tree decoding
    _clear_swift_tree_state(model)

    with torch.inference_mode():
        with model.self_draft():
            step_end = statistics["context_window"] + 1
            optimization_input_ids = generate_ids[:, :step_end]
            optimization_position_ids = position_ids
            if optimization_position_ids is None and full_input_ids is not None:
                prefix_len = full_input_ids.shape[1]
                optimization_position_ids = torch.arange(
                    prefix_len,
                    prefix_len + optimization_input_ids.shape[1],
                    dtype=torch.long,
                    device=optimization_input_ids.device,
                ).unsqueeze(0)

            parallel_draft_output = model.model(
                input_ids=optimization_input_ids,
                attention_mask=None,
                past_key_values=input_past_key_values,
                position_ids=optimization_position_ids
            )

    parallel_draft_logits = model.lm_head(parallel_draft_output[0])
    parallel_draft_output_ids = torch.argmax(parallel_draft_logits, dim=-1)
    verified_token_num = (
        parallel_draft_output_ids[:, :-1] == generate_ids[:, 1:step_end].to(parallel_draft_output_ids.device)
    ).sum(-1).item()
    drafted_token_num = generate_ids[:, 1:step_end].size(-1)
    score = verified_token_num / drafted_token_num
    logging.info('opt_iter {}, retain_ratio {:.4f}, matchness {:.4f}'.format(
        statistics["opt_iter"], candidate_retain_ratio, score
    ))

    if candidate_optimizer is not None:
        candidate_optimizer.register(params=next_point_to_probe, target=score)

    if dynamic_retain_ratio:
        candidate_utility = _dynamic_retain_utility(score, candidate_retain_ratio, statistics)
        _record_dynamic_retain_candidate(
            statistics,
            candidate_retain_ratio,
            score,
            candidate_utility,
            _attn_skip_layer_id_set,
            _mlp_skip_layer_id_set,
        )
        logging.info('retain utility {:.4f} for ratio {:.4f}'.format(candidate_utility, candidate_retain_ratio))

        if candidate_utility > statistics.get("origin_utility", float("-inf")):
            logging.info("=" * 30 + 'utility changed from {:.4f} to {:.4f}'.format(
                statistics.get("origin_utility", float("-inf")), candidate_utility
            ) + "=" * 30)
            statistics["origin_utility"] = candidate_utility
            statistics["origin_score"] = score
            statistics["draft_kv_retain_ratio"] = candidate_retain_ratio
            statistics["best_retain_ratio"] = candidate_retain_ratio
            statistics["best_retain_score"] = score
            statistics["best_retain_utility"] = candidate_utility
            model.draft_kv_retain_ratio = candidate_retain_ratio
            statistics["tolerance_iter"] = 0
        else:
            model.set_skip_layers(origin_attn_skip_layer_id_set, origin_mlp_skip_layer_id_set)
            model.draft_kv_retain_ratio = statistics.get("draft_kv_retain_ratio", 1.0)
            statistics["tolerance_iter"] += 1
    elif score > statistics["origin_score"]:
        logging.info("=" * 30 + 'matchness changed from {:.4f} to {:.4f}'.format(
            statistics["origin_score"], score
        ) + "=" * 30)
        statistics["origin_score"] = score
        statistics["tolerance_iter"] = 0
        if score > statistics["max_score"]:
            statistics["optimization"] = False
            logging.info("=" * 30 + 'Optimization Stopped because the score reaches the expected number!' + "=" * 30)
    else:
        model.set_skip_layers(origin_attn_skip_layer_id_set, origin_mlp_skip_layer_id_set)
        statistics["tolerance_iter"] += 1

    statistics["opt_iter"] += 1

    if dynamic_retain_ratio:
        _maybe_finish_dynamic_retain(statistics)

    if (not dynamic_retain_ratio) and statistics["tolerance_iter"] > statistics["max_tolerance_iter"]:
        statistics["optimization"] = False
        logging.info("=" * 30 + 'Optimization Stopped because the optimization iter reaches the max tolerance!' + "=" * 30)
    if statistics["opt_iter"] > statistics["max_opt_iter"]:
        statistics["optimization"] = False
        logging.info("=" * 30 + 'Optimization Stopped because the optimization iter reaches the maximum!' + "=" * 30)


def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token_num,
        past_key_values_data_list,
        current_length_data,
        sample_p,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate.
    """
    prev_input_len = input_ids.shape[1]
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )

    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )

    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)

    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    prob = sample_p
    if logits_processor is not None:
        sample_token = torch.multinomial(prob, 1)
        sample_token = sample_token[None]
    else:
        sample_token = torch.argmax(prob)
        sample_token = sample_token[None, None]

    new_token_num += accept_length + 1

    return input_ids, new_token_num, sample_token
