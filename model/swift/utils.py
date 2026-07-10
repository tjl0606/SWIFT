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
VERIFY_SCOPE_BETA1 = 128
VERIFY_SCOPE_BETA2 = 256
VERIFY_SCOPE_RECENT_SIZE = VERIFY_SCOPE_BETA2

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
                     logits_processor=None, draft_token_num=None, statistics=None):
    """
    Initializes the swift structure for a given model.
    """
    with torch.inference_mode():
        # Normal verify path: must not reuse a stale tree mask
        _clear_swift_tree_state(model)

        collect_cosine = bool(statistics and statistics.get("cosine_prefill_skip_layers", False))
        if collect_cosine:
            _start_attn_cosine_collection(model)

        try:
            outputs, logits = swift_verify(model, input_ids, past_key_values=past_key_values)
        finally:
            if collect_cosine:
                cosine_scores = _finish_attn_cosine_collection(model)

        if collect_cosine:
            _apply_cosine_prefill_skip_layers(model, statistics, cosine_scores)

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
            statistics=statistics,
        )
    return swift_logits, sample_token, top1_prob


def swift_verify(
        model,
        input_ids=None,
        past_key_values=None,
        position_ids=None,
        attention_mask=None,
):
    """
    Verify the swift structure using the provided model and input.
    """
    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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


def _share_past_key_values(model, past_key_values_data, current_length_data, current_length=None):
    draft_current_length_data = current_length_data.clone()
    if current_length is not None:
        draft_current_length_data.fill_(int(current_length))
    draft_past_key_values = clone_past_key_values(
        model, past_key_values_data, draft_current_length_data
    )
    return past_key_values_data, draft_current_length_data, draft_past_key_values


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
        positive_score_fallback=False,
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
            # The adaptive controller chooses the KV budget; attention scores
            # only rank which middle tokens should fill that budget.
            candidate_scores = scores.to(device=device)[candidate_indices]
            if positive_score_fallback:
                positive_mask = candidate_scores > 0
                scored_indices = candidate_indices[positive_mask]
                scored_scores = candidate_scores[positive_mask]
            else:
                scored_indices = candidate_indices
                scored_scores = candidate_scores
            if scored_indices.numel() > 0:
                topk = min(remaining, scored_indices.numel())
                selected = scored_indices[torch.topk(scored_scores, k=topk).indices]
            else:
                selected = candidate_indices.new_empty(0)
        else:
            # No attention statistics are available on the first decode step,
            # so use a deterministic spread through the non-sink/non-recent span.
            selected = _fill_indices_evenly(candidate_indices, remaining)
        keep_mask[selected] = True

    remaining = keep_len - int(keep_mask.sum().item())
    if remaining > 0:
        candidate_indices = torch.nonzero(~keep_mask, as_tuple=True)[0]
        selected = _fill_indices_evenly(candidate_indices, remaining)
        keep_mask[selected] = True

    return torch.nonzero(keep_mask, as_tuple=True)[0]


def _select_scope_verify_kv_indices(
        full_input_ids,
        prefill_len,
        scores=None,
        beta1=VERIFY_SCOPE_BETA1,
        beta2=VERIFY_SCOPE_BETA2,
        min_retain_tokens=16,
        statistics=None,
):
    """
    SCOPE-style verifier selection: keep the prefill/prompt KV fixed, and only
    compress KV generated during decoding with a beta1 + beta2 budget.
    """
    full_len = int(full_input_ids.shape[1])
    device = full_input_ids.device
    prefill_len = max(0, min(int(prefill_len), full_len))
    beta1 = max(0, int(beta1))
    beta2 = max(0, int(beta2))
    decode_len = full_len - prefill_len
    if decode_len <= 0:
        return torch.arange(full_len, device=device, dtype=torch.long), full_len, full_len

    decode_budget = min(decode_len, beta1 + beta2)
    if decode_len <= decode_budget:
        # Below the SCOPE budget there is no approximate verifier cache: the
        # verifier sees the full decoding history and remains exact.
        if statistics is not None:
            _increment_stat(statistics, "verify_kv_scope_full_decode_uses")
            statistics["verify_kv_scope_beta1"] = int(beta1)
            statistics["verify_kv_scope_beta2"] = int(beta2)
        return torch.arange(full_len, device=device, dtype=torch.long), full_len, full_len

    decode_keep_len = decode_budget
    keep_len = prefill_len + decode_keep_len

    keep_mask = torch.zeros(full_len, device=device, dtype=torch.bool)
    if prefill_len > 0:
        # SCOPE does not rank or evict prompt/prefill KV.
        keep_mask[:prefill_len] = True

    recent_count = min(beta2, decode_keep_len, decode_len)
    if recent_count > 0:
        # beta2 is a hard local decoding window.
        keep_mask[full_len - recent_count:] = True

    remaining = keep_len - int(keep_mask.sum().item())
    decode_middle_end = full_len - recent_count
    if remaining > 0 and decode_middle_end > prefill_len:
        # beta1 is spent only on the decoding middle, ranked by full-context
        # verifier attention statistics when available.
        candidate_indices = torch.arange(
            prefill_len,
            decode_middle_end,
            device=device,
            dtype=torch.long,
        )
        selected = candidate_indices.new_empty(0)
        if scores is not None and scores.numel() == full_len:
            candidate_scores = scores.to(device=device, dtype=torch.float32)[candidate_indices]
            positive_mask = candidate_scores > 0
            scored_indices = candidate_indices[positive_mask]
            scored_scores = candidate_scores[positive_mask]
            if scored_indices.numel() > 0:
                topk = min(remaining, scored_indices.numel())
                selected = scored_indices[torch.topk(scored_scores, k=topk).indices]
        if selected.numel() > 0:
            keep_mask[selected] = True

    remaining = keep_len - int(keep_mask.sum().item())
    if remaining > 0:
        # First steps may not have verifier attention scores yet; even
        # sampling keeps coverage deterministic without task-specific rules.
        candidate_indices = torch.nonzero(~keep_mask, as_tuple=True)[0]
        candidate_indices = candidate_indices[candidate_indices >= prefill_len]
        selected = _fill_indices_evenly(candidate_indices, remaining)
        keep_mask[selected] = True

    if statistics is not None:
        _increment_stat(statistics, "verify_kv_scope_selection_count")
        statistics["verify_kv_scope_beta1"] = int(beta1)
        statistics["verify_kv_scope_beta2"] = int(beta2)
        statistics["verify_kv_scope_recent_size"] = int(beta2)
        statistics["verify_kv_scope_prefill_len"] = int(prefill_len)
        statistics["verify_kv_scope_prefill_kept_sum"] = (
            int(statistics.get("verify_kv_scope_prefill_kept_sum", 0)) + int(prefill_len)
        )
        statistics["verify_kv_scope_decode_len_sum"] = (
            int(statistics.get("verify_kv_scope_decode_len_sum", 0)) + int(decode_len)
        )
        statistics["verify_kv_scope_decode_kept_sum"] = (
            int(statistics.get("verify_kv_scope_decode_kept_sum", 0)) + int(decode_keep_len)
        )

    keep_indices = torch.nonzero(keep_mask, as_tuple=True)[0]
    return keep_indices, full_len, int(keep_indices.numel())


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


def _build_masked_draft_attention_mask(model, past_key_values, input_ids):
    keep_indices = getattr(model, "draft_kv_mask_keep_indices", None)
    full_len = int(getattr(model, "draft_kv_mask_full_len", 0) or 0)
    if keep_indices is None or full_len <= 0 or input_ids is None:
        return None

    past_len = 0
    if past_key_values is not None:
        for past_key_value in past_key_values:
            if past_key_value is not None:
                past_len = int(past_key_value[0].shape[2])
                break

    total_len = past_len + int(input_ids.shape[1])
    if total_len <= 0:
        return None

    attention_mask = torch.zeros(
        (input_ids.shape[0], total_len),
        dtype=torch.bool,
        device=input_ids.device,
    )
    local_keep_indices = keep_indices.to(device=input_ids.device)
    valid = (local_keep_indices >= 0) & (local_keep_indices < min(full_len, total_len))
    if valid.any().item():
        attention_mask[:, local_keep_indices[valid]] = True

    # Draft tokens appended after the original context must remain mutually visible
    # through the normal causal mask; only old, unselected context tokens are hidden.
    if total_len > full_len:
        attention_mask[:, full_len:] = True

    return attention_mask


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
        statistics=None,
):
    """
    Draft new tokens using the swift structure.
    """
    # Normal draft path must not reuse a stale tree mask from the previous tree decoding round
    _clear_swift_tree_state(model)

    use_draft_kv_mask = False
    if hasattr(model, "draft_kv_compress") and model.draft_kv_compress and model.draft_kv_retain_ratio < 0.9999:
        if _draft_kv_cache_mode(statistics) == "mask":
            draft_past_key_values_data, draft_current_length_data, draft_past_key_values = prepare_masked_draft_cache(
                model=model,
                full_input_ids=full_input_ids,
                past_key_values_data=past_key_values_data,
                current_length_data=current_length_data,
                retain_ratio=model.draft_kv_retain_ratio,
                min_retain_tokens=16,
                statistics=statistics,
            )
            use_draft_kv_mask = getattr(model, "draft_kv_mask_keep_indices", None) is not None
        else:
            draft_past_key_values_data, draft_current_length_data, draft_past_key_values = rebuild_compressed_draft_cache(
                model=model,
                full_input_ids=full_input_ids,
                past_key_values_data=past_key_values_data,
                current_length_data=current_length_data,
                retain_ratio=model.draft_kv_retain_ratio,
                min_retain_tokens=16,
                statistics=statistics,
                # sink_tokens=4,
            )
    else:
        _clear_draft_reuse_mapping(model)
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

    collect_draft_scores = (
        (
            _draft_kv_score_source(statistics) == "reuse"
            or (
                _verify_kv_enabled(statistics)
                and _verify_kv_score_source(statistics) == "reuse"
            )
        )
        and getattr(model, "draft_kv_last_keep_indices", None) is not None
    )
    if collect_draft_scores:
        _start_draft_attention_collection(model)

    try:
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
                    draft_attention_mask = (
                        _build_masked_draft_attention_mask(model, draft_past_key_values, input_ids)
                        if use_draft_kv_mask
                        else None
                    )
                    draft_outputs = model.model(
                        input_ids=input_ids,
                        attention_mask=draft_attention_mask,
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
    finally:
        if collect_draft_scores:
            _finish_draft_attention_collection(model, statistics=statistics)

    return (torch.cat(ss_token), torch.cat(ss_prob), ss_op), top1_prob


def reset_swift_mode(model):
    """
    Resets the swift settings to their initial state.
    """
    _clear_swift_tree_state(model)


def _start_attn_cosine_collection(model):
    for layer in getattr(model.model, "layers", []):
        layer.collect_attn_cosine = True
        layer.last_attn_cosine = None


def _finish_attn_cosine_collection(model):
    scores = []
    for layer in getattr(model.model, "layers", []):
        layer.collect_attn_cosine = False
        score = getattr(layer, "last_attn_cosine", None)
        if score is None:
            scores.append(float("nan"))
        else:
            scores.append(float(score.float().cpu().item()))
        layer.last_attn_cosine = None
    return scores


def _draft_kv_score_source(statistics):
    if statistics is None:
        return "heuristic"
    return statistics.get("draft_kv_score_source", "heuristic")


def _draft_kv_cache_mode(statistics):
    if statistics is None:
        return "copy"
    return statistics.get("draft_kv_cache_mode", "copy")


def _verify_kv_enabled(statistics):
    return bool(statistics and statistics.get("verify_kv_compress", False))


def _verify_kv_cache_mode(statistics):
    if statistics is None:
        return "copy"
    return statistics.get("verify_kv_cache_mode", "copy")


def _verify_kv_score_source(statistics):
    if statistics is None:
        return "semantic"
    return statistics.get("verify_kv_score_source", "semantic")


def _verify_kv_retain_ratio(statistics):
    if statistics is None:
        return 1.0
    ratio = statistics.get("verify_kv_retain_ratio")
    if ratio is None:
        ratio = statistics.get("draft_kv_retain_ratio", 1.0)
    return float(ratio)


def _verify_kv_bootstrap_full_steps(statistics):
    if statistics is None:
        return 0
    return max(0, int(statistics.get("verify_kv_bootstrap_full_steps", 0) or 0))


def _verify_kv_prefill_len(statistics, full_len):
    if statistics is None:
        return int(full_len)
    prefill_len = int(statistics.get("verify_kv_prefill_len", full_len) or 0)
    return max(0, min(prefill_len, int(full_len)))


def _verify_kv_scope_beta1(statistics):
    if statistics is None:
        return VERIFY_SCOPE_BETA1
    return max(0, int(statistics.get("verify_kv_scope_beta1", VERIFY_SCOPE_BETA1) or 0))


def _verify_kv_scope_beta2(statistics):
    if statistics is None:
        return VERIFY_SCOPE_BETA2
    legacy_recent_size = int(
        statistics.get("verify_kv_scope_recent_size", VERIFY_SCOPE_BETA2) or VERIFY_SCOPE_BETA2
    )
    return max(0, int(statistics.get("verify_kv_scope_beta2", legacy_recent_size) or 0))


def _verify_kv_scope_recent_size(statistics):
    return _verify_kv_scope_beta2(statistics)


def _verify_kv_scope_score_full_only(statistics):
    if statistics is None:
        return True
    return bool(statistics.get("verify_kv_scope_score_full_only", True))


def _verify_kv_dynamic_enabled(statistics):
    return bool(
        statistics
        and statistics.get("verify_kv_dynamic", False)
        and statistics.get("verify_kv_compress", False)
        and statistics.get("verify_kv_score_source") == "scope"
    )


def _verify_kv_scope_needs_compression(full_input_ids, statistics):
    if full_input_ids is None:
        return False
    full_len = int(full_input_ids.shape[1])
    prefill_len = _verify_kv_prefill_len(statistics, full_len)
    decode_len = max(0, full_len - prefill_len)
    beta1 = _verify_kv_scope_beta1(statistics)
    beta2 = _verify_kv_scope_beta2(statistics)
    if statistics is not None:
        statistics["verify_kv_scope_beta1"] = int(beta1)
        statistics["verify_kv_scope_beta2"] = int(beta2)
        statistics["verify_kv_scope_recent_size"] = int(beta2)
        statistics["verify_kv_scope_prefill_len"] = int(prefill_len)
    # Until generated decoding KV exceeds beta1+beta2, the SCOPE verifier path
    # should fall through to full-cache verification.
    return decode_len > (beta1 + beta2)


def verify_confidence_margin(logits, best_candidate, accept_length):
    if logits is None or logits.numel() == 0 or logits.shape[-1] < 2:
        return None, None

    candidate_index = int(best_candidate.item() if hasattr(best_candidate, "item") else best_candidate)
    candidate_index = max(0, min(candidate_index, int(logits.shape[0]) - 1))
    step_count = max(1, min(int(accept_length) + 1, int(logits.shape[1])))
    selected_logits = logits[candidate_index, :step_count].detach().float()
    top2 = torch.topk(selected_logits, k=2, dim=-1).values
    margins = top2[:, 0] - top2[:, 1]
    return float(margins.mean().cpu().item()), float(margins.min().cpu().item())


def update_verify_kv_dynamic_controller(
        statistics,
        accepted_total_tokens,
        accepted_draft_tokens,
        drafted_tokens,
        confidence_margin,
        confidence_min=None,
        allow_switch=True,
):
    if not _verify_kv_dynamic_enabled(statistics) or drafted_tokens <= 0:
        return

    accepted_total_tokens = int(accepted_total_tokens)
    accepted_draft_tokens = max(0, int(accepted_draft_tokens))
    drafted_tokens = max(1, int(drafted_tokens))
    confidence_margin = 0.0 if confidence_margin is None else float(confidence_margin)
    confidence_min = confidence_margin if confidence_min is None else float(confidence_min)
    # Dynamic verifier compression uses accepted draft tokens and verifier
    # confidence as quality signals; it does not inspect task labels.
    token_acceptance = float(accepted_draft_tokens) / float(drafted_tokens)

    acceptance_history = statistics.setdefault("verify_kv_dynamic_acceptance_history", [])
    mean_history = statistics.setdefault("verify_kv_dynamic_mean_history", [])
    margin_history = statistics.setdefault("verify_kv_dynamic_margin_history", [])
    min_margin_history = statistics.setdefault("verify_kv_dynamic_min_margin_history", [])
    acceptance_history.append(token_acceptance)
    mean_history.append(float(accepted_total_tokens))
    margin_history.append(confidence_margin)
    min_margin_history.append(confidence_min)

    statistics["verify_kv_dynamic_observations"] = len(acceptance_history)
    statistics["verify_kv_dynamic_acceptance_sum"] = (
        float(statistics.get("verify_kv_dynamic_acceptance_sum", 0.0)) + token_acceptance
    )
    statistics["verify_kv_dynamic_mean_sum"] = (
        float(statistics.get("verify_kv_dynamic_mean_sum", 0.0)) + float(accepted_total_tokens)
    )
    statistics["verify_kv_dynamic_confidence_margin_sum"] = (
        float(statistics.get("verify_kv_dynamic_confidence_margin_sum", 0.0)) + confidence_margin
    )
    statistics["verify_kv_dynamic_confidence_min_sum"] = (
        float(statistics.get("verify_kv_dynamic_confidence_min_sum", 0.0)) + confidence_min
    )

    window = max(1, int(statistics.get("verify_kv_dynamic_window", 32) or 32))
    min_observations = max(1, int(statistics.get("verify_kv_dynamic_min_observations", window) or window))
    if len(acceptance_history) < min_observations:
        return

    recent_acceptance = acceptance_history[-window:]
    recent_mean = mean_history[-window:]
    recent_margin = margin_history[-window:]
    recent_min_margin = min_margin_history[-window:]
    window_acceptance = sum(recent_acceptance) / float(len(recent_acceptance))
    window_mean = sum(recent_mean) / float(len(recent_mean))
    window_margin = sum(recent_margin) / float(len(recent_margin))
    window_min_margin = sum(recent_min_margin) / float(len(recent_min_margin))

    statistics["verify_kv_dynamic_last_window_acceptance"] = float(window_acceptance)
    statistics["verify_kv_dynamic_last_window_mean"] = float(window_mean)
    statistics["verify_kv_dynamic_last_window_margin"] = float(window_margin)
    statistics["verify_kv_dynamic_last_window_min_margin"] = float(window_min_margin)

    cooldown_remaining = int(statistics.get("verify_kv_dynamic_cooldown_remaining", 0) or 0)
    if cooldown_remaining > 0:
        statistics["verify_kv_dynamic_cooldown_remaining"] = cooldown_remaining - 1
        return
    if not allow_switch:
        return

    beta1 = _verify_kv_scope_beta1(statistics)
    min_beta1 = max(0, int(statistics.get("verify_kv_dynamic_min_beta1", 64) or 0))
    max_beta1 = int(statistics.get("verify_kv_dynamic_max_beta1", beta1) or beta1)
    max_beta1 = max(min_beta1, max_beta1)
    step = max(1, int(statistics.get("verify_kv_dynamic_step", 32) or 32))
    patience = max(1, int(statistics.get("verify_kv_dynamic_patience", 2) or 2))
    cooldown = max(0, int(statistics.get("verify_kv_dynamic_cooldown", 16) or 0))

    acceptance_floor = float(statistics.get("verify_kv_dynamic_acceptance_floor", 0.88))
    mean_floor = float(statistics.get("verify_kv_dynamic_mean_floor", 3.0))
    confidence_floor = float(statistics.get("verify_kv_dynamic_confidence_floor", 0.5))
    confidence_low = float(statistics.get("verify_kv_dynamic_confidence_low", 0.25))

    low_quality = (
        window_acceptance < acceptance_floor
        or window_mean < mean_floor
        or window_margin < confidence_low
    )
    high_quality = (
        window_acceptance >= acceptance_floor
        and window_mean >= mean_floor
        and window_margin >= confidence_floor
    )

    if low_quality:
        statistics["verify_kv_dynamic_bad_windows"] = (
            int(statistics.get("verify_kv_dynamic_bad_windows", 0)) + 1
        )
        statistics["verify_kv_dynamic_good_windows"] = 0
    elif high_quality:
        statistics["verify_kv_dynamic_good_windows"] = (
            int(statistics.get("verify_kv_dynamic_good_windows", 0)) + 1
        )
        statistics["verify_kv_dynamic_bad_windows"] = 0
    else:
        statistics["verify_kv_dynamic_good_windows"] = 0
        statistics["verify_kv_dynamic_bad_windows"] = 0

    action = "keep"
    new_beta1 = beta1
    if low_quality and int(statistics.get("verify_kv_dynamic_bad_windows", 0)) >= patience:
        # Poor windows make verification less approximate by increasing beta1.
        new_beta1 = min(max_beta1, beta1 + step)
        action = "less_compress" if new_beta1 != beta1 else "keep"
        statistics["verify_kv_dynamic_bad_windows"] = 0
    elif high_quality and int(statistics.get("verify_kv_dynamic_good_windows", 0)) >= patience:
        # Stable high-quality windows trade verifier context for speed.
        new_beta1 = max(min_beta1, beta1 - step)
        action = "more_compress" if new_beta1 != beta1 else "keep"
        statistics["verify_kv_dynamic_good_windows"] = 0

    action_counts = statistics.setdefault("verify_kv_dynamic_action_counts", {})
    action_counts[action] = int(action_counts.get(action, 0)) + 1
    statistics["verify_kv_dynamic_decision_count"] = int(
        statistics.get("verify_kv_dynamic_decision_count", 0)
    ) + 1
    statistics["verify_kv_dynamic_last_action"] = action

    if new_beta1 == beta1:
        statistics["verify_kv_dynamic_keep_decisions"] = int(
            statistics.get("verify_kv_dynamic_keep_decisions", 0)
        ) + 1
        return

    statistics["verify_kv_scope_beta1"] = int(new_beta1)
    statistics["verify_kv_dynamic_current_beta1"] = int(new_beta1)
    statistics["verify_kv_dynamic_cooldown_remaining"] = cooldown
    statistics["verify_kv_dynamic_switches"] = int(
        statistics.get("verify_kv_dynamic_switches", 0)
    ) + 1
    if new_beta1 < beta1:
        statistics["verify_kv_dynamic_more_compress_switches"] = int(
            statistics.get("verify_kv_dynamic_more_compress_switches", 0)
        ) + 1
    else:
        statistics["verify_kv_dynamic_less_compress_switches"] = int(
            statistics.get("verify_kv_dynamic_less_compress_switches", 0)
        ) + 1
    logging.info(
        "Verify KV dynamic {}: beta1 {} -> {} "
        "(acceptance {:.4f}, mean {:.4f}, margin {:.4f})".format(
            action,
            beta1,
            new_beta1,
            window_acceptance,
            window_mean,
            window_margin,
        )
    )


def _increment_stat(statistics, key):
    if statistics is not None:
        statistics[key] = int(statistics.get(key, 0)) + 1


def _clear_draft_reuse_mapping(model):
    model.draft_kv_last_keep_indices = None
    model.draft_kv_last_full_len = 0
    model.draft_kv_last_cache_mode = None
    model.draft_kv_mask_keep_indices = None
    model.draft_kv_mask_full_len = 0


def _clear_verify_kv_mapping(model):
    model.verify_kv_last_keep_indices = None
    model.verify_kv_last_full_len = 0
    model.verify_kv_last_cache_mode = None
    model.verify_kv_mask_keep_indices = None
    model.verify_kv_mask_full_len = 0


def reset_draft_attention_reuse(model):
    model.draft_kv_reuse_scores = None
    model.draft_kv_reuse_scores_len = 0
    model.verify_kv_semantic_scores = None
    model.verify_kv_semantic_scores_len = 0
    model.draft_kv_pending_attention_scores = None
    model.draft_kv_pending_attention_scores_len = 0
    _clear_draft_reuse_mapping(model)
    _clear_verify_kv_mapping(model)
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.collect_draft_attn_scores = False
        attn.draft_attn_score_sum = None
        attn.draft_attn_score_count = 0
        attn.collect_verify_attn_scores = False
        attn.verify_attn_score_sum = None
        attn.verify_attn_score_count = 0


def _set_draft_reuse_mapping(model, keep_indices, full_len, cache_mode="copy"):
    model.draft_kv_last_keep_indices = keep_indices.detach()
    model.draft_kv_last_full_len = int(full_len)
    model.draft_kv_last_cache_mode = cache_mode


def _set_verify_kv_mapping(model, keep_indices, full_len, cache_mode="copy"):
    model.verify_kv_last_keep_indices = keep_indices.detach()
    model.verify_kv_last_full_len = int(full_len)
    model.verify_kv_last_cache_mode = cache_mode


def _get_reused_draft_attention_scores(model, full_len, device, statistics=None, stat_prefix="draft_kv"):
    scores = getattr(model, "draft_kv_reuse_scores", None)
    miss_key = f"{stat_prefix}_reuse_score_misses"
    hit_key = f"{stat_prefix}_reuse_score_hits"
    if scores is None or scores.numel() == 0:
        _increment_stat(statistics, miss_key)
        return None

    reused_scores = torch.zeros(full_len, dtype=torch.float32, device=device)
    copy_len = min(int(full_len), int(scores.numel()))
    if copy_len <= 0:
        _increment_stat(statistics, miss_key)
        return None

    reused_scores[:copy_len].copy_(scores.to(device=device, dtype=torch.float32)[:copy_len])
    _increment_stat(statistics, hit_key)
    return reused_scores


def _get_reused_semantic_attention_scores(model, full_len, device, statistics=None, stat_prefix="verify_kv"):
    scores = getattr(model, "verify_kv_semantic_scores", None)
    miss_key = f"{stat_prefix}_semantic_score_misses"
    hit_key = f"{stat_prefix}_semantic_score_hits"
    if scores is None or scores.numel() == 0:
        _increment_stat(statistics, miss_key)
        return None

    reused_scores = torch.zeros(full_len, dtype=torch.float32, device=device)
    copy_len = min(int(full_len), int(scores.numel()))
    if copy_len <= 0:
        _increment_stat(statistics, miss_key)
        return None

    reused_scores[:copy_len].copy_(scores.to(device=device, dtype=torch.float32)[:copy_len])
    _increment_stat(statistics, hit_key)
    return reused_scores


def _update_verify_semantic_attention_scores(model, attention_scores, statistics=None):
    if attention_scores is None or attention_scores.numel() == 0:
        return False

    current_scores = attention_scores.detach().float()
    full_len = int(current_scores.numel())
    if full_len <= 0:
        return False

    ema = 0.7
    if statistics is not None:
        ema = float(statistics.get("draft_kv_reuse_ema", ema))
    ema = min(1.0, max(0.0, ema))

    previous_scores = getattr(model, "verify_kv_semantic_scores", None)
    if previous_scores is not None and previous_scores.numel() > 0:
        merged_scores = current_scores.clone()
        copy_len = min(int(previous_scores.numel()), full_len)
        previous_slice = previous_scores.to(
            device=current_scores.device,
            dtype=torch.float32,
        )[:copy_len]
        merged_scores[:copy_len] = previous_slice * ema + current_scores[:copy_len] * (1.0 - ema)
    else:
        merged_scores = current_scores

    model.verify_kv_semantic_scores = merged_scores.detach()
    model.verify_kv_semantic_scores_len = full_len
    _increment_stat(statistics, "verify_kv_semantic_score_updates")
    return True


def _start_draft_attention_collection(model):
    model.draft_kv_pending_attention_scores = None
    model.draft_kv_pending_attention_scores_len = 0
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.collect_draft_attn_scores = True
        attn.draft_attn_score_sum = None
        attn.draft_attn_score_count = 0


def _finish_draft_attention_collection(model, statistics=None):
    keep_indices = getattr(model, "draft_kv_last_keep_indices", None)
    full_len = int(getattr(model, "draft_kv_last_full_len", 0) or 0)
    cache_mode = getattr(model, "draft_kv_last_cache_mode", "copy")
    if keep_indices is None or full_len <= 0:
        return

    compressed_len = int(keep_indices.numel())
    layer_scores = []
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.collect_draft_attn_scores = False
        score_sum = getattr(attn, "draft_attn_score_sum", None)
        score_count = int(getattr(attn, "draft_attn_score_count", 0) or 0)
        if score_sum is not None and score_count > 0:
            if cache_mode == "mask":
                usable_len = min(full_len, int(score_sum.numel()))
            else:
                usable_len = min(compressed_len, int(score_sum.numel()))
            if usable_len > 0:
                layer_scores.append(score_sum[:usable_len].detach().float() / float(score_count))
        attn.draft_attn_score_sum = None
        attn.draft_attn_score_count = 0

    if not layer_scores:
        model.draft_kv_pending_attention_scores = None
        model.draft_kv_pending_attention_scores_len = 0
        return

    usable_len = min(score.numel() for score in layer_scores)
    device = layer_scores[0].device
    collected_scores = torch.stack([score[:usable_len].to(device) for score in layer_scores]).mean(dim=0)
    full_scores = torch.zeros(full_len, dtype=torch.float32, device=device)
    if cache_mode == "mask":
        full_scores[:usable_len] = collected_scores[:usable_len]
    else:
        local_keep_indices = keep_indices.to(device=device)[:usable_len]
        valid = local_keep_indices < full_len
        full_scores[local_keep_indices[valid]] = collected_scores[valid]

    model.draft_kv_pending_attention_scores = full_scores.detach()
    model.draft_kv_pending_attention_scores_len = full_len


def _start_verify_attention_collection(model):
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.collect_verify_attn_scores = True
        attn.verify_attn_score_sum = None
        attn.verify_attn_score_count = 0


def _finish_verify_attention_collection(model, full_len, statistics=None):
    keep_indices = getattr(model, "verify_kv_last_keep_indices", None)
    cache_mode = getattr(model, "verify_kv_last_cache_mode", "copy")
    full_len = int(full_len)
    if full_len <= 0:
        return

    compressed_len = int(keep_indices.numel()) if keep_indices is not None else full_len
    layer_scores = []
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.collect_verify_attn_scores = False
        score_sum = getattr(attn, "verify_attn_score_sum", None)
        score_count = int(getattr(attn, "verify_attn_score_count", 0) or 0)
        if score_sum is not None and score_count > 0:
            if cache_mode == "mask" or keep_indices is None:
                usable_len = min(full_len, int(score_sum.numel()))
            else:
                usable_len = min(compressed_len, int(score_sum.numel()))
            if usable_len > 0:
                layer_scores.append(score_sum[:usable_len].detach().float() / float(score_count))
        attn.verify_attn_score_sum = None
        attn.verify_attn_score_count = 0

    if not layer_scores:
        return

    usable_len = min(score.numel() for score in layer_scores)
    device = layer_scores[0].device
    collected_scores = torch.stack([score[:usable_len].to(device) for score in layer_scores]).mean(dim=0)
    full_scores = torch.zeros(full_len, dtype=torch.float32, device=device)
    if cache_mode == "mask" or keep_indices is None:
        full_scores[:usable_len] = collected_scores[:usable_len]
    else:
        local_keep_indices = keep_indices.to(device=device)[:usable_len]
        valid = local_keep_indices < full_len
        full_scores[local_keep_indices[valid]] = collected_scores[valid]

    _update_verify_semantic_attention_scores(model, full_scores, statistics=statistics)


def apply_draft_attention_reuse_reward(
        model,
        accepted_draft_tokens,
        drafted_tokens,
        statistics=None,
):
    attention_scores = getattr(model, "draft_kv_pending_attention_scores", None)
    full_len = int(getattr(model, "draft_kv_pending_attention_scores_len", 0) or 0)
    if attention_scores is None or attention_scores.numel() == 0 or full_len <= 0:
        _increment_stat(statistics, "draft_kv_reuse_score_empty_updates")
        return False

    drafted_tokens = int(drafted_tokens)
    if drafted_tokens <= 0:
        model.draft_kv_pending_attention_scores = None
        model.draft_kv_pending_attention_scores_len = 0
        _increment_stat(statistics, "draft_kv_reuse_score_empty_updates")
        return False

    accepted_draft_tokens = max(0, int(accepted_draft_tokens))
    reward = float(accepted_draft_tokens) / float(drafted_tokens)

    device = attention_scores.device
    attention_scores = attention_scores.to(device=device, dtype=torch.float32)
    full_len = min(full_len, int(attention_scores.numel()))
    attention_scores = attention_scores[:full_len]
    # Accepted draft tokens are used as a light reward signal: attention
    # patterns from more successful drafts get ranked higher next step.
    weighted_score = attention_scores * (0.5 + reward)

    previous_full = torch.zeros_like(weighted_score)
    previous_scores = getattr(model, "draft_kv_reuse_scores", None)
    if previous_scores is not None and previous_scores.numel() > 0:
        copy_len = min(int(previous_scores.numel()), full_len)
        previous_full[:copy_len].copy_(previous_scores.to(device=device, dtype=torch.float32)[:copy_len])

    full_scores = previous_full * 0.5 + weighted_score * 0.5
    model.draft_kv_reuse_scores = full_scores.detach()
    model.draft_kv_reuse_scores_len = full_len
    model.draft_kv_pending_attention_scores = None
    model.draft_kv_pending_attention_scores_len = 0

    if statistics is not None:
        statistics["draft_kv_reuse_last_reward"] = reward
    _increment_stat(statistics, "draft_kv_reuse_score_updates")
    return True


def _cosine_eligible_attn_layers(num_hidden_layers, statistics):
    keep_first = max(0, int(statistics.get("cosine_keep_first_layers", 1)))
    keep_last = max(0, int(statistics.get("cosine_keep_last_layers", 2)))
    last_exclusive = max(keep_first, num_hidden_layers - keep_last)
    return list(range(keep_first, last_exclusive))


def _safe_cosine_score(scores, layer_id):
    if layer_id < 0 or layer_id >= len(scores):
        return -1.0
    score = scores[layer_id]
    if isinstance(score, float) and math.isnan(score):
        return -1.0
    return float(score)


def _rank_layers_by_cosine(scores, eligible_layers):
    return sorted(
        [int(layer_id) for layer_id in eligible_layers],
        key=lambda layer_id: (-_safe_cosine_score(scores, layer_id), int(layer_id)),
    )


def _build_cosine_attn_skip_layers(scores, statistics):
    eligible_layers = _cosine_eligible_attn_layers(len(scores), statistics)
    ranked_layers = _rank_layers_by_cosine(scores, eligible_layers)
    mode = statistics.get("cosine_skip_mode", "topk")
    max_skip_layers = statistics.get("cosine_max_skip_layers")

    if max_skip_layers is None:
        max_skip_layers = int(len(eligible_layers) * float(statistics.get("skip_ratio", 0.0)))
    max_skip_layers = max(0, min(int(max_skip_layers), len(eligible_layers)))

    if mode == "threshold":
        alpha = float(statistics.get("cosine_attn_alpha", 0.985))
        selected = [
            layer_id
            for layer_id in ranked_layers
            if _safe_cosine_score(scores, layer_id) >= alpha
        ]
        if max_skip_layers > 0:
            selected = selected[:max_skip_layers]
    else:
        selected = ranked_layers[:max_skip_layers]

    return selected, ranked_layers, eligible_layers


def _build_cosine_mlp_skip_layers(num_hidden_layers, statistics):
    interval = int(statistics.get("cosine_mlp_interval", 0))
    if interval <= 0:
        return []

    keep_first = max(0, int(statistics.get("cosine_keep_first_layers", 1)))
    keep_last = max(0, int(statistics.get("cosine_keep_last_layers", 2)))
    last_exclusive = max(keep_first, num_hidden_layers - keep_last)
    return [
        layer_id
        for layer_id in range(keep_first, last_exclusive)
        if (layer_id + 1) % interval == 0
    ]


def _apply_cosine_prefill_skip_layers(model, statistics, cosine_scores):
    attn_skip_layers, ranked_layers, eligible_layers = _build_cosine_attn_skip_layers(
        cosine_scores,
        statistics,
    )
    mlp_skip_layers = _build_cosine_mlp_skip_layers(len(cosine_scores), statistics)

    model.set_skip_layers(attn_skip_layers, mlp_skip_layers)

    statistics["cosine_attn_scores"] = [float(score) for score in cosine_scores]
    statistics["cosine_attn_ranking"] = [int(layer_id) for layer_id in ranked_layers]
    statistics["cosine_attn_eligible_layers"] = [int(layer_id) for layer_id in eligible_layers]
    statistics["cosine_attn_skip_layers"] = [int(layer_id) for layer_id in attn_skip_layers]
    statistics["cosine_mlp_skip_layers"] = [int(layer_id) for layer_id in mlp_skip_layers]
    statistics["cosine_current_attn_skip_count"] = len(attn_skip_layers)
    statistics["cosine_prefill_count"] = int(statistics.get("cosine_prefill_count", 0)) + 1
    statistics["cosine_attn_skip_count_sum"] = (
        int(statistics.get("cosine_attn_skip_count_sum", 0)) + len(attn_skip_layers)
    )
    statistics["cosine_mlp_skip_count_sum"] = (
        int(statistics.get("cosine_mlp_skip_count_sum", 0)) + len(mlp_skip_layers)
    )

    logging.info(
        "Cosine prefill skip layers: attn=%s mlp=%s",
        statistics["cosine_attn_skip_layers"],
        statistics["cosine_mlp_skip_layers"],
    )


def rebuild_compressed_draft_cache(
        model,
        full_input_ids,
        past_key_values_data,
        current_length_data,
        retain_ratio=1.0,
        min_retain_tokens=16,
        statistics=None,
):
    """
    Rebuild a draft cache by retaining sink, recent, and attention-heavy tokens.
    """
    keep_indices, full_len, keep_len = _select_draft_kv_keep_indices(
        model=model,
        full_input_ids=full_input_ids,
        past_key_values_data=past_key_values_data,
        current_length_data=current_length_data,
        retain_ratio=retain_ratio,
        min_retain_tokens=min_retain_tokens,
        statistics=statistics,
    )
    if keep_indices is None or keep_len == full_len:
        _clear_draft_reuse_mapping(model)
        return _copy_past_key_values(model, past_key_values_data, current_length_data)

    _set_draft_reuse_mapping(model, keep_indices, full_len, cache_mode="copy")
    _increment_stat(statistics, "draft_kv_copy_cache_rebuilds")
    return _copy_selected_kv_cache(
        model=model,
        past_key_values_data=past_key_values_data,
        current_length_data=current_length_data,
        keep_indices=keep_indices,
    )


def _select_draft_kv_keep_indices(
        model,
        full_input_ids,
        past_key_values_data,
        current_length_data,
        retain_ratio=1.0,
        min_retain_tokens=16,
        statistics=None,
):
    if full_input_ids is None:
        return None, 0, 0

    full_len = full_input_ids.shape[1]
    keep_len = max(min_retain_tokens, int(full_len * retain_ratio))
    keep_len = min(keep_len, full_len)

    if keep_len == full_len:
        return torch.arange(full_len, device=full_input_ids.device, dtype=torch.long), full_len, keep_len

    _clear_swift_tree_state(model)
    score_source = _draft_kv_score_source(statistics)
    if score_source == "observation":
        # Accurate but expensive: runs an extra observation forward pass before
        # drafting, which is why this path is not the current default.
        scores = _get_observation_attention_scores(
            model=model,
            full_input_ids=full_input_ids,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
        )
    elif score_source == "reuse":
        # Current default: reuse attention statistics collected from the
        # previous draft instead of paying for another observation pass.
        scores = _get_reused_draft_attention_scores(
            model=model,
            full_len=full_len,
            device=full_input_ids.device,
            statistics=statistics,
        )
    else:
        scores = None
    keep_indices = _select_smart_kv_indices(
        full_input_ids=full_input_ids,
        keep_len=keep_len,
        scores=scores,
    )
    return keep_indices, full_len, keep_len


def _select_verify_kv_keep_indices(
        model,
        full_input_ids,
        past_key_values_data,
        current_length_data,
        retain_ratio=1.0,
        min_retain_tokens=16,
        statistics=None,
):
    if full_input_ids is None:
        return None, 0, 0

    full_len = full_input_ids.shape[1]
    keep_len = max(min_retain_tokens, int(full_len * retain_ratio))
    keep_len = min(keep_len, full_len)

    if keep_len == full_len:
        return torch.arange(full_len, device=full_input_ids.device, dtype=torch.long), full_len, keep_len

    _clear_swift_tree_state(model)
    score_source = _verify_kv_score_source(statistics)
    if score_source == "observation":
        scores = _get_observation_attention_scores(
            model=model,
            full_input_ids=full_input_ids,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
        )
    elif score_source == "semantic":
        scores = _get_reused_semantic_attention_scores(
            model=model,
            full_len=full_len,
            device=full_input_ids.device,
            statistics=statistics,
            stat_prefix="verify_kv",
        )
    elif score_source == "scope":
        # SCOPE uses verifier semantic scores, but score collection is limited
        # to full-context verifier passes by default to avoid a masked-context
        # feedback loop.
        scores = _get_reused_semantic_attention_scores(
            model=model,
            full_len=full_len,
            device=full_input_ids.device,
            statistics=statistics,
            stat_prefix="verify_kv",
        )
        return _select_scope_verify_kv_indices(
            full_input_ids=full_input_ids,
            prefill_len=_verify_kv_prefill_len(statistics, full_len),
            scores=scores,
            beta1=_verify_kv_scope_beta1(statistics),
            beta2=_verify_kv_scope_beta2(statistics),
            min_retain_tokens=min_retain_tokens,
            statistics=statistics,
        )
    elif score_source == "reuse":
        scores = _get_reused_draft_attention_scores(
            model=model,
            full_len=full_len,
            device=full_input_ids.device,
            statistics=statistics,
            stat_prefix="verify_kv",
        )
    else:
        scores = None
    keep_indices = _select_smart_kv_indices(
        full_input_ids=full_input_ids,
        keep_len=keep_len,
        scores=scores,
        positive_score_fallback=(score_source == "semantic"),
    )
    return keep_indices, full_len, keep_len


def prepare_masked_draft_cache(
        model,
        full_input_ids,
        past_key_values_data,
        current_length_data,
        retain_ratio=1.0,
        min_retain_tokens=16,
        statistics=None,
):
    """
    Prepare a draft cache that keeps the original KV layout and masks out
    unselected context tokens instead of physically copying selected KV.
    """
    keep_indices, full_len, keep_len = _select_draft_kv_keep_indices(
        model=model,
        full_input_ids=full_input_ids,
        past_key_values_data=past_key_values_data,
        current_length_data=current_length_data,
        retain_ratio=retain_ratio,
        min_retain_tokens=min_retain_tokens,
        statistics=statistics,
    )
    if keep_indices is None or keep_len == full_len:
        _clear_draft_reuse_mapping(model)
        return _share_past_key_values(model, past_key_values_data, current_length_data)

    _set_draft_reuse_mapping(model, keep_indices, full_len, cache_mode="mask")
    # Mask mode avoids physically copying selected KV into a compact cache.
    # The original KV layout remains full length and unselected old tokens are
    # hidden by an attention mask during draft decoding.
    model.draft_kv_mask_keep_indices = keep_indices.detach()
    model.draft_kv_mask_full_len = int(full_len)
    _increment_stat(statistics, "draft_kv_mask_cache_rebuilds")
    return _share_past_key_values(
        model,
        past_key_values_data,
        current_length_data,
        current_length=full_len,
    )


def prepare_approx_verify_cache(
        model,
        full_input_ids,
        past_key_values_data,
        current_length_data,
        retain_ratio=1.0,
        min_retain_tokens=16,
        statistics=None,
):
    """
    Prepare the approximate verifier KV cache.

    Copy mode uses a compact verifier cache and later copies accepted-token KV
    back into the full cache.  Mask mode keeps the full cache layout and hides
    unselected old tokens during tree verification.
    """
    keep_indices, full_len, keep_len = _select_verify_kv_keep_indices(
        model=model,
        full_input_ids=full_input_ids,
        past_key_values_data=past_key_values_data,
        current_length_data=current_length_data,
        retain_ratio=retain_ratio,
        min_retain_tokens=min_retain_tokens,
        statistics=statistics,
    )
    if keep_indices is None or keep_len == full_len:
        _clear_verify_kv_mapping(model)
        return _share_past_key_values(model, past_key_values_data, current_length_data), None, False

    cache_mode = _verify_kv_cache_mode(statistics)
    _set_verify_kv_mapping(model, keep_indices, full_len, cache_mode=cache_mode)
    if cache_mode == "mask":
        model.verify_kv_mask_keep_indices = keep_indices.detach()
        model.verify_kv_mask_full_len = int(full_len)
        _increment_stat(statistics, "verify_kv_mask_cache_rebuilds")
        return (
            _share_past_key_values(
                model,
                past_key_values_data,
                current_length_data,
                current_length=full_len,
            ),
            None,
            True,
        )

    _clear_verify_kv_mapping(model)
    _set_verify_kv_mapping(model, keep_indices, full_len, cache_mode="copy")
    _increment_stat(statistics, "verify_kv_copy_cache_rebuilds")
    return (
        _copy_selected_kv_cache(
            model=model,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            keep_indices=keep_indices,
        ),
        keep_len,
        False,
    )


def _build_masked_verify_attention_mask(model, past_key_values, input_ids):
    keep_indices = getattr(model, "verify_kv_mask_keep_indices", None)
    full_len = int(getattr(model, "verify_kv_mask_full_len", 0) or 0)
    if keep_indices is None or full_len <= 0 or input_ids is None:
        return None

    past_len = 0
    if past_key_values is not None:
        for past_key_value in past_key_values:
            if past_key_value is not None:
                past_len = int(past_key_value[0].shape[2])
                break

    total_len = past_len + int(input_ids.shape[1])
    if total_len <= 0:
        return None

    attention_mask = torch.zeros(
        (input_ids.shape[0], total_len),
        dtype=torch.bool,
        device=input_ids.device,
    )
    local_keep_indices = keep_indices.to(device=input_ids.device)
    valid = (local_keep_indices >= 0) & (local_keep_indices < min(full_len, total_len))
    if valid.any().item():
        attention_mask[:, local_keep_indices[valid]] = True

    if total_len > full_len:
        attention_mask[:, full_len:] = True

    return attention_mask


def reset_past_key_values(past_key_values):
    """
    Resets the current lengths in the past key-values to zero.
    """
    for i in range(len(past_key_values)):
        for j in range(2):
            past_key_values[i][j].current_length.fill_(0)
    return past_key_values


def recompute_accepted_kv_full_context(
        model,
        past_key_values_data,
        current_length_data,
        input_ids,
        accepted_tokens,
):
    if accepted_tokens is None or accepted_tokens.numel() == 0:
        return False

    prev_input_len = int(input_ids.shape[1])
    _clear_swift_tree_state(model)
    full_current_length_data = current_length_data.clone()
    full_current_length_data.fill_(prev_input_len)
    full_past_key_values = clone_past_key_values(
        model,
        past_key_values_data,
        full_current_length_data,
    )
    position_ids = torch.arange(
        prev_input_len,
        prev_input_len + int(accepted_tokens.shape[1]),
        dtype=torch.long,
        device=accepted_tokens.device,
    ).unsqueeze(0)
    with torch.inference_mode():
        model.model(
            input_ids=accepted_tokens,
            attention_mask=None,
            past_key_values=full_past_key_values,
            position_ids=position_ids,
        )
    return True


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
        past_key_values_data=None,
        current_length_data=None,
        statistics=None,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    """
    position_ids = swift_position_ids + input_ids.shape[1]
    verify_kv_source_data = None
    verify_kv_source_base_offset = None
    verify_attention_mask = None
    verify_past_key_values = past_key_values
    use_verify_kv_mask = False
    used_approx_verify_kv = False
    verify_sample_step = int(statistics.get("verify_kv_sample_step", 0) or 0) if statistics is not None else 0
    use_bootstrap_full_kv = (
        _verify_kv_enabled(statistics)
        and verify_sample_step < _verify_kv_bootstrap_full_steps(statistics)
    )
    verify_score_source = _verify_kv_score_source(statistics)
    scope_needs_compression = (
        verify_score_source == "scope"
        and _verify_kv_scope_needs_compression(input_ids, statistics)
    )

    if (
            _verify_kv_enabled(statistics)
            and past_key_values_data is not None
            and current_length_data is not None
            and (
                scope_needs_compression
                or (
                    verify_score_source != "scope"
                    and _verify_kv_retain_ratio(statistics) < 0.9999
                )
            )
            and not use_bootstrap_full_kv
    ):
        (
            verify_past_key_values_data,
            _verify_current_length_data,
            verify_past_key_values,
        ), verify_kv_source_base_offset, use_verify_kv_mask = prepare_approx_verify_cache(
            model=model,
            full_input_ids=input_ids,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            retain_ratio=_verify_kv_retain_ratio(statistics),
            min_retain_tokens=16,
            statistics=statistics,
        )
        if verify_kv_source_base_offset is not None:
            verify_kv_source_data = verify_past_key_values_data
            used_approx_verify_kv = True
        if use_verify_kv_mask:
            used_approx_verify_kv = True
            verify_attention_mask = _build_masked_verify_attention_mask(
                model,
                verify_past_key_values,
                tree_candidates,
            )
    else:
        _clear_verify_kv_mapping(model)
        if use_bootstrap_full_kv:
            _increment_stat(statistics, "verify_kv_bootstrap_full_uses")
        elif (
                _verify_kv_enabled(statistics)
                and verify_score_source == "scope"
                and not scope_needs_compression
        ):
            _increment_stat(statistics, "verify_kv_scope_full_decode_uses")

    collect_verify_scores = (
        _verify_kv_enabled(statistics)
        and (
            verify_score_source == "semantic"
            or (
                verify_score_source == "scope"
                and (
                    # SCOPE scores are normally collected only from full-cache
                    # verifier passes; compressed/masked passes would bias the
                    # next selection toward tokens that were already visible.
                    not _verify_kv_scope_score_full_only(statistics)
                    or not used_approx_verify_kv
                )
            )
        )
    )
    if collect_verify_scores:
        _start_verify_attention_collection(model)

    try:
        outputs, tree_logits = swift_verify(
            model,
            tree_candidates,
            past_key_values=verify_past_key_values,
            position_ids=position_ids,
            attention_mask=verify_attention_mask,
        )
    finally:
        if collect_verify_scores:
            _finish_verify_attention_collection(
                model,
                full_len=input_ids.shape[1],
                statistics=statistics,
            )
        if _verify_kv_enabled(statistics):
            statistics["verify_kv_sample_step"] = verify_sample_step + 1

    logits = tree_logits[0, retrieve_indices]

    return logits, outputs, verify_kv_source_data, verify_kv_source_base_offset, used_approx_verify_kv


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

    cache_mode = _draft_kv_cache_mode(statistics)
    if use_compressed_optimization_kv and cache_mode == "mask":
        cur_past_key_values_data = input_past_key_values_data
    else:
        cur_past_key_values_data = []
        for i in range(len(input_past_key_values_data)):
            cur_past_key_values_data.append(input_past_key_values_data[i].clone())
    cur_current_length_data = input_current_length_data.clone()

    if use_compressed_optimization_kv:
        if cache_mode == "mask":
            cur_past_key_values_data, cur_current_length_data, input_past_key_values = prepare_masked_draft_cache(
                model=model,
                full_input_ids=full_input_ids,
                past_key_values_data=cur_past_key_values_data,
                current_length_data=cur_current_length_data,
                retain_ratio=candidate_retain_ratio,
                statistics=statistics,
            )
        else:
            cur_past_key_values_data, cur_current_length_data, input_past_key_values = rebuild_compressed_draft_cache(
                model=model,
                full_input_ids=full_input_ids,
                past_key_values_data=cur_past_key_values_data,
                current_length_data=cur_current_length_data,
                retain_ratio=candidate_retain_ratio,
                statistics=statistics,
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

            optimization_attention_mask = (
                _build_masked_draft_attention_mask(model, input_past_key_values, optimization_input_ids)
                if use_compressed_optimization_kv and cache_mode == "mask"
                else None
            )
            parallel_draft_output = model.model(
                input_ids=optimization_input_ids,
                attention_mask=optimization_attention_mask,
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
        kv_source_data_list=None,
        kv_source_base_offset=None,
        precommitted_kv=False,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate.
    """
    prev_input_len = input_ids.shape[1]
    source_base_offset = prev_input_len if kv_source_base_offset is None else int(kv_source_base_offset)
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + source_base_offset
    )

    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )

    accepted_len = accept_length + 1
    if precommitted_kv:
        current_length_data.fill_(prev_input_len + accepted_len)
    else:
        source_data_list = kv_source_data_list if kv_source_data_list is not None else past_key_values_data_list
        for past_key_values_data, source_past_key_values_data in zip(past_key_values_data_list, source_data_list):
            tgt = source_past_key_values_data[..., select_indices.to(source_past_key_values_data.device), :]
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
