"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import hashlib
import json
import math
import os
import re
import time
from pyexpat import model
import statistics

from fastchat.utils import str_to_torch_dtype

from evaluation_llama.eval import run_eval

from transformers import AutoTokenizer
from bayes_opt import BayesianOptimization, UtilityFunction

from model.swift.utils import *
from model.swift.modeling_llama import LlamaForCausalLM
from model.swift.kv_cache import initialize_past_key_values


def _cache_key_part(value):
    return str(value).replace("/", "_").replace(" ", "_")


def parse_retain_ratio_grid(value, initial_ratio=None):
    retain_ratios = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        retain_ratio = float(item)
        if retain_ratio <= 0.0 or retain_ratio > 1.0:
            raise ValueError("retain ratios must be in the range (0, 1].")
        if not any(abs(retain_ratio - existing) < 1e-9 for existing in retain_ratios):
            retain_ratios.append(retain_ratio)

    if initial_ratio is not None and not any(abs(float(initial_ratio) - existing) < 1e-9 for existing in retain_ratios):
        retain_ratios.insert(0, float(initial_ratio))

    if not retain_ratios:
        raise ValueError("retain ratio grid must contain at least one ratio.")

    return retain_ratios


def format_retain_ratio_grid(retain_ratios):
    return ",".join(str(float(retain_ratio)) for retain_ratio in retain_ratios)


def extend_aggressive_adaptive_ratio_ladder(retain_ratios, min_ratio=0.1, step=0.1):
    values = [float(retain_ratio) for retain_ratio in retain_ratios]
    min_ratio = float(min_ratio)
    step = float(step)
    current = min_ratio
    while current <= 1.0 + 1e-9:
        rounded = round(current, 10)
        if not any(abs(rounded - existing) < 1e-9 for existing in values):
            values.append(rounded)
        current += step
    return sorted(values)


def retain_ratio_run_name(args):
    if getattr(args, "local_adaptive_controller", False):
        if getattr(args, "dynamic_retain_ratio", False):
            return "dynamic3"
        if getattr(args, "cosine_prefill_skip_layers", False):
            if getattr(args, "adaptive_cold_start", False):
                if getattr(args, "verify_kv_compress", False):
                    return "dynamic-6-2"
                return "dynamic-6"
            if getattr(args, "lyapunov_adaptive_controller", False):
                return "dynamic-5-lyapunov"
            if getattr(args, "adaptive_final2_controller", False):
                return "dynamic-4-final2"
            if getattr(args, "adaptive_final_controller", False):
                return "dynamic-4-final"
            if getattr(args, "adaptive_aggressive_controller", False):
                return "dynamic-4-aggressive"
            return "dynamic-4"
        if getattr(args, "load_selected_swift_config", False):
            return "selected-config-adaptive"
        return "adaptive-" + str(args.draft_kv_retain_ratio)
    if args.dynamic_retain_ratio:
        return "dynamic-2"
    if getattr(args, "cosine_prefill_skip_layers", False):
        return "dynamic-4"
    return str(args.draft_kv_retain_ratio)


def build_layer_optimizer(num_hidden_layers, random_state=1):
    pbounds = {f"x{i}": (0, 1) for i in range((num_hidden_layers - 2) * 2)}
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        random_state=random_state,
        verbose=1,
        allow_duplicate_points=True,
    )
    optimizer.set_gp_params(alpha=1e-2)
    return optimizer


def _ratio_key(retain_ratio):
    return str(float(retain_ratio))


def _adaptive_step_config_key(retain_ratio, attn_skip_count):
    return f"ratio-{float(retain_ratio):.4f}|attn-{int(attn_skip_count)}"


def _json_safe_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe_value(value.detach().cpu().item())
        return [_json_safe_value(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _flops_trace_enabled(statistics):
    return bool(statistics and statistics.get("flops_trace_enabled", False))


def _flops_trace_file(statistics):
    if not statistics:
        return None
    trace_file = statistics.get("flops_trace_file")
    return str(trace_file) if trace_file else None


def _write_flops_trace(statistics, entry):
    if not _flops_trace_enabled(statistics):
        return
    trace_file = _flops_trace_file(statistics)
    if not trace_file:
        return
    trace_dir = os.path.dirname(os.path.expanduser(trace_file))
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    with open(os.path.expanduser(trace_file), "a", encoding="utf-8") as fout:
        fout.write(json.dumps(_json_safe_value(entry), ensure_ascii=False) + "\n")


def _model_flop_config(model):
    config = getattr(model, "config", None)
    if config is None:
        return {}
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    num_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads) or num_heads or 0)
    head_dim = hidden_size // num_heads if num_heads > 0 else int(getattr(config, "head_dim", 0) or 0)
    kv_dim = num_kv_heads * head_dim
    return {
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0) or 0),
        "hidden_size": hidden_size,
        "intermediate_size": int(getattr(config, "intermediate_size", 0) or 0),
        "vocab_size": int(getattr(config, "vocab_size", 0) or 0),
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": int(head_dim),
        "kv_dim": int(kv_dim),
        "macs_are_two_flops": True,
        "notes": (
            "Estimated core FLOPs include attention projections, attention QK/AV, "
            "and MLP matmuls. LM-head FLOPs are reported separately. "
            "physical uses the actual dense KV tensor length; logical uses the "
            "visible/selected KV length."
        ),
    }


def _layer_flop_terms(model):
    cfg = _model_flop_config(model)
    hidden_size = int(cfg.get("hidden_size", 0))
    intermediate_size = int(cfg.get("intermediate_size", 0))
    kv_dim = int(cfg.get("kv_dim", hidden_size))
    vocab_size = int(cfg.get("vocab_size", 0))
    return {
        "attn_projection_per_token": int(4 * hidden_size * hidden_size + 4 * hidden_size * kv_dim),
        "attention_per_kv": int(4 * hidden_size),
        "mlp_per_token": int(6 * hidden_size * intermediate_size),
        "lm_head_per_token": int(2 * hidden_size * vocab_size),
    }


def _estimate_component_flops(model, q_tokens, attention_kv_len_sum,
                              attn_layer_count, mlp_layer_count, include_lm_head=True):
    q_tokens = max(0, int(q_tokens))
    attention_kv_len_sum = max(0, int(attention_kv_len_sum))
    attn_layer_count = max(0, int(attn_layer_count))
    mlp_layer_count = max(0, int(mlp_layer_count))
    terms = _layer_flop_terms(model)
    attn_projection = attn_layer_count * q_tokens * terms["attn_projection_per_token"]
    attention_matmul = attn_layer_count * attention_kv_len_sum * terms["attention_per_kv"]
    mlp = mlp_layer_count * q_tokens * terms["mlp_per_token"]
    lm_head = q_tokens * terms["lm_head_per_token"] if include_lm_head else 0
    return {
        "attn_projection": int(attn_projection),
        "attention_matmul": int(attention_matmul),
        "mlp": int(mlp),
        "core": int(attn_projection + attention_matmul + mlp),
        "lm_head": int(lm_head),
        "core_plus_lm_head": int(attn_projection + attention_matmul + mlp + lm_head),
    }


def _sum_estimated_flops(statistics, prefix, estimate):
    if statistics is None or not estimate:
        return
    for key, value in estimate.items():
        statistics[f"flops_estimated_{prefix}_{key}_sum"] = int(
            statistics.get(f"flops_estimated_{prefix}_{key}_sum", 0)
        ) + int(value)


def _current_skip_layers_for_trace(model):
    if not hasattr(model, "get_skip_layers"):
        return [], []
    attn_skip_layers, mlp_skip_layers = model.get_skip_layers()
    return (
        sorted(int(layer_id) for layer_id in list(attn_skip_layers)),
        sorted(int(layer_id) for layer_id in list(mlp_skip_layers)),
    )


def _attention_edge_count_from_swift_buffers(swift_buffers, q_len):
    swift_mask = None
    if swift_buffers is not None:
        swift_mask = swift_buffers.get("swift_attn_mask")
    if swift_mask is None:
        q_len = max(0, int(q_len))
        return q_len * (q_len + 1) // 2
    mask = swift_mask
    while getattr(mask, "dim", lambda: 0)() > 2:
        mask = mask[0]
    mask = mask.detach()
    return int(mask.to(torch.bool).sum().item())


def _kv_trace_lengths(model, prefix, fallback_full_len, compressed_enabled, cache_mode):
    fallback_full_len = max(0, int(fallback_full_len))
    keep_indices = getattr(model, f"{prefix}_last_keep_indices", None)
    mapped_full_len = int(getattr(model, f"{prefix}_last_full_len", 0) or 0)
    mapped_cache_mode = getattr(model, f"{prefix}_last_cache_mode", None)
    if keep_indices is not None and int(keep_indices.numel()) > 0:
        visible_len = int(keep_indices.numel())
        full_len = mapped_full_len if mapped_full_len > 0 else fallback_full_len
        actual_mode = str(mapped_cache_mode or cache_mode or "copy")
        physical_len = full_len if actual_mode == "mask" else visible_len
        compressed = visible_len < full_len
    else:
        full_len = fallback_full_len
        visible_len = fallback_full_len
        physical_len = fallback_full_len
        actual_mode = "full"
        compressed = False
    return {
        "enabled": bool(compressed_enabled),
        "cache_mode": actual_mode,
        "full_len": int(full_len),
        "visible_len": int(visible_len),
        "physical_len": int(physical_len),
        "compressed": bool(compressed),
    }


def _capture_draft_trace(model, statistics, full_input_ids, top1_prob):
    draft_token_count = len(top1_prob) if top1_prob is not None else 0
    full_len = int(full_input_ids.shape[1]) if full_input_ids is not None else 0
    attn_skip_layers, mlp_skip_layers = _current_skip_layers_for_trace(model)
    num_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 0)
    kv_lengths = _kv_trace_lengths(
        model,
        "draft_kv",
        fallback_full_len=full_len,
        compressed_enabled=bool(getattr(model, "draft_kv_compress", False)),
        cache_mode=getattr(model, "draft_kv_cache_mode", statistics.get("draft_kv_cache_mode", "copy")),
    )
    visible_sum = sum(kv_lengths["visible_len"] + offset + 1 for offset in range(draft_token_count))
    physical_sum = sum(kv_lengths["physical_len"] + offset + 1 for offset in range(draft_token_count))
    executed_attn_layers = max(0, num_layers - len(attn_skip_layers))
    executed_mlp_layers = max(0, num_layers - len(mlp_skip_layers))
    return {
        "token_count": int(draft_token_count),
        "retain_ratio": float(getattr(model, "draft_kv_retain_ratio", statistics.get("draft_kv_retain_ratio", 1.0))),
        "kv": kv_lengths,
        "attn_skip_layers": attn_skip_layers,
        "mlp_skip_layers": mlp_skip_layers,
        "attn_skip_count": int(len(attn_skip_layers)),
        "mlp_skip_count": int(len(mlp_skip_layers)),
        "executed_attn_layers": int(executed_attn_layers),
        "executed_mlp_layers": int(executed_mlp_layers),
        "attention_kv_len_sum_logical": int(visible_sum),
        "attention_kv_len_sum_physical": int(physical_sum),
        "estimated_flops_logical": _estimate_component_flops(
            model,
            draft_token_count,
            visible_sum,
            executed_attn_layers,
            executed_mlp_layers,
            include_lm_head=True,
        ),
        "estimated_flops_physical": _estimate_component_flops(
            model,
            draft_token_count,
            physical_sum,
            executed_attn_layers,
            executed_mlp_layers,
            include_lm_head=True,
        ),
    }


def _capture_verify_trace(model, statistics, input_ids, tree_candidates, swift_buffers,
                          used_approx_verify_kv, safe_verify_commit=False):
    full_len = int(input_ids.shape[1]) if input_ids is not None else 0
    q_len = int(tree_candidates.shape[1]) if tree_candidates is not None else 0
    num_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 0)
    kv_lengths = _kv_trace_lengths(
        model,
        "verify_kv",
        fallback_full_len=full_len,
        compressed_enabled=bool(statistics.get("verify_kv_compress", False)),
        cache_mode=statistics.get("verify_kv_cache_mode", "copy"),
    )
    tree_edges = _attention_edge_count_from_swift_buffers(swift_buffers, q_len)
    logical_sum = q_len * kv_lengths["visible_len"] + tree_edges
    physical_sum = q_len * (kv_lengths["physical_len"] + q_len)
    return {
        "q_len": int(q_len),
        "used_approx_kv": bool(used_approx_verify_kv),
        "safe_commit": bool(safe_verify_commit),
        "kv": kv_lengths,
        "score_source": statistics.get("verify_kv_score_source"),
        "retain_ratio": statistics.get("verify_kv_retain_ratio"),
        "scope_beta1": int(statistics.get("verify_kv_scope_beta1", 0) or 0),
        "scope_beta2": int(statistics.get("verify_kv_scope_beta2", 0) or 0),
        "tree_attention_edges_logical": int(tree_edges),
        "attention_kv_len_sum_logical": int(logical_sum),
        "attention_kv_len_sum_physical": int(physical_sum),
        "executed_attn_layers": int(num_layers),
        "executed_mlp_layers": int(num_layers),
        "estimated_flops_logical": _estimate_component_flops(
            model,
            q_len,
            logical_sum,
            num_layers,
            num_layers,
            include_lm_head=True,
        ),
        "estimated_flops_physical": _estimate_component_flops(
            model,
            q_len,
            physical_sum,
            num_layers,
            num_layers,
            include_lm_head=True,
        ),
    }


def _prefill_attention_kv_len_sum(prompt_len, logical=False):
    prompt_len = max(0, int(prompt_len))
    if logical:
        return prompt_len * (prompt_len + 1) // 2
    return prompt_len * prompt_len


def _record_flops_prefill(statistics, model, prompt_len):
    if not _flops_trace_enabled(statistics):
        return
    num_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 0)
    logical_sum = _prefill_attention_kv_len_sum(prompt_len, logical=True)
    physical_sum = _prefill_attention_kv_len_sum(prompt_len, logical=False)
    estimate_logical = _estimate_component_flops(
        model,
        prompt_len,
        logical_sum,
        num_layers,
        num_layers,
        include_lm_head=True,
    )
    estimate_physical = _estimate_component_flops(
        model,
        prompt_len,
        physical_sum,
        num_layers,
        num_layers,
        include_lm_head=True,
    )
    _sum_estimated_flops(statistics, "prefill_logical", estimate_logical)
    _sum_estimated_flops(statistics, "prefill_physical", estimate_physical)
    statistics["flops_trace_prefill_count"] = int(statistics.get("flops_trace_prefill_count", 0)) + 1
    _write_flops_trace(statistics, {
        "record_type": "prefill",
        "phase": statistics.get("flops_trace_phase", "eval"),
        "question_index": statistics.get("flops_trace_question_index"),
        "question_id": statistics.get("flops_trace_question_id"),
        "turn_index": statistics.get("flops_trace_turn_index"),
        "prompt_len": int(prompt_len),
        "num_layers": int(num_layers),
        "attention_kv_len_sum_logical": int(logical_sum),
        "attention_kv_len_sum_physical": int(physical_sum),
        "estimated_flops_logical": estimate_logical,
        "estimated_flops_physical": estimate_physical,
    })


def _accumulate_flops_trace_step(statistics, draft_trace, verify_trace,
                                 safe_commit_trace=None):
    if statistics is None:
        return
    statistics["flops_trace_step_count"] = int(statistics.get("flops_trace_step_count", 0)) + 1
    if draft_trace:
        statistics["flops_trace_draft_token_count"] = int(
            statistics.get("flops_trace_draft_token_count", 0)
        ) + int(draft_trace.get("token_count", 0))
        statistics["flops_trace_draft_kv_visible_len_sum"] = int(
            statistics.get("flops_trace_draft_kv_visible_len_sum", 0)
        ) + int(draft_trace.get("kv", {}).get("visible_len", 0))
        statistics["flops_trace_draft_kv_physical_len_sum"] = int(
            statistics.get("flops_trace_draft_kv_physical_len_sum", 0)
        ) + int(draft_trace.get("kv", {}).get("physical_len", 0))
        statistics["flops_trace_draft_attn_skip_count_sum"] = int(
            statistics.get("flops_trace_draft_attn_skip_count_sum", 0)
        ) + int(draft_trace.get("attn_skip_count", 0))
        statistics["flops_trace_draft_mlp_skip_count_sum"] = int(
            statistics.get("flops_trace_draft_mlp_skip_count_sum", 0)
        ) + int(draft_trace.get("mlp_skip_count", 0))
        _sum_estimated_flops(statistics, "draft_logical", draft_trace.get("estimated_flops_logical", {}))
        _sum_estimated_flops(statistics, "draft_physical", draft_trace.get("estimated_flops_physical", {}))
    if verify_trace:
        statistics["flops_trace_verify_q_token_count"] = int(
            statistics.get("flops_trace_verify_q_token_count", 0)
        ) + int(verify_trace.get("q_len", 0))
        statistics["flops_trace_verify_kv_visible_len_sum"] = int(
            statistics.get("flops_trace_verify_kv_visible_len_sum", 0)
        ) + int(verify_trace.get("kv", {}).get("visible_len", 0))
        statistics["flops_trace_verify_kv_physical_len_sum"] = int(
            statistics.get("flops_trace_verify_kv_physical_len_sum", 0)
        ) + int(verify_trace.get("kv", {}).get("physical_len", 0))
        if verify_trace.get("used_approx_kv", False):
            statistics["flops_trace_verify_approx_steps"] = int(
                statistics.get("flops_trace_verify_approx_steps", 0)
            ) + 1
        _sum_estimated_flops(statistics, "verify_logical", verify_trace.get("estimated_flops_logical", {}))
        _sum_estimated_flops(statistics, "verify_physical", verify_trace.get("estimated_flops_physical", {}))
    if safe_commit_trace:
        _sum_estimated_flops(statistics, "safe_commit_logical", safe_commit_trace.get("estimated_flops_logical", {}))
        _sum_estimated_flops(statistics, "safe_commit_physical", safe_commit_trace.get("estimated_flops_physical", {}))


def _safe_commit_trace(model, input_len, accepted_token_count):
    num_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 0)
    accepted_token_count = max(0, int(accepted_token_count))
    input_len = max(0, int(input_len))
    logical_sum = sum(input_len + offset + 1 for offset in range(accepted_token_count))
    physical_sum = logical_sum
    return {
        "q_len": int(accepted_token_count),
        "attention_kv_len_sum_logical": int(logical_sum),
        "attention_kv_len_sum_physical": int(physical_sum),
        "estimated_flops_logical": _estimate_component_flops(
            model,
            accepted_token_count,
            logical_sum,
            num_layers,
            num_layers,
            include_lm_head=False,
        ),
        "estimated_flops_physical": _estimate_component_flops(
            model,
            accepted_token_count,
            physical_sum,
            num_layers,
            num_layers,
            include_lm_head=False,
        ),
    }


def _record_flops_step(statistics, model, step_entry):
    if not _flops_trace_enabled(statistics):
        return
    _accumulate_flops_trace_step(
        statistics,
        step_entry.get("draft"),
        step_entry.get("verify"),
        step_entry.get("safe_commit"),
    )
    _write_flops_trace(statistics, step_entry)


def _record_flops_sample_summary(statistics, input_len, new_token_num, step_count,
                                 accept_length_list, draft_token_num):
    if not _flops_trace_enabled(statistics):
        return
    accepted_draft_tokens = max(0, int(sum(accept_length_list) - len(accept_length_list)))
    token_acceptance_rate = (
        accepted_draft_tokens / float(draft_token_num)
        if int(draft_token_num) > 0
        else 0.0
    )
    statistics["flops_trace_sample_count"] = int(statistics.get("flops_trace_sample_count", 0)) + 1
    statistics["flops_trace_generated_token_count"] = int(
        statistics.get("flops_trace_generated_token_count", 0)
    ) + int(new_token_num)
    statistics["flops_trace_accepted_draft_token_count"] = int(
        statistics.get("flops_trace_accepted_draft_token_count", 0)
    ) + int(accepted_draft_tokens)
    _write_flops_trace(statistics, {
        "record_type": "sample_summary",
        "phase": statistics.get("flops_trace_phase", "eval"),
        "question_index": statistics.get("flops_trace_question_index"),
        "question_id": statistics.get("flops_trace_question_id"),
        "turn_index": statistics.get("flops_trace_turn_index"),
        "prompt_len": int(input_len),
        "new_token_num": int(new_token_num),
        "step_count": int(step_count),
        "draft_token_num": int(draft_token_num),
        "accepted_draft_tokens": int(accepted_draft_tokens),
        "mean_accepted_tokens": (
            float(sum(accept_length_list) / len(accept_length_list))
            if accept_length_list
            else 0.0
        ),
        "token_acceptance_rate": float(token_acceptance_rate),
    })


def _initialize_flops_trace_file(statistics, model, args, answer_file):
    if not statistics.get("flops_trace_enabled", False):
        return
    trace_file = statistics.get("flops_trace_file")
    if not trace_file:
        root, _ext = os.path.splitext(answer_file)
        trace_hash = hashlib.sha1(os.path.abspath(answer_file).encode("utf-8")).hexdigest()[:12]
        trace_file = os.path.join(os.path.dirname(root), f"flops_trace-{trace_hash}.jsonl")
        statistics["flops_trace_file"] = trace_file
    trace_dir = os.path.dirname(os.path.expanduser(trace_file))
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    meta = {
        "record_type": "run_meta",
        "task_name": args.task_name,
        "model_id": args.model_id,
        "answer_file": answer_file,
        "trace_file": trace_file,
        "model_flop_config": _model_flop_config(model),
        "draft_kv_compress": bool(args.draft_kv_compress),
        "draft_kv_cache_mode": args.draft_kv_cache_mode,
        "draft_kv_score_source": args.draft_kv_score_source,
        "verify_kv_compress": bool(args.verify_kv_compress),
        "verify_kv_cache_mode": args.verify_kv_cache_mode,
        "verify_kv_score_source": args.verify_kv_score_source,
        "verify_kv_dynamic": bool(args.verify_kv_dynamic),
        "local_adaptive_controller": bool(args.local_adaptive_controller),
        "adaptive_final2_controller": bool(args.adaptive_final2_controller),
        "adaptive_cold_start": bool(args.adaptive_cold_start),
    }
    with open(os.path.expanduser(trace_file), "w", encoding="utf-8") as fout:
        fout.write(json.dumps(_json_safe_value(meta), ensure_ascii=False) + "\n")


def _current_attn_skip_count(model, statistics=None):
    if hasattr(model, "get_skip_layers"):
        attn_skip_layers, _mlp_skip_layers = model.get_skip_layers()
        return len(list(attn_skip_layers))
    if statistics is not None:
        return int(statistics.get("cosine_current_attn_skip_count", 0))
    return 0


def _record_adaptive_step_config(statistics, current_ratio, current_attn_skip_count,
                                 accepted_tokens, drafted_tokens):
    config_stats = statistics.setdefault("adaptive_step_config_stats", {})
    key = _adaptive_step_config_key(current_ratio, current_attn_skip_count)
    entry = config_stats.setdefault(key, {
        "current_ratio": float(current_ratio),
        "current_attn_skip_count": int(current_attn_skip_count),
        "step_count": 0,
        "accepted_tokens": 0,
        "drafted_tokens": 0,
        "mean_accepted_step_mean": 0.0,
        "mean_accepted_step_m2": 0.0,
    })
    step_count = int(entry.get("step_count", 0)) + 1
    step_mean_accept = 1.0 + max(0, int(accepted_tokens))
    mean = float(entry.get("mean_accepted_step_mean", 0.0))
    m2 = float(entry.get("mean_accepted_step_m2", 0.0))
    delta = step_mean_accept - mean
    mean += delta / step_count
    delta2 = step_mean_accept - mean
    m2 += delta * delta2
    entry["step_count"] = step_count
    entry["accepted_tokens"] = int(entry.get("accepted_tokens", 0)) + max(0, int(accepted_tokens))
    entry["drafted_tokens"] = int(entry.get("drafted_tokens", 0)) + max(0, int(drafted_tokens))
    entry["mean_accepted_step_mean"] = float(mean)
    entry["mean_accepted_step_m2"] = float(m2)
    statistics["adaptive_step_trace_count"] = int(statistics.get("adaptive_step_trace_count", 0)) + 1


def _parse_optional_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    items = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            items.append(float(item))
    return items


def _parse_optional_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    items = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            items.append(int(item))
    return items


def _unique_sorted_values(values):
    unique_values = []
    for value in values:
        if not any(abs(float(value) - float(existing)) < 1e-9 for existing in unique_values):
            unique_values.append(value)
    return sorted(unique_values)


def _default_adaptive_cold_start_prompt(statistics):
    # Dynamic-6 uses one task-agnostic repeat-passage warmup.  The goal is to
    # exercise KV retention and layer skipping while final2 is already allowed
    # to adapt, not to seed the table with benchmark-specific content.
    return (
        "Repeat the passage below exactly three times. Do not answer, explain, "
        "summarize, translate, or add extra text.\n\n"
        "Passage:\n"
        "Careful systems improve when each step is measured, compared, and adjusted. "
        "A small change can make the next step easier to verify, and a steady record "
        "helps decide what to keep.\n\n"
        "Repeated passage:"
    )


def _cold_start_ratio_values(statistics):
    explicit_ratios = _parse_optional_float_list(statistics.get("adaptive_cold_start_ratios", ""))
    if explicit_ratios:
        return _unique_sorted_values([min(1.0, max(0.0, ratio)) for ratio in explicit_ratios if ratio > 0.0])

    ladder = [float(value) for value in statistics.get("adaptive_ratio_ladder", [])]
    if not ladder:
        initial_ratio = float(statistics.get("adaptive_initial_retain_ratio", statistics.get("draft_kv_retain_ratio", 1.0)))
        ladder = [initial_ratio]
    ladder = _unique_sorted_values(ladder)
    initial_ratio = float(statistics.get("adaptive_initial_retain_ratio", statistics.get("draft_kv_retain_ratio", ladder[0])))
    nearest_index = min(range(len(ladder)), key=lambda index: abs(ladder[index] - initial_ratio))
    start_index = max(0, nearest_index - 1)
    end_index = min(len(ladder), nearest_index + 3)
    selected = list(ladder[start_index:end_index])
    if initial_ratio not in selected:
        selected.append(initial_ratio)
    return _unique_sorted_values(selected)


def _cold_start_max_skip_count(model, statistics):
    num_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0))
    keep_first = max(0, int(statistics.get("cosine_keep_first_layers", 1)))
    keep_last = max(0, int(statistics.get("cosine_keep_last_layers", 2)))
    return max(0, num_layers - keep_first - keep_last)


def _cold_start_skip_count_values(model, statistics):
    max_skip_count = _cold_start_max_skip_count(model, statistics)
    explicit_counts = _parse_optional_int_list(statistics.get("adaptive_cold_start_skip_counts", ""))
    if explicit_counts:
        values = [min(max_skip_count, max(0, count)) for count in explicit_counts]
        return sorted(set(values))

    current_count = _current_attn_skip_count(model, statistics)
    delta = max(1, int(statistics.get("adaptive_cold_start_skip_delta", 2)))
    values = [
        max(0, current_count - delta),
        current_count,
        min(max_skip_count, current_count + delta),
    ]
    return sorted(set(values))


def _build_cold_start_attn_skip_layers(model, statistics, target_skip_count):
    num_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0))
    keep_first = max(0, int(statistics.get("cosine_keep_first_layers", 1)))
    keep_last = max(0, int(statistics.get("cosine_keep_last_layers", 2)))
    last_layer_exclusive = max(keep_first, num_layers - keep_last)
    eligible_layers = list(range(keep_first, last_layer_exclusive))
    ranked_layers = list(statistics.get("cosine_attn_ranking") or [])
    ranked_layers = [int(layer_id) for layer_id in ranked_layers if int(layer_id) in eligible_layers]
    if not ranked_layers:
        ranked_layers = [layer_id for layer_id in eligible_layers if layer_id % 2 == 1]
        ranked_layers += [layer_id for layer_id in eligible_layers if layer_id % 2 == 0]
    target_skip_count = min(max(0, int(target_skip_count)), len(eligible_layers))
    return sorted(ranked_layers[:target_skip_count])


def _merge_adaptive_config_entry(target_entry, source_entry, pseudo_count):
    # Merge cold-start observations as a bounded pseudo-count.  This gives
    # final2 a weak prior without letting the warmup dominate real task stats.
    source_count = max(1, int(source_entry.get("step_count", 0)))
    pseudo_count = max(1, int(pseudo_count))
    source_mean = float(source_entry.get("mean_accepted_step_mean", 0.0))
    source_m2 = float(source_entry.get("mean_accepted_step_m2", 0.0))
    if source_count > 1 and pseudo_count > 1:
        source_variance = max(0.0, source_m2 / (source_count - 1))
        pseudo_m2 = source_variance * (pseudo_count - 1)
    else:
        pseudo_m2 = 0.0

    accepted_per_step = float(source_entry.get("accepted_tokens", 0)) / source_count
    drafted_per_step = float(source_entry.get("drafted_tokens", 0)) / source_count
    pseudo_accepted = int(round(accepted_per_step * pseudo_count))
    pseudo_drafted = int(round(drafted_per_step * pseudo_count))

    target_count = int(target_entry.get("step_count", 0))
    target_mean = float(target_entry.get("mean_accepted_step_mean", 0.0))
    target_m2 = float(target_entry.get("mean_accepted_step_m2", 0.0))
    if target_count <= 0:
        target_entry["step_count"] = pseudo_count
        target_entry["accepted_tokens"] = pseudo_accepted
        target_entry["drafted_tokens"] = pseudo_drafted
        target_entry["mean_accepted_step_mean"] = source_mean
        target_entry["mean_accepted_step_m2"] = pseudo_m2
        target_entry["cold_start_step_count"] = int(target_entry.get("cold_start_step_count", 0)) + pseudo_count
        return

    merged_count = target_count + pseudo_count
    delta = source_mean - target_mean
    merged_mean = target_mean + delta * pseudo_count / merged_count
    merged_m2 = target_m2 + pseudo_m2 + delta * delta * target_count * pseudo_count / merged_count
    target_entry["step_count"] = merged_count
    target_entry["accepted_tokens"] = int(target_entry.get("accepted_tokens", 0)) + pseudo_accepted
    target_entry["drafted_tokens"] = int(target_entry.get("drafted_tokens", 0)) + pseudo_drafted
    target_entry["mean_accepted_step_mean"] = float(merged_mean)
    target_entry["mean_accepted_step_m2"] = float(merged_m2)
    target_entry["cold_start_step_count"] = int(target_entry.get("cold_start_step_count", 0)) + pseudo_count


def _merge_adaptive_cold_start_stats(statistics, cold_statistics):
    target_stats = statistics.setdefault("adaptive_step_config_stats", {})
    source_stats = cold_statistics.get("adaptive_step_config_stats", {})
    effective_count = max(1, int(statistics.get("adaptive_cold_start_effective_count", 16)))
    merged_configs = 0
    merged_steps = 0
    for key, source_entry in source_stats.items():
        source_count = int(source_entry.get("step_count", 0))
        if source_count <= 0:
            continue
        pseudo_count = min(effective_count, source_count)
        target_entry = target_stats.setdefault(key, {
            "current_ratio": float(source_entry.get("current_ratio", 1.0)),
            "current_attn_skip_count": int(source_entry.get("current_attn_skip_count", 0)),
            "step_count": 0,
            "accepted_tokens": 0,
            "drafted_tokens": 0,
            "mean_accepted_step_mean": 0.0,
            "mean_accepted_step_m2": 0.0,
        })
        _merge_adaptive_config_entry(target_entry, source_entry, pseudo_count)
        merged_configs += 1
        merged_steps += pseudo_count
    statistics["adaptive_cold_start_merged_config_count"] = int(statistics.get("adaptive_cold_start_merged_config_count", 0)) + merged_configs
    statistics["adaptive_cold_start_effective_step_count"] = int(statistics.get("adaptive_cold_start_effective_step_count", 0)) + merged_steps


def _merge_cold_start_flops_stats(statistics, cold_statistics):
    if not _flops_trace_enabled(statistics):
        return
    for key, value in cold_statistics.items():
        if key.startswith("flops_estimated_"):
            target_key = key.replace("flops_estimated_", "flops_cold_start_estimated_", 1)
        elif key.startswith("flops_trace_"):
            if key in {
                "flops_trace_enabled",
                "flops_trace_file",
                "flops_trace_phase",
                "flops_trace_question_index",
                "flops_trace_question_id",
                "flops_trace_turn_index",
            }:
                continue
            target_key = key.replace("flops_trace_", "flops_cold_start_trace_", 1)
        else:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            statistics[target_key] = statistics.get(target_key, 0) + value


def _build_cold_start_statistics(statistics, retain_ratio=None, mode="probe"):
    # Cold start runs in an isolated statistics dict.  Only the config table is
    # merged back, so warmup KV state and global acceptance references do not
    # leak into the actual benchmark stream.
    mode = str(mode).lower()
    dynamic_mode = mode == "dynamic"
    if retain_ratio is None:
        retain_ratio = float(statistics.get("adaptive_initial_retain_ratio", statistics.get("draft_kv_retain_ratio", 1.0)))
    retain_ratio = float(retain_ratio)
    ratio_ladder = (
        [float(value) for value in statistics.get("adaptive_ratio_ladder", [retain_ratio])]
        if dynamic_mode
        else [retain_ratio]
    )
    if not any(abs(retain_ratio - value) < 1e-9 for value in ratio_ladder):
        ratio_ladder.append(retain_ratio)
    cold_statistics = dict(statistics)
    cold_statistics.update({
        "optimization": False,
        "bayes": False,
        "local_adaptive_controller": True,
        "dynamic_retain_ratio": False,
        "adaptive_aggressive_controller": False,
        "adaptive_final_controller": False,
        "adaptive_final2_controller": bool(dynamic_mode and statistics.get("adaptive_final2_controller", False)),
        "lyapunov_adaptive_controller": False,
        "adaptive_layer_controller": bool(dynamic_mode and statistics.get("adaptive_layer_controller", False)),
        "cosine_prefill_skip_layers": bool(dynamic_mode and statistics.get("cosine_prefill_skip_layers", False)),
        "draft_kv_retain_ratio": retain_ratio,
        "adaptive_initial_retain_ratio": retain_ratio,
        "adaptive_ratio_ladder": sorted(set(ratio_ladder)),
        "adaptive_window": int(statistics.get("adaptive_window", 16)) if dynamic_mode else 10 ** 9,
        "adaptive_min_observations": int(statistics.get("adaptive_min_observations", 24)) if dynamic_mode else 10 ** 9,
        "adaptive_step_config_stats": {},
        "adaptive_step_trace_count": 0,
        "mean_accepted_tokens": [],
        "accepted_tokens": [],
        "drafted_tokens": [],
        "adaptive_switch_events": [],
        "final2_decisions": [],
        "flops_trace_phase": "adaptive_cold_start",
        "flops_trace_question_index": "adaptive_cold_start",
        "flops_trace_question_id": "adaptive_cold_start",
        "flops_trace_turn_index": None,
    })
    return cold_statistics


def run_adaptive_cold_start(model, tokenizer, statistics, logits_processor=None):
    if not statistics.get("adaptive_cold_start", False):
        return
    if not statistics.get("local_adaptive_controller", False):
        return

    start_time = time.perf_counter()
    mode = str(statistics.get("adaptive_cold_start_mode", "dynamic")).lower()
    prompt = str(statistics.get("adaptive_cold_start_prompt", "")).strip()
    if not prompt or prompt.lower() == "auto":
        prompt = _default_adaptive_cold_start_prompt(statistics)

    ratios = _cold_start_ratio_values(statistics)
    skip_counts = _cold_start_skip_count_values(model, statistics)
    max_configs = max(1, int(statistics.get("adaptive_cold_start_max_configs", 12)))
    max_new_tokens = max(1, int(statistics.get("adaptive_cold_start_max_new_tokens", 128)))
    max_steps = max(1, int(statistics.get("adaptive_cold_start_max_steps", max_new_tokens)))
    original_attn_skip_layers, original_mlp_skip_layers = model.get_skip_layers()
    original_attn_skip_layers = [int(layer_id) for layer_id in list(original_attn_skip_layers)]
    original_mlp_skip_layers = [int(layer_id) for layer_id in list(original_mlp_skip_layers)]
    original_retain_ratio = getattr(model, "draft_kv_retain_ratio", statistics.get("draft_kv_retain_ratio", 1.0))

    configs = []
    if mode == "probe":
        # Probe mode is kept as an ablation: it evaluates fixed ratio/layer
        # pairs and then merges their table entries as low-weight priors.
        for ratio in ratios:
            for skip_count in skip_counts:
                configs.append((float(ratio), int(skip_count)))
                if len(configs) >= max_configs:
                    break
            if len(configs) >= max_configs:
                break
    else:
        # Dynamic mode is the default dynamic-6 path.  It lets final2 adapt
        # during one repeat-passage prompt instead of sweeping synthetic configs.
        configs.append((
            float(statistics.get("adaptive_initial_retain_ratio", statistics.get("draft_kv_retain_ratio", 1.0))),
            _current_attn_skip_count(model, statistics),
        ))

    statistics["adaptive_cold_start_version"] = "dynamic-6"
    statistics["adaptive_cold_start_mode"] = mode
    statistics["adaptive_cold_start_requested_config_count"] = len(configs)
    statistics["adaptive_cold_start_prompt"] = prompt
    raw_steps = 0
    raw_new_tokens = 0
    raw_draft_tokens = 0
    raw_accept_lengths = []

    try:
        device = next(model.parameters()).device
        if getattr(tokenizer, "chat_template", None):
            warmup_input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
        else:
            encoded = tokenizer(prompt, return_tensors="pt")
            warmup_input_ids = encoded["input_ids"].to(device)
        for retain_ratio, skip_count in configs:
            if mode == "probe":
                attn_skip_layers = _build_cold_start_attn_skip_layers(model, statistics, skip_count)
                model.set_skip_layers(attn_skip_layers, original_mlp_skip_layers)
            else:
                model.set_skip_layers(original_attn_skip_layers, original_mlp_skip_layers)
            cold_statistics = _build_cold_start_statistics(statistics, retain_ratio, mode=mode)
            _output_ids, new_token_num, step_count, accept_lengths, draft_token_num = swift_forward(
                warmup_input_ids,
                model,
                tokenizer,
                max_new_tokens=max_new_tokens,
                statistics=cold_statistics,
                optimizer=None,
                utility=None,
                logits_processor=logits_processor,
                max_steps=max_steps,
            )
            _merge_adaptive_cold_start_stats(statistics, cold_statistics)
            _merge_cold_start_flops_stats(statistics, cold_statistics)
            raw_steps += int(step_count)
            raw_new_tokens += int(new_token_num)
            raw_draft_tokens += int(draft_token_num)
            raw_accept_lengths.extend([int(value) for value in accept_lengths])
            if mode != "probe":
                statistics["adaptive_cold_start_dynamic_switches"] = int(cold_statistics.get("adaptive_total_switches", 0))
                statistics["adaptive_cold_start_dynamic_layer_switches"] = int(cold_statistics.get("adaptive_layer_total_switches", 0))
                statistics["adaptive_cold_start_final_ratio_counts"] = dict(cold_statistics.get("adaptive_final_ratio_counts", {}))
                statistics["adaptive_cold_start_ratio_step_counts"] = dict(cold_statistics.get("adaptive_ratio_step_counts", {}))
                statistics["adaptive_cold_start_final2_action_counts"] = dict(cold_statistics.get("adaptive_final2_controller_action_counts", {}))
    finally:
        model.set_skip_layers(original_attn_skip_layers, original_mlp_skip_layers)
        model.draft_kv_retain_ratio = original_retain_ratio
        reset_draft_attention_reuse(model)

    elapsed = time.perf_counter() - start_time
    statistics["adaptive_cold_start_time_sec"] = float(elapsed)
    statistics["adaptive_cold_start_raw_step_count"] = raw_steps
    statistics["adaptive_cold_start_raw_new_tokens"] = raw_new_tokens
    statistics["adaptive_cold_start_raw_draft_tokens"] = raw_draft_tokens
    if raw_accept_lengths:
        statistics["adaptive_cold_start_mean_accepted_tokens"] = float(sum(raw_accept_lengths) / len(raw_accept_lengths))
    if raw_draft_tokens > 0:
        accepted_draft_tokens = max(0, sum(raw_accept_lengths) - len(raw_accept_lengths))
        statistics["adaptive_cold_start_token_acceptance_rate"] = float(accepted_draft_tokens / raw_draft_tokens)


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _lyapunov_config_key(retain_ratio, attn_skip_count):
    return _adaptive_step_config_key(retain_ratio, attn_skip_count)


def _lyapunov_get_attn_skip_count(model):
    if not hasattr(model, "get_skip_layers"):
        return 0
    attn_skip_layers, _mlp_skip_layers = model.get_skip_layers()
    return len(list(attn_skip_layers))


def _lyapunov_skip_denominator(statistics, model):
    eligible = statistics.get("cosine_attn_eligible_layers") or []
    if eligible:
        return max(1, len(eligible))
    num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", 0)
    return max(1, int(num_layers) - 2)


def _lyapunov_update_config_stats(statistics, retain_ratio, attn_skip_count, acceptance):
    config_stats = statistics.setdefault("lyapunov_config_stats", {})
    key = _lyapunov_config_key(retain_ratio, attn_skip_count)
    entry = config_stats.setdefault(key, {"count": 0, "mean": 0.0, "m2": 0.0})
    count = int(entry.get("count", 0)) + 1
    mean = float(entry.get("mean", 0.0))
    m2 = float(entry.get("m2", 0.0))
    delta = float(acceptance) - mean
    mean += delta / count
    delta2 = float(acceptance) - mean
    m2 += delta * delta2
    entry["count"] = count
    entry["mean"] = mean
    entry["m2"] = m2
    return entry


def _lyapunov_predict_acceptance(statistics, current_acceptance, current_ratio,
                                 target_ratio, current_skip_count,
                                 target_skip_count, action_kind):
    config_stats = statistics.get("lyapunov_config_stats", {})
    key = _lyapunov_config_key(target_ratio, target_skip_count)
    std_floor = max(0.0, float(statistics.get("adaptive_std_floor", 0.05)))
    beta = max(0.0, float(statistics.get("lyapunov_prediction_beta", 0.5)))
    entry = config_stats.get(key)
    if entry and int(entry.get("count", 0)) > 0:
        count = int(entry.get("count", 0))
        mean = float(entry.get("mean", current_acceptance))
        if count > 1:
            std = math.sqrt(max(0.0, float(entry.get("m2", 0.0)) / (count - 1)))
        else:
            std = std_floor
        uncertainty = beta * max(std, std_floor) / math.sqrt(count)
        return _clamp01(mean - uncertainty), count, uncertainty, "empirical_lcb"

    ratio_slope = max(0.0, float(statistics.get("lyapunov_ratio_acceptance_slope", 0.2)))
    layer_slope = max(0.0, float(statistics.get("lyapunov_layer_acceptance_slope", 0.015)))
    cold_start_penalty = max(0.0, float(statistics.get("lyapunov_cold_start_penalty", 0.03)))
    predicted = float(current_acceptance)
    predicted += ratio_slope * (float(target_ratio) - float(current_ratio))
    predicted -= layer_slope * (int(target_skip_count) - int(current_skip_count))
    predicted -= cold_start_penalty
    return _clamp01(predicted), 0, cold_start_penalty, "prior_lcb"


def _lyapunov_layer_can_increase(statistics, model, state):
    if not _adaptive_layer_controller_enabled(statistics):
        return False
    max_extra_skip_layers = statistics.get("adaptive_max_extra_skip_layers")
    if max_extra_skip_layers is not None:
        if int(state.get("aggressive_extra_skip_count", 0)) >= int(max_extra_skip_layers):
            return False
    attn_skip_layers, _mlp_skip_layers = model.get_skip_layers()
    attn_skip_layers = [int(layer_id) for layer_id in list(attn_skip_layers)]
    max_skip_layers = statistics.get("adaptive_max_skip_layers")
    if max_skip_layers is not None and len(attn_skip_layers) >= int(max_skip_layers):
        return False
    skipped = set(attn_skip_layers)
    ranked_layers = statistics.get("cosine_attn_ranking") or []
    if not ranked_layers:
        eligible_layers = statistics.get("cosine_attn_eligible_layers") or list(range(len(statistics.get("cosine_attn_scores", []))))
        ranked_layers = sorted(
            [int(layer_id) for layer_id in eligible_layers],
            key=lambda layer_id: (-_cosine_score_for_layer(statistics, layer_id), int(layer_id)),
        )
    return any(int(layer_id) not in skipped for layer_id in ranked_layers)


def _lyapunov_layer_can_reduce(statistics, model):
    if not _adaptive_layer_controller_enabled(statistics):
        return False
    attn_skip_layers, _mlp_skip_layers = model.get_skip_layers()
    return len(list(attn_skip_layers)) > 0


def _initialize_local_adaptive_state(statistics, model):
    if not statistics.get("local_adaptive_controller", False):
        return None

    if statistics.get("dynamic_retain_ratio", False):
        initial_ratio = float(statistics.get(
            "retain_final_ratio",
            statistics.get(
                "best_retain_ratio",
                statistics.get("draft_kv_retain_ratio", 1.0),
            ),
        ))
    else:
        initial_ratio = float(statistics.get(
            "adaptive_initial_retain_ratio",
            statistics.get("draft_kv_retain_ratio", 1.0),
        ))
    ladder = [float(retain_ratio) for retain_ratio in statistics.get("adaptive_ratio_ladder", [initial_ratio])]
    if not any(abs(initial_ratio - retain_ratio) < 1e-9 for retain_ratio in ladder):
        ladder.append(initial_ratio)
    ladder = sorted(set(ladder))
    current_level = min(range(len(ladder)), key=lambda idx: abs(ladder[idx] - initial_ratio))
    current_ratio = float(ladder[current_level])

    model.draft_kv_retain_ratio = current_ratio
    statistics.setdefault("adaptive_total_switches", 0)
    statistics.setdefault("adaptive_question_count", 0)
    statistics.setdefault("adaptive_questions_with_switch", 0)
    statistics.setdefault("adaptive_ratio_step_counts", {})
    statistics.setdefault("adaptive_final_ratio_counts", {})
    statistics.setdefault("adaptive_layer_total_switches", 0)
    statistics.setdefault("adaptive_layer_questions_with_switch", 0)
    statistics.setdefault("adaptive_less_skip_total_switches", 0)
    statistics.setdefault("adaptive_more_skip_total_switches", 0)
    statistics.setdefault("adaptive_aggressive_ratio_down_switches", 0)
    statistics.setdefault("adaptive_final_controller_action_counts", {})
    statistics.setdefault("adaptive_final2_controller_action_counts", {})
    statistics.setdefault("final2_prediction_source_counts", {})
    statistics.setdefault("final2_decision_count", 0)
    statistics.setdefault("final2_keep_decisions", 0)
    statistics.setdefault("lyapunov_ratio_up_switches", 0)
    statistics.setdefault("lyapunov_ratio_down_switches", 0)
    statistics.setdefault("lyapunov_keep_decisions", 0)
    statistics.setdefault("lyapunov_action_counts", {})
    statistics.setdefault("lyapunov_virtual_queue", 0.0)
    statistics.setdefault("lyapunov_virtual_queue_max", 0.0)
    statistics.setdefault("lyapunov_virtual_queue_sum", 0.0)
    statistics.setdefault("lyapunov_decision_count", 0)
    statistics.setdefault("adaptive_global_accepted_tokens", 0)
    statistics.setdefault("adaptive_global_draft_tokens", 0)
    statistics.setdefault("adaptive_global_step_acceptance_history", [])
    statistics.setdefault("adaptive_global_step_mean_accepted_history", [])
    statistics.setdefault("adaptive_step_config_stats", {})
    statistics.setdefault("adaptive_step_trace_count", 0)
    statistics["adaptive_question_count"] += 1
    statistics["adaptive_last_sample"] = None
    statistics["adaptive_current_ratio"] = current_ratio

    return {
        "ladder": ladder,
        "level": current_level,
        "initial_ratio": current_ratio,
        "current_ratio": current_ratio,
        "accepted_history": [],
        "draft_history": [],
        "step_acceptance_history": [],
        "ratio_history": [],
        "switches": [],
        "layer_switches": [],
        "pending_ratio_probe": None,
        "low_streak": 0,
        "high_streak": 0,
        "aggressive_stable_streak": 0,
        "aggressive_extra_skip_count": 0,
        "lyapunov_decisions": [],
        "final2_decisions": [],
        "cooldown": 0,
    }


def _local_adaptive_ready(statistics):
    if not statistics.get("local_adaptive_controller", False):
        return False
    if statistics.get("dynamic_retain_ratio", False):
        return (
            not statistics.get("optimization", False)
            and (
                statistics.get("retain_stage") == "done"
                or "retain_final_ratio" in statistics
                or "best_retain_ratio" in statistics
            )
        )
    return True


def _local_adaptive_acceptance_state(statistics, state, window):
    accepted_history = state["accepted_history"]
    draft_history = state["draft_history"]

    window_accepted = sum(accepted_history[-window:])
    window_draft = sum(draft_history[-window:])
    window_acceptance = window_accepted / window_draft if window_draft > 0 else 0.0

    reference_accepted = int(statistics.get("adaptive_global_accepted_tokens", 0))
    reference_draft = int(statistics.get("adaptive_global_draft_tokens", 0))
    reference_steps = statistics.get("adaptive_global_step_acceptance_history", [])
    reference_acceptance = (
        reference_accepted / reference_draft
        if reference_draft > 0
        else window_acceptance
    )
    reference_std = float(np.std(reference_steps)) if len(reference_steps) > 1 else 0.0
    return window_acceptance, reference_acceptance, reference_std, len(reference_steps)


def _record_local_adaptive_global_step(statistics, accepted_tokens, drafted_tokens, step_acceptance):
    statistics["adaptive_global_accepted_tokens"] = (
        int(statistics.get("adaptive_global_accepted_tokens", 0)) + accepted_tokens
    )
    statistics["adaptive_global_draft_tokens"] = (
        int(statistics.get("adaptive_global_draft_tokens", 0)) + drafted_tokens
    )
    reference_steps = statistics.setdefault("adaptive_global_step_acceptance_history", [])
    reference_steps.append(float(step_acceptance))
    mean_steps = statistics.setdefault("adaptive_global_step_mean_accepted_history", [])
    mean_steps.append(1.0 + max(0, int(accepted_tokens)))


def _set_local_adaptive_ratio(statistics, model, state, new_level, step_idx, direction,
                              window_acceptance, reference_acceptance, threshold_std,
                              lower_threshold, upper_threshold, reference_observations,
                              up_std_k, down_std_k):
    old_ratio = float(state["current_ratio"])
    new_ratio = float(state["ladder"][new_level])
    state["level"] = int(new_level)
    state["current_ratio"] = new_ratio
    state["low_streak"] = 0
    state["high_streak"] = 0
    state["aggressive_stable_streak"] = 0
    state["cooldown"] = int(statistics.get("adaptive_cooldown", 0))
    model.draft_kv_retain_ratio = new_ratio
    statistics["adaptive_current_ratio"] = new_ratio
    statistics["adaptive_total_switches"] += 1
    if direction == "down_aggressive":
        statistics["adaptive_aggressive_ratio_down_switches"] = int(
            statistics.get("adaptive_aggressive_ratio_down_switches", 0)
        ) + 1
    if direction == "lyapunov_ratio_down":
        statistics["lyapunov_ratio_down_switches"] = int(
            statistics.get("lyapunov_ratio_down_switches", 0)
        ) + 1
    if direction == "lyapunov_ratio_up":
        statistics["lyapunov_ratio_up_switches"] = int(
            statistics.get("lyapunov_ratio_up_switches", 0)
        ) + 1

    switch = {
        "step": int(step_idx),
        "direction": direction,
        "old_ratio": old_ratio,
        "new_ratio": new_ratio,
        "window_acceptance": float(window_acceptance),
        "reference_acceptance": float(reference_acceptance),
        "reference_observations": int(reference_observations),
        "threshold_std": float(threshold_std),
        "up_std_k": float(up_std_k),
        "down_std_k": float(down_std_k),
        "lower_threshold": float(lower_threshold),
        "upper_threshold": float(upper_threshold),
    }
    state["switches"].append(switch)
    if _adaptive_layer_controller_enabled(statistics) and direction in ("up", "down_aggressive"):
        state["pending_ratio_probe"] = {
            "direction": direction,
            "step": int(step_idx),
            "old_ratio": old_ratio,
            "new_ratio": new_ratio,
            "baseline_acceptance": float(window_acceptance),
        }
    logging.info(
        "Local adaptive ratio {}: {:.4f} -> {:.4f} "
        "(window {:.4f}, global reference {:.4f}, std {:.4f})".format(
            direction, old_ratio, new_ratio, window_acceptance, reference_acceptance, threshold_std
        )
    )


def _adaptive_layer_controller_enabled(statistics):
    return bool(
        statistics.get("adaptive_layer_controller", False)
        and statistics.get("cosine_prefill_skip_layers", False)
    )


def _cosine_score_for_layer(statistics, layer_id):
    scores = statistics.get("cosine_attn_scores", [])
    if layer_id < 0 or layer_id >= len(scores):
        return -1.0
    try:
        return float(scores[layer_id])
    except (TypeError, ValueError):
        return -1.0


def _reduce_one_cosine_skip_layer(statistics, model, state, step_idx, reason,
                                  window_acceptance=None, reference_acceptance=None):
    if not _adaptive_layer_controller_enabled(statistics):
        return False

    attn_skip_layers, mlp_skip_layers = model.get_skip_layers()
    attn_skip_layers = [int(layer_id) for layer_id in list(attn_skip_layers)]
    mlp_skip_layers = [int(layer_id) for layer_id in list(mlp_skip_layers)]
    if not attn_skip_layers:
        return False

    layer_to_restore = min(
        attn_skip_layers,
        key=lambda layer_id: (_cosine_score_for_layer(statistics, layer_id), -int(layer_id)),
    )
    new_attn_skip_layers = [
        layer_id for layer_id in attn_skip_layers if layer_id != layer_to_restore
    ]
    model.set_skip_layers(new_attn_skip_layers, mlp_skip_layers)

    statistics["cosine_attn_skip_layers"] = [int(layer_id) for layer_id in new_attn_skip_layers]
    statistics["cosine_current_attn_skip_count"] = len(new_attn_skip_layers)
    statistics["adaptive_layer_total_switches"] = int(
        statistics.get("adaptive_layer_total_switches", 0)
    ) + 1
    statistics["adaptive_less_skip_total_switches"] = int(
        statistics.get("adaptive_less_skip_total_switches", 0)
    ) + 1
    state["aggressive_extra_skip_count"] = max(
        0, int(state.get("aggressive_extra_skip_count", 0)) - 1
    )

    switch = {
        "step": int(step_idx),
        "direction": "less_skip",
        "reason": reason,
        "restored_attn_layer": int(layer_to_restore),
        "restored_attn_layer_cosine": float(_cosine_score_for_layer(statistics, layer_to_restore)),
        "old_attn_skip_count": int(len(attn_skip_layers)),
        "new_attn_skip_count": int(len(new_attn_skip_layers)),
    }
    if window_acceptance is not None:
        switch["window_acceptance"] = float(window_acceptance)
    if reference_acceptance is not None:
        switch["reference_acceptance"] = float(reference_acceptance)

    state["layer_switches"].append(switch)
    state["pending_ratio_probe"] = None
    state["low_streak"] = 0
    state["high_streak"] = 0
    state["aggressive_stable_streak"] = 0
    state["cooldown"] = int(statistics.get("adaptive_cooldown", 0))
    logging.info(
        "Local adaptive layer less_skip: restored attn layer %s (cosine %.4f), skip_count %s -> %s",
        layer_to_restore,
        switch["restored_attn_layer_cosine"],
        len(attn_skip_layers),
        len(new_attn_skip_layers),
    )
    return True


def _increase_one_cosine_skip_layer(statistics, model, state, step_idx, reason,
                                    window_acceptance=None, reference_acceptance=None):
    if not _adaptive_layer_controller_enabled(statistics):
        return False

    max_extra_skip_layers = statistics.get("adaptive_max_extra_skip_layers")
    if max_extra_skip_layers is not None:
        if int(state.get("aggressive_extra_skip_count", 0)) >= int(max_extra_skip_layers):
            return False

    attn_skip_layers, mlp_skip_layers = model.get_skip_layers()
    attn_skip_layers = [int(layer_id) for layer_id in list(attn_skip_layers)]
    mlp_skip_layers = [int(layer_id) for layer_id in list(mlp_skip_layers)]
    max_skip_layers = statistics.get("adaptive_max_skip_layers")
    if max_skip_layers is not None and len(attn_skip_layers) >= int(max_skip_layers):
        return False
    skipped = set(attn_skip_layers)

    ranked_layers = statistics.get("cosine_attn_ranking") or []
    if not ranked_layers:
        eligible_layers = statistics.get("cosine_attn_eligible_layers") or list(range(len(statistics.get("cosine_attn_scores", []))))
        ranked_layers = sorted(
            [int(layer_id) for layer_id in eligible_layers],
            key=lambda layer_id: (-_cosine_score_for_layer(statistics, layer_id), int(layer_id)),
        )

    candidates = [int(layer_id) for layer_id in ranked_layers if int(layer_id) not in skipped]
    if not candidates:
        return False

    layer_to_skip = candidates[0]
    new_attn_skip_layers = sorted(attn_skip_layers + [layer_to_skip])
    model.set_skip_layers(new_attn_skip_layers, mlp_skip_layers)

    statistics["cosine_attn_skip_layers"] = [int(layer_id) for layer_id in new_attn_skip_layers]
    statistics["cosine_current_attn_skip_count"] = len(new_attn_skip_layers)
    statistics["adaptive_layer_total_switches"] = int(
        statistics.get("adaptive_layer_total_switches", 0)
    ) + 1
    statistics["adaptive_more_skip_total_switches"] = int(
        statistics.get("adaptive_more_skip_total_switches", 0)
    ) + 1
    state["aggressive_extra_skip_count"] = int(state.get("aggressive_extra_skip_count", 0)) + 1

    switch = {
        "step": int(step_idx),
        "direction": "more_skip",
        "reason": reason,
        "skipped_attn_layer": int(layer_to_skip),
        "skipped_attn_layer_cosine": float(_cosine_score_for_layer(statistics, layer_to_skip)),
        "old_attn_skip_count": int(len(attn_skip_layers)),
        "new_attn_skip_count": int(len(new_attn_skip_layers)),
        "aggressive_extra_skip_count": int(state.get("aggressive_extra_skip_count", 0)),
    }
    if window_acceptance is not None:
        switch["window_acceptance"] = float(window_acceptance)
    if reference_acceptance is not None:
        switch["reference_acceptance"] = float(reference_acceptance)

    state["layer_switches"].append(switch)
    state["pending_ratio_probe"] = None
    state["low_streak"] = 0
    state["high_streak"] = 0
    state["aggressive_stable_streak"] = 0
    state["cooldown"] = int(statistics.get("adaptive_cooldown", 0))
    logging.info(
        "Local adaptive layer more_skip: skipped attn layer %s (cosine %.4f), skip_count %s -> %s",
        layer_to_skip,
        switch["skipped_attn_layer_cosine"],
        len(attn_skip_layers),
        len(new_attn_skip_layers),
    )
    return True


def _maybe_finish_pending_ratio_probe(statistics, model, state, step_idx):
    probe = state.get("pending_ratio_probe")
    if not probe or not _adaptive_layer_controller_enabled(statistics):
        return False

    fallback_window = max(1, int(statistics.get("adaptive_layer_fallback_window", 16)))
    if int(step_idx) - int(probe["step"]) < fallback_window:
        return False

    window = int(statistics.get("adaptive_window", 32))
    (
        window_acceptance,
        reference_acceptance,
        reference_std,
        _reference_observations,
    ) = _local_adaptive_acceptance_state(statistics, state, window)
    min_delta = float(statistics.get("adaptive_layer_improvement_delta", 0.0))
    baseline_acceptance = float(probe.get("baseline_acceptance", 0.0))
    probe_direction = probe.get("direction", "up")

    if probe_direction == "up" and window_acceptance <= baseline_acceptance + min_delta:
        adjusted = _reduce_one_cosine_skip_layer(
            statistics,
            model,
            state,
            step_idx,
            "ratio_probe_no_improvement",
            window_acceptance=window_acceptance,
            reference_acceptance=reference_acceptance,
        )
        if not adjusted:
            state["pending_ratio_probe"] = None
        return adjusted

    if probe_direction == "down_aggressive":
        effective_std = max(reference_std, float(statistics.get("adaptive_std_floor", 0.05)))
        aggressive_tolerance = max(0.0, float(statistics.get("adaptive_aggressive_tolerance", 0.02)))
        aggressive_std_k = max(0.0, float(statistics.get("adaptive_aggressive_std_k", 0.5)))
        no_regression_delta = max(aggressive_tolerance, aggressive_std_k * effective_std)
        if window_acceptance >= baseline_acceptance - no_regression_delta:
            adjusted = _increase_one_cosine_skip_layer(
                statistics,
                model,
                state,
                step_idx,
                "ratio_down_probe_no_regression",
                window_acceptance=window_acceptance,
                reference_acceptance=reference_acceptance,
            )
            if not adjusted:
                state["pending_ratio_probe"] = None
            return adjusted

        state["aggressive_stable_streak"] = 0
        state["cooldown"] = int(statistics.get("adaptive_cooldown", 0))
        state["pending_ratio_probe"] = None
        return True

    state["pending_ratio_probe"] = None
    return False



def _final_adaptive_window_metrics(statistics, state, window):
    accepted_history = state["accepted_history"]
    draft_history = state["draft_history"]
    window_steps = min(max(1, int(window)), len(accepted_history))
    window_accepted = sum(accepted_history[-window_steps:])
    window_draft = sum(draft_history[-window_steps:])
    window_token_acceptance = window_accepted / window_draft if window_draft > 0 else 0.0
    window_mean_accept = 1.0 + (window_accepted / window_steps if window_steps > 0 else 0.0)
    window_draft_len = window_draft / window_steps if window_steps > 0 else 0.0

    reference_steps = len(statistics.get("adaptive_global_step_acceptance_history", []))
    reference_accepted = int(statistics.get("adaptive_global_accepted_tokens", 0))
    reference_draft = int(statistics.get("adaptive_global_draft_tokens", 0))
    reference_token_acceptance = (
        reference_accepted / reference_draft
        if reference_draft > 0
        else window_token_acceptance
    )
    reference_mean_accept = (
        1.0 + reference_accepted / reference_steps
        if reference_steps > 0
        else window_mean_accept
    )
    reference_draft_len = (
        reference_draft / reference_steps
        if reference_steps > 0
        else window_draft_len
    )
    return {
        "window_steps": int(window_steps),
        "window_accepted_tokens": int(window_accepted),
        "window_drafted_tokens": int(window_draft),
        "window_token_acceptance": float(window_token_acceptance),
        "window_mean_accept": float(window_mean_accept),
        "window_draft_len": float(window_draft_len),
        "reference_observations": int(reference_steps),
        "reference_token_acceptance": float(reference_token_acceptance),
        "reference_mean_accept": float(reference_mean_accept),
        "reference_draft_len": float(reference_draft_len),
    }


def _annotate_final_controller_switch(state, switch_kind, metrics, action_score=None):
    if switch_kind == "ratio":
        switches = state.get("switches", [])
    else:
        switches = state.get("layer_switches", [])
    if not switches:
        return
    update = {
        "final_window_mean_accept": float(metrics["window_mean_accept"]),
        "final_window_token_acceptance": float(metrics["window_token_acceptance"]),
        "final_window_draft_len": float(metrics["window_draft_len"]),
        "final_reference_mean_accept": float(metrics["reference_mean_accept"]),
        "final_reference_draft_len": float(metrics["reference_draft_len"]),
    }
    if action_score is not None:
        update["final_action_score"] = float(action_score)
    switches[-1].update(update)


def _record_final_controller_action(statistics, action):
    counts = statistics.setdefault("adaptive_final_controller_action_counts", {})
    counts[action] = int(counts.get(action, 0)) + 1


def _final_can_reduce_layer(statistics, model):
    return _adaptive_layer_controller_enabled(statistics) and _lyapunov_layer_can_reduce(statistics, model)


def _final_can_increase_layer(statistics, model, state, current_skip_count, metrics):
    if not _adaptive_layer_controller_enabled(statistics):
        return False
    hard_max = int(statistics.get("final_hard_max_skip_layers", 18))
    soft_max = int(statistics.get("final_soft_max_skip_layers", min(17, hard_max)))
    if current_skip_count >= hard_max:
        return False
    current_ratio = float(state["current_ratio"])
    min_ratio = float(statistics.get("final_min_ratio_for_more_skip", 0.4))
    if current_ratio < min_ratio:
        return False
    low_guard_ratio = float(statistics.get("final_low_ratio_guard", 0.2))
    low_guard_skip = int(statistics.get("final_low_ratio_guard_skip_layers", 17))
    if current_ratio <= low_guard_ratio and current_skip_count >= low_guard_skip:
        return False

    if current_skip_count < soft_max:
        return _lyapunov_layer_can_increase(statistics, model, state)

    hard_probe_margin = float(statistics.get("final_hard_probe_mean_margin", 0.5))
    hard_probe_token_floor = float(statistics.get("final_hard_probe_token_acceptance_floor", 0.92))
    hard_probe_draft_margin = float(statistics.get("final_hard_probe_draft_len_margin", 0.3))
    target_mean = float(statistics.get("final_target_mean_accepted", 3.0))
    more_draft_floor = float(statistics.get("final_more_skip_draft_len_floor", 2.2))
    if (
        metrics["window_mean_accept"] >= target_mean + hard_probe_margin
        and metrics["window_token_acceptance"] >= hard_probe_token_floor
        and metrics["window_draft_len"] >= more_draft_floor + hard_probe_draft_margin
    ):
        return _lyapunov_layer_can_increase(statistics, model, state)
    return False


def _final_apply_ratio_switch(statistics, model, state, step_idx, new_level, direction, metrics, action_score=None):
    _set_local_adaptive_ratio(
        statistics,
        model,
        state,
        new_level,
        step_idx,
        direction,
        metrics["window_token_acceptance"],
        metrics["reference_token_acceptance"],
        0.0,
        metrics["window_mean_accept"],
        metrics["reference_mean_accept"],
        metrics["reference_observations"],
        0.0,
        0.0,
    )
    _annotate_final_controller_switch(state, "ratio", metrics, action_score=action_score)
    _record_final_controller_action(statistics, direction)
    return True


def _final_apply_layer_reduce(statistics, model, state, step_idx, reason, metrics):
    adjusted = _reduce_one_cosine_skip_layer(
        statistics,
        model,
        state,
        step_idx,
        reason,
        window_acceptance=metrics["window_token_acceptance"],
        reference_acceptance=metrics["reference_token_acceptance"],
    )
    if adjusted:
        _annotate_final_controller_switch(state, "layer", metrics)
        _record_final_controller_action(statistics, "less_skip")
    return adjusted


def _final_apply_layer_increase(statistics, model, state, step_idx, reason, metrics, action_score=None):
    adjusted = _increase_one_cosine_skip_layer(
        statistics,
        model,
        state,
        step_idx,
        reason,
        window_acceptance=metrics["window_token_acceptance"],
        reference_acceptance=metrics["reference_token_acceptance"],
    )
    if adjusted:
        _annotate_final_controller_switch(state, "layer", metrics, action_score=action_score)
        _record_final_controller_action(statistics, "more_skip")
    return adjusted


def _update_final_adaptive_controller(statistics, model, state, step_idx):
    window = int(statistics.get("adaptive_window", 32))
    metrics = _final_adaptive_window_metrics(statistics, state, window)
    current_ratio = float(state["current_ratio"])
    current_level = int(state["level"])
    current_skip_count = _current_attn_skip_count(model, statistics)
    ladder = state["ladder"]
    patience = max(1, int(statistics.get("adaptive_patience", 1)))

    target_mean = float(statistics.get("final_target_mean_accepted", 3.0))
    bad_mean = float(statistics.get("final_bad_mean_accepted", 2.5))
    severe_mean = float(statistics.get("final_severe_mean_accepted", 2.1))
    token_floor = float(statistics.get("final_token_acceptance_floor", 0.85))
    draft_floor = float(statistics.get("final_draft_len_floor", 2.0))
    more_token_floor = float(statistics.get("final_more_skip_token_acceptance_floor", 0.90))
    more_draft_floor = float(statistics.get("final_more_skip_draft_len_floor", 2.2))
    stable_margin = float(statistics.get("final_stable_mean_margin", 0.1))
    hard_max = int(statistics.get("final_hard_max_skip_layers", 18))
    soft_max = int(statistics.get("final_soft_max_skip_layers", min(17, hard_max)))
    low_guard_ratio = float(statistics.get("final_low_ratio_guard", 0.2))
    low_guard_skip = int(statistics.get("final_low_ratio_guard_skip_layers", 17))
    layer18_floor = float(statistics.get("final_hard_layer_mean_floor", 2.8))

    mean_accept = metrics["window_mean_accept"]
    token_acceptance = metrics["window_token_acceptance"]
    draft_len = metrics["window_draft_len"]

    severe = (
        mean_accept < severe_mean
        or token_acceptance < token_floor
        or draft_len < draft_floor
        or current_skip_count > hard_max
    )
    bad = mean_accept < bad_mean

    if severe or bad:
        state["low_streak"] = int(state.get("low_streak", 0)) + 1
        state["high_streak"] = 0
        state["aggressive_stable_streak"] = 0
        if state["low_streak"] < patience:
            return False

        if current_skip_count > hard_max and _final_can_reduce_layer(statistics, model):
            return _final_apply_layer_reduce(statistics, model, state, step_idx, "final_above_hard_skip_cap", metrics)
        if current_skip_count >= hard_max and mean_accept < layer18_floor and _final_can_reduce_layer(statistics, model):
            return _final_apply_layer_reduce(statistics, model, state, step_idx, "final_low_mean_at_hard_skip_cap", metrics)
        if current_skip_count >= low_guard_skip and current_ratio <= low_guard_ratio and current_level < len(ladder) - 1:
            return _final_apply_ratio_switch(statistics, model, state, step_idx, current_level + 1, "final_ratio_up_low_ratio_high_skip", metrics)
        if draft_len < draft_floor and current_level < len(ladder) - 1:
            return _final_apply_ratio_switch(statistics, model, state, step_idx, current_level + 1, "final_ratio_up_short_draft", metrics)
        if current_skip_count > soft_max and _final_can_reduce_layer(statistics, model):
            return _final_apply_layer_reduce(statistics, model, state, step_idx, "final_low_mean_above_soft_skip_cap", metrics)
        if current_level < len(ladder) - 1:
            return _final_apply_ratio_switch(statistics, model, state, step_idx, current_level + 1, "final_ratio_up_low_mean", metrics)
        if _final_can_reduce_layer(statistics, model):
            return _final_apply_layer_reduce(statistics, model, state, step_idx, "final_low_mean_at_max_ratio", metrics)
        return False

    stable = (
        mean_accept >= target_mean + stable_margin
        and token_acceptance >= more_token_floor
        and draft_len >= more_draft_floor
    )
    if not stable:
        state["low_streak"] = 0
        state["high_streak"] = 0
        state["aggressive_stable_streak"] = 0
        return False

    state["high_streak"] = int(state.get("high_streak", 0)) + 1
    state["low_streak"] = 0
    state["aggressive_stable_streak"] = 0
    if state["high_streak"] < patience:
        return False

    candidates = []
    if current_level > 0:
        next_ratio = float(ladder[current_level - 1])
        blocks_low_ratio_high_skip = current_skip_count >= low_guard_skip and next_ratio <= low_guard_ratio
        if not blocks_low_ratio_high_skip:
            score = float(statistics.get("final_ratio_down_gain_weight", 1.0)) * (current_ratio - next_ratio)
            candidates.append((score, "ratio_down", current_level - 1))

    if _final_can_increase_layer(statistics, model, state, current_skip_count, metrics):
        skip_denominator = _lyapunov_skip_denominator(statistics, model)
        score = float(statistics.get("final_layer_skip_gain_weight", 3.0)) / skip_denominator
        candidates.append((score, "more_skip", None))

    if not candidates:
        return False

    candidates.sort(key=lambda item: (item[0], 1 if item[1] == "more_skip" else 0), reverse=True)
    action_score, action, target_level = candidates[0]
    if action == "ratio_down":
        return _final_apply_ratio_switch(
            statistics,
            model,
            state,
            step_idx,
            target_level,
            "final_ratio_down_high_mean",
            metrics,
            action_score=action_score,
        )
    if action == "more_skip":
        return _final_apply_layer_increase(
            statistics,
            model,
            state,
            step_idx,
            "final_high_mean_more_skip",
            metrics,
            action_score=action_score,
        )
    return False


def _record_final2_controller_action(statistics, action):
    counts = statistics.setdefault("adaptive_final2_controller_action_counts", {})
    counts[action] = int(counts.get(action, 0)) + 1


def _record_final2_prediction_source(statistics, source):
    counts = statistics.setdefault("final2_prediction_source_counts", {})
    counts[source] = int(counts.get(source, 0)) + 1


def _final2_window_metrics(statistics, state, window):
    accepted_history = state["accepted_history"]
    draft_history = state["draft_history"]
    window_steps = min(max(1, int(window)), len(accepted_history))
    window_accepted = sum(accepted_history[-window_steps:])
    window_draft = sum(draft_history[-window_steps:])
    window_token_acceptance = window_accepted / window_draft if window_draft > 0 else 0.0
    window_mean_accept = 1.0 + (window_accepted / window_steps if window_steps > 0 else 0.0)
    window_draft_len = window_draft / window_steps if window_steps > 0 else 0.0

    reference_steps = statistics.get("adaptive_global_step_mean_accepted_history", [])
    reference_observations = len(reference_steps)
    if reference_observations > 0:
        reference_mean_accept = float(sum(reference_steps) / reference_observations)
        reference_mean_std = float(np.std(reference_steps)) if reference_observations > 1 else 0.0
    else:
        reference_mean_accept = float(window_mean_accept)
        reference_mean_std = 0.0

    reference_draft = int(statistics.get("adaptive_global_draft_tokens", 0))
    reference_accepted = int(statistics.get("adaptive_global_accepted_tokens", 0))
    reference_token_acceptance = (
        reference_accepted / reference_draft
        if reference_draft > 0
        else window_token_acceptance
    )
    reference_draft_len = (
        reference_draft / reference_observations
        if reference_observations > 0
        else window_draft_len
    )

    std_floor = max(0.0, float(statistics.get("final2_mean_std_floor", 0.10)))
    effective_std = max(reference_mean_std, std_floor)
    low_std_k = max(0.0, float(statistics.get("final2_low_std_k", 0.5)))
    high_std_k = max(0.0, float(statistics.get("final2_high_std_k", 0.5)))
    lower_bound = reference_mean_accept - low_std_k * effective_std
    upper_bound = reference_mean_accept + high_std_k * effective_std

    return {
        "window_steps": int(window_steps),
        "window_accepted_tokens": int(window_accepted),
        "window_drafted_tokens": int(window_draft),
        "window_token_acceptance": float(window_token_acceptance),
        "window_mean_accept": float(window_mean_accept),
        "window_draft_len": float(window_draft_len),
        "reference_observations": int(reference_observations),
        "reference_token_acceptance": float(reference_token_acceptance),
        "reference_mean_accept": float(reference_mean_accept),
        "reference_mean_std": float(reference_mean_std),
        "reference_draft_len": float(reference_draft_len),
        "effective_mean_std": float(effective_std),
        "lower_mean_bound": float(lower_bound),
        "upper_mean_bound": float(upper_bound),
    }


def _final2_predict_config(statistics, metrics, current_ratio, target_ratio,
                           current_skip_count, target_skip_count):
    key = _adaptive_step_config_key(target_ratio, target_skip_count)
    entry = statistics.get("adaptive_step_config_stats", {}).get(key)
    min_observations = max(1, int(statistics.get("final2_min_config_observations", 16)))
    std_floor = max(0.0, float(statistics.get("final2_mean_std_floor", 0.10)))
    beta = max(0.0, float(statistics.get("final2_prediction_beta", 0.5)))
    if entry and int(entry.get("step_count", 0)) >= min_observations:
        count = int(entry.get("step_count", 0))
        mean = float(entry.get(
            "mean_accepted_step_mean",
            (int(entry.get("accepted_tokens", 0)) + count) / count if count > 0 else metrics["window_mean_accept"],
        ))
        if count > 1:
            std = math.sqrt(max(0.0, float(entry.get("mean_accepted_step_m2", 0.0)) / (count - 1)))
        else:
            std = std_floor
        uncertainty = beta * max(std, std_floor) / math.sqrt(count)
        accepted = int(entry.get("accepted_tokens", 0))
        drafted = int(entry.get("drafted_tokens", 0))
        predicted_token_acceptance = accepted / drafted if drafted > 0 else metrics["window_token_acceptance"]
        predicted_draft_len = drafted / count if count > 0 else metrics["window_draft_len"]
        return {
            "predicted_mean_accept": max(1.0, float(mean - uncertainty)),
            "predicted_token_acceptance": float(predicted_token_acceptance),
            "predicted_draft_len": float(predicted_draft_len),
            "empirical_count": int(count),
            "prediction_uncertainty": float(uncertainty),
            "prediction_source": "empirical_lcb",
        }

    ratio_slope = max(0.0, float(statistics.get("final2_ratio_mean_slope", 1.0)))
    layer_slope = max(0.0, float(statistics.get("final2_layer_mean_slope", 0.45)))
    cold_start_penalty = max(0.0, float(statistics.get("final2_cold_start_penalty", 0.15)))
    predicted = float(metrics["window_mean_accept"])
    predicted += ratio_slope * (float(target_ratio) - float(current_ratio))
    predicted -= layer_slope * (int(target_skip_count) - int(current_skip_count))
    predicted -= cold_start_penalty
    return {
        "predicted_mean_accept": max(1.0, float(predicted)),
        "predicted_token_acceptance": float(metrics["window_token_acceptance"]),
        "predicted_draft_len": float(metrics["window_draft_len"]),
        "empirical_count": 0,
        "prediction_uncertainty": float(cold_start_penalty),
        "prediction_source": "prior_lcb",
    }


def _final2_can_increase_layer(statistics, model, state, current_skip_count, metrics):
    if not _adaptive_layer_controller_enabled(statistics):
        return False
    return _lyapunov_layer_can_increase(statistics, model, state)


def _final2_make_candidate(statistics, metrics, current_ratio, current_skip_count,
                           kind, direction, target_level, target_ratio,
                           target_skip_count, switch_cost):
    prediction = _final2_predict_config(
        statistics,
        metrics,
        current_ratio,
        target_ratio,
        current_skip_count,
        target_skip_count,
    )
    candidate = {
        "kind": kind,
        "direction": direction,
        "target_level": target_level,
        "target_ratio": float(target_ratio),
        "target_skip_count": int(target_skip_count),
        "switch_cost": float(switch_cost),
    }
    candidate.update(prediction)
    return candidate


def _final2_record_keep(statistics):
    statistics["final2_keep_decisions"] = int(statistics.get("final2_keep_decisions", 0)) + 1
    _record_final2_controller_action(statistics, "keep")


def _final2_annotate_switch(state, switch_kind, metrics, candidate):
    switches = state.get("switches", []) if switch_kind == "ratio" else state.get("layer_switches", [])
    if not switches:
        return
    switches[-1].update({
        "final2_window_mean_accept": float(metrics["window_mean_accept"]),
        "final2_reference_mean_accept": float(metrics["reference_mean_accept"]),
        "final2_reference_mean_std": float(metrics["reference_mean_std"]),
        "final2_lower_mean_bound": float(metrics["lower_mean_bound"]),
        "final2_upper_mean_bound": float(metrics["upper_mean_bound"]),
        "final2_window_token_acceptance": float(metrics["window_token_acceptance"]),
        "final2_window_draft_len": float(metrics["window_draft_len"]),
        "final2_predicted_mean_accept": float(candidate["predicted_mean_accept"]),
        "final2_predicted_token_acceptance": float(candidate["predicted_token_acceptance"]),
        "final2_predicted_draft_len": float(candidate["predicted_draft_len"]),
        "final2_prediction_source": candidate["prediction_source"],
        "final2_empirical_count": int(candidate["empirical_count"]),
        "final2_prediction_uncertainty": float(candidate["prediction_uncertainty"]),
        "final2_score": float(candidate.get("score", 0.0)),
    })


def _final2_apply_candidate(statistics, model, state, step_idx, metrics, candidate):
    adjusted = False
    if candidate["kind"] in ("ratio_up", "ratio_down"):
        _set_local_adaptive_ratio(
            statistics,
            model,
            state,
            int(candidate["target_level"]),
            step_idx,
            candidate["direction"],
            metrics["window_token_acceptance"],
            metrics["reference_token_acceptance"],
            metrics["effective_mean_std"],
            metrics["lower_mean_bound"],
            metrics["upper_mean_bound"],
            metrics["reference_observations"],
            float(statistics.get("final2_low_std_k", 0.5)),
            float(statistics.get("final2_high_std_k", 0.5)),
        )
        _final2_annotate_switch(state, "ratio", metrics, candidate)
        adjusted = True
    elif candidate["kind"] == "more_skip":
        adjusted = _increase_one_cosine_skip_layer(
            statistics,
            model,
            state,
            step_idx,
            candidate["direction"],
            window_acceptance=metrics["window_token_acceptance"],
            reference_acceptance=metrics["reference_token_acceptance"],
        )
        if adjusted:
            _final2_annotate_switch(state, "layer", metrics, candidate)
    elif candidate["kind"] == "less_skip":
        adjusted = _reduce_one_cosine_skip_layer(
            statistics,
            model,
            state,
            step_idx,
            candidate["direction"],
            window_acceptance=metrics["window_token_acceptance"],
            reference_acceptance=metrics["reference_token_acceptance"],
        )
        if adjusted:
            _final2_annotate_switch(state, "layer", metrics, candidate)

    if adjusted:
        _record_final2_controller_action(statistics, candidate["kind"])
        _record_final2_prediction_source(statistics, candidate["prediction_source"])
        state.setdefault("final2_decisions", []).append({
            "step": int(step_idx),
            "action": candidate["kind"],
            "direction": candidate["direction"],
            "current_ratio": float(state.get("current_ratio", candidate["target_ratio"])),
            "target_ratio": float(candidate["target_ratio"]),
            "target_attn_skip_count": int(candidate["target_skip_count"]),
            "window_mean_accept": float(metrics["window_mean_accept"]),
            "lower_mean_bound": float(metrics["lower_mean_bound"]),
            "upper_mean_bound": float(metrics["upper_mean_bound"]),
            "predicted_mean_accept": float(candidate["predicted_mean_accept"]),
            "prediction_source": candidate["prediction_source"],
            "score": float(candidate.get("score", 0.0)),
        })
    return adjusted


def _update_final2_adaptive_controller(statistics, model, state, step_idx):
    statistics["final2_decision_count"] = int(statistics.get("final2_decision_count", 0)) + 1
    window = int(statistics.get("adaptive_window", 32))
    metrics = _final2_window_metrics(statistics, state, window)
    current_ratio = float(state["current_ratio"])
    current_level = int(state["level"])
    current_skip_count = _current_attn_skip_count(model, statistics)
    ladder = state["ladder"]
    patience = max(1, int(statistics.get("adaptive_patience", 1)))

    token_floor = float(statistics.get("final2_token_acceptance_floor", 0.85))
    more_token_floor = float(statistics.get("final2_more_aggressive_token_acceptance_floor", 0.90))
    draft_floor = float(statistics.get("final2_draft_len_floor", 2.0))
    more_draft_floor = float(statistics.get("final2_more_aggressive_draft_len_floor", 2.2))
    switch_cost = float(statistics.get("final2_switch_cost", 0.02))
    layer_switch_cost = float(statistics.get("final2_layer_switch_cost", switch_cost))

    recovery = (
        metrics["window_mean_accept"] < metrics["lower_mean_bound"]
        or metrics["window_token_acceptance"] < token_floor
        or metrics["window_draft_len"] < draft_floor
    )
    aggressive = (
        metrics["window_mean_accept"] > metrics["upper_mean_bound"]
        and metrics["window_token_acceptance"] >= more_token_floor
        and metrics["window_draft_len"] >= more_draft_floor
    )

    if recovery:
        state["low_streak"] = int(state.get("low_streak", 0)) + 1
        state["high_streak"] = 0
        state["aggressive_stable_streak"] = 0
        if state["low_streak"] < patience:
            _final2_record_keep(statistics)
            return False

        candidates = []
        if current_level < len(ladder) - 1:
            candidates.append(_final2_make_candidate(
                statistics, metrics, current_ratio, current_skip_count,
                "ratio_up", "final2_ratio_up_recovery", current_level + 1,
                float(ladder[current_level + 1]), current_skip_count, switch_cost,
            ))
        if _lyapunov_layer_can_reduce(statistics, model):
            candidates.append(_final2_make_candidate(
                statistics, metrics, current_ratio, current_skip_count,
                "less_skip", "final2_less_skip_recovery", current_level,
                current_ratio, max(0, current_skip_count - 1), layer_switch_cost,
            ))

        if not candidates:
            _final2_record_keep(statistics)
            return False
        for candidate in candidates:
            candidate["score"] = float(candidate["predicted_mean_accept"] - candidate["switch_cost"])
        candidates.sort(key=lambda item: (item["score"], 1 if item["kind"] == "less_skip" else 0), reverse=True)
        return _final2_apply_candidate(statistics, model, state, step_idx, metrics, candidates[0])

    if not aggressive:
        state["low_streak"] = 0
        state["high_streak"] = 0
        state["aggressive_stable_streak"] = 0
        _final2_record_keep(statistics)
        return False

    state["high_streak"] = int(state.get("high_streak", 0)) + 1
    state["low_streak"] = 0
    state["aggressive_stable_streak"] = 0
    if state["high_streak"] < patience:
        _final2_record_keep(statistics)
        return False

    candidates = []
    if current_level > 0:
        target_ratio = float(ladder[current_level - 1])
        candidates.append(_final2_make_candidate(
            statistics, metrics, current_ratio, current_skip_count,
            "ratio_down", "final2_ratio_down_high_mean", current_level - 1,
            target_ratio, current_skip_count, switch_cost,
        ))
    if _final2_can_increase_layer(statistics, model, state, current_skip_count, metrics):
        candidates.append(_final2_make_candidate(
            statistics, metrics, current_ratio, current_skip_count,
            "more_skip", "final2_more_skip_high_mean", current_level,
            current_ratio, current_skip_count + 1, layer_switch_cost,
        ))

    safe_candidates = []
    for candidate in candidates:
        if candidate["predicted_mean_accept"] < metrics["lower_mean_bound"]:
            continue
        if candidate["predicted_token_acceptance"] < more_token_floor:
            continue
        if candidate["predicted_draft_len"] < more_draft_floor:
            continue
        if candidate["kind"] == "ratio_down":
            compression_gain = float(statistics.get("final2_ratio_down_gain_weight", 1.0)) * (
                current_ratio - float(candidate["target_ratio"])
            )
        else:
            skip_denominator = _lyapunov_skip_denominator(statistics, model)
            compression_gain = float(statistics.get("final2_layer_skip_gain_weight", 2.0)) / skip_denominator
        candidate["score"] = float(compression_gain - candidate["switch_cost"])
        safe_candidates.append(candidate)

    if not safe_candidates:
        _final2_record_keep(statistics)
        return False
    safe_candidates.sort(key=lambda item: (item["score"], 1 if item["kind"] == "ratio_down" else 0), reverse=True)
    return _final2_apply_candidate(statistics, model, state, step_idx, metrics, safe_candidates[0])


def _update_lyapunov_adaptive_controller(statistics, model, state, step_idx,
                                          window_acceptance, reference_acceptance,
                                          reference_std, reference_observations):
    target = float(statistics.get("lyapunov_acceptance_target", 0.92))
    if target <= 0.0:
        target = float(reference_acceptance)
    target = _clamp01(target)

    old_queue = max(0.0, float(statistics.get("lyapunov_virtual_queue", 0.0)))
    observed_queue = max(old_queue + target - float(window_acceptance), 0.0)
    statistics["lyapunov_virtual_queue"] = observed_queue
    statistics["lyapunov_virtual_queue_max"] = max(
        float(statistics.get("lyapunov_virtual_queue_max", 0.0)), observed_queue
    )
    statistics["lyapunov_virtual_queue_sum"] = float(
        statistics.get("lyapunov_virtual_queue_sum", 0.0)
    ) + observed_queue
    statistics["lyapunov_decision_count"] = int(
        statistics.get("lyapunov_decision_count", 0)
    ) + 1

    current_ratio = float(state["current_ratio"])
    current_level = int(state["level"])
    current_skip_count = _lyapunov_get_attn_skip_count(model)
    _lyapunov_update_config_stats(
        statistics,
        current_ratio,
        current_skip_count,
        window_acceptance,
    )

    ladder = state["ladder"]
    candidates = [
        {
            "kind": "keep",
            "direction": "keep",
            "target_level": current_level,
            "target_ratio": current_ratio,
            "target_skip_count": current_skip_count,
            "switch_cost": 0.0,
        }
    ]
    if current_level > 0:
        candidates.append({
            "kind": "ratio_down",
            "direction": "lyapunov_ratio_down",
            "target_level": current_level - 1,
            "target_ratio": float(ladder[current_level - 1]),
            "target_skip_count": current_skip_count,
            "switch_cost": float(statistics.get("lyapunov_switch_cost", 0.01)),
        })
    if current_level < len(ladder) - 1:
        candidates.append({
            "kind": "ratio_up",
            "direction": "lyapunov_ratio_up",
            "target_level": current_level + 1,
            "target_ratio": float(ladder[current_level + 1]),
            "target_skip_count": current_skip_count,
            "switch_cost": float(statistics.get("lyapunov_switch_cost", 0.01)),
        })
    if _lyapunov_layer_can_increase(statistics, model, state):
        candidates.append({
            "kind": "more_skip",
            "direction": "more_skip",
            "target_level": current_level,
            "target_ratio": current_ratio,
            "target_skip_count": current_skip_count + 1,
            "switch_cost": float(statistics.get("lyapunov_layer_switch_cost", statistics.get("lyapunov_switch_cost", 0.01))),
        })
    if _lyapunov_layer_can_reduce(statistics, model):
        candidates.append({
            "kind": "less_skip",
            "direction": "less_skip",
            "target_level": current_level,
            "target_ratio": current_ratio,
            "target_skip_count": max(0, current_skip_count - 1),
            "switch_cost": float(statistics.get("lyapunov_layer_switch_cost", statistics.get("lyapunov_switch_cost", 0.01))),
        })

    v_weight = max(0.0, float(statistics.get("lyapunov_v", 0.1)))
    layer_penalty_weight = max(0.0, float(statistics.get("lyapunov_layer_penalty_weight", 0.02)))
    skip_denominator = _lyapunov_skip_denominator(statistics, model)
    best = None
    for candidate in candidates:
        predicted_acceptance, empirical_count, uncertainty, prediction_source = _lyapunov_predict_acceptance(
            statistics,
            window_acceptance,
            current_ratio,
            candidate["target_ratio"],
            current_skip_count,
            candidate["target_skip_count"],
            candidate["kind"],
        )
        predicted_queue = max(observed_queue + target - predicted_acceptance, 0.0)
        normalized_skip = candidate["target_skip_count"] / skip_denominator
        config_penalty = float(candidate["target_ratio"]) - layer_penalty_weight * normalized_skip
        score = (
            0.5 * (observed_queue ** 2 + predicted_queue ** 2)
            + v_weight * config_penalty
            + float(candidate["switch_cost"])
        )
        candidate.update({
            "score": float(score),
            "predicted_acceptance": float(predicted_acceptance),
            "predicted_queue": float(predicted_queue),
            "config_penalty": float(config_penalty),
            "empirical_count": int(empirical_count),
            "prediction_uncertainty": float(uncertainty),
            "prediction_source": prediction_source,
        })
        if best is None or candidate["score"] < best["score"] - 1e-12:
            best = candidate

    if best is None:
        return False

    action_counts = statistics.setdefault("lyapunov_action_counts", {})
    action_counts[best["kind"]] = int(action_counts.get(best["kind"], 0)) + 1
    decision = {
        "step": int(step_idx),
        "action": best["kind"],
        "direction": best["direction"],
        "virtual_queue_before": float(old_queue),
        "virtual_queue_observed": float(observed_queue),
        "acceptance_target": float(target),
        "window_acceptance": float(window_acceptance),
        "reference_acceptance": float(reference_acceptance),
        "reference_observations": int(reference_observations),
        "reference_std": float(reference_std),
        "current_ratio": float(current_ratio),
        "target_ratio": float(best["target_ratio"]),
        "current_attn_skip_count": int(current_skip_count),
        "target_attn_skip_count": int(best["target_skip_count"]),
        "score": float(best["score"]),
        "predicted_acceptance": float(best["predicted_acceptance"]),
        "predicted_queue": float(best["predicted_queue"]),
        "config_penalty": float(best["config_penalty"]),
        "prediction_source": best["prediction_source"],
        "empirical_count": int(best["empirical_count"]),
        "prediction_uncertainty": float(best["prediction_uncertainty"]),
    }
    state.setdefault("lyapunov_decisions", []).append(decision)

    if best["kind"] == "keep":
        statistics["lyapunov_keep_decisions"] = int(
            statistics.get("lyapunov_keep_decisions", 0)
        ) + 1
        return False

    adjusted = False
    if best["kind"] in ("ratio_down", "ratio_up"):
        _set_local_adaptive_ratio(
            statistics,
            model,
            state,
            int(best["target_level"]),
            step_idx,
            best["direction"],
            window_acceptance,
            reference_acceptance,
            max(reference_std, float(statistics.get("adaptive_std_floor", 0.05))),
            target,
            target,
            reference_observations,
            0.0,
            0.0,
        )
        state["switches"][-1].update({
            "lyapunov_score": float(best["score"]),
            "lyapunov_virtual_queue": float(observed_queue),
            "lyapunov_predicted_acceptance": float(best["predicted_acceptance"]),
            "lyapunov_predicted_queue": float(best["predicted_queue"]),
            "lyapunov_config_penalty": float(best["config_penalty"]),
        })
        adjusted = True
    elif best["kind"] == "more_skip":
        adjusted = _increase_one_cosine_skip_layer(
            statistics,
            model,
            state,
            step_idx,
            "lyapunov_policy",
            window_acceptance=window_acceptance,
            reference_acceptance=reference_acceptance,
        )
    elif best["kind"] == "less_skip":
        adjusted = _reduce_one_cosine_skip_layer(
            statistics,
            model,
            state,
            step_idx,
            "lyapunov_policy",
            window_acceptance=window_acceptance,
            reference_acceptance=reference_acceptance,
        )

    if adjusted and state.get("layer_switches") and best["kind"] in ("more_skip", "less_skip"):
        state["layer_switches"][-1].update({
            "lyapunov_score": float(best["score"]),
            "lyapunov_virtual_queue": float(observed_queue),
            "lyapunov_predicted_acceptance": float(best["predicted_acceptance"]),
            "lyapunov_predicted_queue": float(best["predicted_queue"]),
            "lyapunov_config_penalty": float(best["config_penalty"]),
        })
    return adjusted


def _update_local_adaptive_controller(
        statistics, model, state, step_idx, accepted_tokens, drafted_tokens, allow_switch=True):
    if state is None or drafted_tokens <= 0:
        return

    accepted_tokens = max(0, int(accepted_tokens))
    drafted_tokens = max(1, int(drafted_tokens))
    step_acceptance = accepted_tokens / drafted_tokens

    current_ratio = float(state["current_ratio"])
    current_attn_skip_count = _current_attn_skip_count(model, statistics)

    state["accepted_history"].append(accepted_tokens)
    state["draft_history"].append(drafted_tokens)
    state["step_acceptance_history"].append(step_acceptance)
    state["ratio_history"].append(current_ratio)

    _record_adaptive_step_config(
        statistics,
        current_ratio,
        current_attn_skip_count,
        accepted_tokens,
        drafted_tokens,
    )

    ratio_counts = statistics.setdefault("adaptive_ratio_step_counts", {})
    ratio_key = _ratio_key(current_ratio)
    ratio_counts[ratio_key] = int(ratio_counts.get(ratio_key, 0)) + 1

    window = int(statistics.get("adaptive_window", 32))
    min_observations = int(statistics.get("adaptive_min_observations", window))
    observation_count = len(state["step_acceptance_history"])

    if allow_switch and observation_count >= window and observation_count >= min_observations:
        if state["cooldown"] > 0:
            state["cooldown"] -= 1
        else:
            if statistics.get("lyapunov_adaptive_controller", False):
                (
                    window_acceptance,
                    reference_acceptance,
                    reference_std,
                    reference_observations,
                ) = _local_adaptive_acceptance_state(statistics, state, window)
                _update_lyapunov_adaptive_controller(
                    statistics,
                    model,
                    state,
                    step_idx,
                    window_acceptance,
                    reference_acceptance,
                    reference_std,
                    reference_observations,
                )
            elif statistics.get("adaptive_final2_controller", False):
                _update_final2_adaptive_controller(statistics, model, state, step_idx)
            elif statistics.get("adaptive_final_controller", False):
                _update_final_adaptive_controller(statistics, model, state, step_idx)
            else:
                layer_adjusted = _maybe_finish_pending_ratio_probe(statistics, model, state, step_idx)
                if not layer_adjusted and state.get("pending_ratio_probe") is None:
                    (
                        window_acceptance,
                        reference_acceptance,
                        reference_std,
                        reference_observations,
                    ) = _local_adaptive_acceptance_state(statistics, state, window)
                    effective_std = max(reference_std, float(statistics.get("adaptive_std_floor", 0.05)))
                    up_std_k = float(statistics.get("adaptive_up_std_k", statistics.get("adaptive_std_k", 1.0)))
                    down_std_k = float(statistics.get("adaptive_down_std_k", statistics.get("adaptive_std_k", 1.0)))
                    lower_threshold = reference_acceptance - up_std_k * effective_std
                    upper_threshold = reference_acceptance + down_std_k * effective_std
                    patience = max(1, int(statistics.get("adaptive_patience", 1)))

                    aggressive_enabled = bool(statistics.get("adaptive_aggressive_controller", False))
                    aggressive_tolerance = max(0.0, float(statistics.get("adaptive_aggressive_tolerance", 0.02)))
                    aggressive_std_k = max(0.0, float(statistics.get("adaptive_aggressive_std_k", 0.5)))
                    no_regression_delta = max(aggressive_tolerance, aggressive_std_k * effective_std)
                    no_regression_threshold = reference_acceptance - no_regression_delta
                    aggressive_patience = max(1, int(statistics.get("adaptive_aggressive_patience", patience)))

                    if window_acceptance < lower_threshold:
                        state["low_streak"] += 1
                        state["high_streak"] = 0
                        state["aggressive_stable_streak"] = 0
                        if state["low_streak"] >= patience:
                            if state["level"] < len(state["ladder"]) - 1:
                                _set_local_adaptive_ratio(
                                    statistics,
                                    model,
                                    state,
                                    state["level"] + 1,
                                    step_idx,
                                    "up",
                                    window_acceptance,
                                    reference_acceptance,
                                    effective_std,
                                    lower_threshold,
                                    upper_threshold,
                                    reference_observations,
                                    up_std_k,
                                    down_std_k,
                                )
                            else:
                                _reduce_one_cosine_skip_layer(
                                    statistics,
                                    model,
                                    state,
                                    step_idx,
                                    "low_acceptance_at_max_ratio",
                                    window_acceptance=window_acceptance,
                                    reference_acceptance=reference_acceptance,
                                )
                    elif aggressive_enabled and window_acceptance >= no_regression_threshold:
                        state["aggressive_stable_streak"] += 1
                        state["low_streak"] = 0
                        state["high_streak"] = 0
                        if state["aggressive_stable_streak"] >= aggressive_patience:
                            if state["level"] > 0:
                                _set_local_adaptive_ratio(
                                    statistics,
                                    model,
                                    state,
                                    state["level"] - 1,
                                    step_idx,
                                    "down_aggressive",
                                    window_acceptance,
                                    reference_acceptance,
                                    effective_std,
                                    no_regression_threshold,
                                    upper_threshold,
                                    reference_observations,
                                    up_std_k,
                                    down_std_k,
                                )
                            else:
                                _increase_one_cosine_skip_layer(
                                    statistics,
                                    model,
                                    state,
                                    step_idx,
                                    "stable_acceptance_at_min_ratio",
                                    window_acceptance=window_acceptance,
                                    reference_acceptance=reference_acceptance,
                                )
                    elif window_acceptance > upper_threshold:
                        state["high_streak"] += 1
                        state["low_streak"] = 0
                        state["aggressive_stable_streak"] = 0
                        if state["high_streak"] >= patience and state["level"] > 0:
                            _set_local_adaptive_ratio(
                                statistics,
                                model,
                                state,
                                state["level"] - 1,
                                step_idx,
                                "down",
                                window_acceptance,
                                reference_acceptance,
                                effective_std,
                                lower_threshold,
                                upper_threshold,
                                reference_observations,
                                up_std_k,
                                down_std_k,
                            )
                    else:
                        state["low_streak"] = 0
                        state["high_streak"] = 0
                        state["aggressive_stable_streak"] = 0

    _record_local_adaptive_global_step(statistics, accepted_tokens, drafted_tokens, step_acceptance)


def _finish_local_adaptive_sample(statistics, state, total_accepted_tokens, total_draft_tokens):
    if state is None:
        return

    final_ratio = float(state["current_ratio"])
    final_ratio_counts = statistics.setdefault("adaptive_final_ratio_counts", {})
    final_ratio_key = _ratio_key(final_ratio)
    final_ratio_counts[final_ratio_key] = int(final_ratio_counts.get(final_ratio_key, 0)) + 1
    if state["switches"]:
        statistics["adaptive_questions_with_switch"] += 1
    if state["layer_switches"]:
        statistics["adaptive_layer_questions_with_switch"] += 1

    total_draft_tokens = max(0, int(total_draft_tokens))
    total_accepted_tokens = max(0, int(total_accepted_tokens))
    sample_acceptance = total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0
    sample_state = {
        "initial_ratio": float(state["initial_ratio"]),
        "final_ratio": final_ratio,
        "switch_count": len(state["switches"]),
        "switches": state["switches"],
        "layer_switch_count": len(state["layer_switches"]),
        "layer_switches": state["layer_switches"],
        "final_attn_skip_count": int(statistics.get("cosine_current_attn_skip_count", 0)),
        "aggressive_extra_skip_count": int(state.get("aggressive_extra_skip_count", 0)),
        "ratio_step_counts": {},
        "sample_acceptance": float(sample_acceptance),
    }
    for retain_ratio in state["ratio_history"]:
        ratio_key = _ratio_key(retain_ratio)
        sample_state["ratio_step_counts"][ratio_key] = sample_state["ratio_step_counts"].get(ratio_key, 0) + 1
    if statistics.get("lyapunov_adaptive_controller", False):
        sample_state["lyapunov_decision_count"] = len(state.get("lyapunov_decisions", []))
        sample_state["lyapunov_decisions"] = state.get("lyapunov_decisions", [])
        sample_state["lyapunov_final_virtual_queue"] = float(statistics.get("lyapunov_virtual_queue", 0.0))
    statistics["adaptive_last_sample"] = sample_state


def build_skip_layer_cache_key(args):
    draft_token_num = args.draft_token_num if args.draft_token_num is not None else "auto"
    parts = [
        args.model_id,
        args.task_name,
        f"data-{args.data_num}",
        f"seed-{args.seed}",
        f"dtype-{args.dtype}",
        f"temp-{args.temperature}",
        f"top-p-{args.top_p}",
        f"max-new-{args.max_new_tokens}",
        f"opt-interval-{args.opt_interval}",
        f"bayes-interval-{args.bayes_interval}",
        f"max-opt-{args.max_opt_iter}",
        f"max-tolerance-{args.max_tolerance_iter}",
        f"max-score-{args.max_score}",
        f"context-window-{args.context_window}",
        f"skip-ratio-{args.skip_ratio}",
        f"draft-token-num-{draft_token_num}",
        f"opt-compressed-draft-kv-{args.optimize_with_compressed_draft_kv}",
        f"draft-kv-cache-mode-{args.draft_kv_cache_mode}",
        f"draft-kv-score-source-{args.draft_kv_score_source}",
    ]
    if args.dynamic_retain_ratio:
        parts.extend([
            "dynamic-retain-ratio-True",
            f"retain-ratio-grid-{args.retain_ratio_grid}",
            f"retain-target-score-{args.retain_target_score}",
            f"retain-utility-lambda-{args.retain_utility_lambda}",
            f"retain-utility-mode-{args.retain_utility_mode}",
            f"retain-compression-weight-{args.retain_compression_weight}",
            f"retain-score-tolerance-{args.retain_score_tolerance}",
            f"retain-ucb-c-{args.retain_ucb_c}",
            f"retain-warmup-rounds-{args.retain_warmup_rounds}",
            f"retain-filter-top-k-{args.retain_filter_top_k}",
            f"retain-refine-rounds-{args.retain_refine_rounds}",
            f"retain-final-tolerance-{args.retain_final_tolerance}",
            f"final-layer-refine-rounds-{args.final_layer_refine_rounds}",
        ])
    return "__".join(_cache_key_part(part) for part in parts)


def _compile_stop_patterns(stop_config):
    if not stop_config or not stop_config.get("patterns"):
        return []
    return [
        re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
        for pattern in stop_config["patterns"]
    ]


def _should_stop_generation(input_ids, input_len, tokenizer, stop_patterns, stop_config=None):
    if not stop_patterns:
        return False

    generated_ids = input_ids[0, input_len:]
    if generated_ids.numel() == 0:
        return False

    min_chars_before_match = 0
    if stop_config:
        min_chars_before_match = stop_config.get("min_chars_before_match", 0)

    generated_text = tokenizer.decode(
        generated_ids,
        spaces_between_special_tokens=False,
    )
    for pattern in stop_patterns:
        for match in pattern.finditer(generated_text):
            prefix = generated_text[: match.start()].strip()
            if len(prefix) >= min_chars_before_match:
                return True
    return False


def _collect_token_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        token_ids = []
        for item in value:
            token_ids.extend(_collect_token_ids(item))
        return token_ids
    return [int(value)]


def _get_eos_token_ids(model, tokenizer):
    eos_token_ids = set(_collect_token_ids(getattr(tokenizer, "eos_token_id", None)))
    eos_token_ids.update(_collect_token_ids(getattr(getattr(model, "config", None), "eos_token_id", None)))
    eos_token_ids.update(_collect_token_ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None)))
    return eos_token_ids


def _contains_eos_token(token_ids, eos_token_ids):
    return bool(eos_token_ids) and any(int(token_id) in eos_token_ids for token_id in token_ids)


def swift_forward(input_ids, model, tokenizer, max_new_tokens, statistics=None, optimizer=None, utility=None,
                  logits_processor=None, max_steps=512, stop_config=None):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    accept_length_list = []

    # Initialize the past key and value states
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data
    
    model.draft_kv_compress = statistics.get("draft_kv_compress", False)
    model.draft_kv_retain_ratio = statistics.get("draft_kv_retain_ratio", 1.0)
    model.draft_kv_cache_mode = statistics.get("draft_kv_cache_mode", "copy")
    model.draft_kv_score_source = statistics.get("draft_kv_score_source", "heuristic")
    reset_draft_attention_reuse(model)
    statistics["verify_kv_sample_step"] = 0
    statistics["adaptive_last_sample"] = None
    statistics.setdefault("flops_trace_phase", "eval")
    local_adaptive_state = None
    if _local_adaptive_ready(statistics):
        local_adaptive_state = _initialize_local_adaptive_state(statistics, model)
    fixed_draft_token_num = statistics.get("draft_token_num", None)
    stop_patterns = _compile_stop_patterns(stop_config)
    eos_token_ids = _get_eos_token_ids(model, tokenizer)

    input_len = input_ids.shape[1]
    statistics["verify_kv_prefill_len"] = int(input_len)
    cur_length = input_len
    reset_swift_mode(model)
    swift_logits, sample_token, top1_prob = initialize_swift(input_ids, model, max_new_tokens,
                                                             past_key_values, past_key_values_data,
                                                             current_length_data, logits_processor=logits_processor,
                                                             statistics=statistics,
                                                             draft_token_num=fixed_draft_token_num)
    _record_flops_prefill(statistics, model, input_len)
    pending_draft_trace = _capture_draft_trace(model, statistics, input_ids, top1_prob)

    # Clone the prefilled past key and value states for swift optimization
    input_past_key_values_data = []
    for i in range(len(past_key_values_data)):
        input_past_key_values_data.append(past_key_values_data[i].clone())
    input_current_length_data = current_length_data.clone()

    new_token_num = 0
    draft_token_num = 0
    total_acc_num = 0
    for idx in range(max_steps):
        step_input_len_before = int(cur_length)
        step_new_token_num_before = int(new_token_num)
        draft_trace = pending_draft_trace
        # drafted tokens + 1 bonus verified token
        step_draft_token_num = len(top1_prob)
        draft_token_num += step_draft_token_num
        # Initialize the swift buffer
        swift_choices = eval(f"{get_choices_list(top1_prob, logits_processor=logits_processor)}")
        swift_buffers = generate_swift_buffers(swift_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device)
        model.swift_buffers = swift_buffers
        model.swift_choices = swift_choices
        model.model.swift_mask = swift_buffers["swift_attn_mask"]

        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            swift_logits,
            swift_buffers["tree_indices"],
            swift_buffers["retrieve_indices"],
            sample_token,
            logits_processor
        )

        (
            logits,
            outputs,
            verify_kv_source_data,
            verify_kv_source_base_offset,
            used_approx_verify_kv,
        ) = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            swift_buffers["swift_position_ids"],
            input_ids,
            swift_buffers["retrieve_indices"],
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            statistics=statistics,
        )

        best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, swift_logits[2],
                swift_buffers["p_indices"], tree_candidates, swift_buffers["b_indices"]
            )
        verify_margin, verify_min_margin = verify_confidence_margin(
            logits,
            best_candidate,
            accept_length,
        )

        safe_verify_commit = (
            statistics.get("verify_kv_safe_commit", False)
            and used_approx_verify_kv
        )
        verify_trace = _capture_verify_trace(
            model,
            statistics,
            input_ids,
            tree_candidates,
            swift_buffers,
            used_approx_verify_kv=used_approx_verify_kv,
            safe_verify_commit=safe_verify_commit,
        )
        safe_commit_trace = None
        if safe_verify_commit:
            accepted_tokens = candidates[
                None,
                best_candidate,
                : accept_length + 1,
            ].to(input_ids.device)
            if recompute_accepted_kv_full_context(
                model,
                past_key_values_data,
                current_length_data,
                input_ids,
                accepted_tokens,
            ):
                safe_commit_trace = _safe_commit_trace(
                    model,
                    input_len=int(input_ids.shape[1]),
                    accepted_token_count=int(accepted_tokens.shape[1]),
                )
                statistics["verify_kv_safe_commit_uses"] = (
                    int(statistics.get("verify_kv_safe_commit_uses", 0)) + 1
                )

        input_ids, new_token_num, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            swift_buffers["retrieve_indices"],
            logits_processor,
            new_token_num,
            past_key_values_data,
            current_length_data,
            sample_p,
            kv_source_data_list=verify_kv_source_data,
            kv_source_base_offset=verify_kv_source_base_offset,
            precommitted_kv=safe_verify_commit,
        )

        if statistics.get("log_draft_tokens", False):
            try:
                drafted_path = candidates[best_candidate].tolist()
                accepted_path = drafted_path[:accept_length + 1]

                drafted_text = tokenizer.decode(drafted_path)
                accepted_text = tokenizer.decode(accepted_path)

                log_entry = {
                    "step": idx,
                    "draft_kv_compress": model.draft_kv_compress,
                    "draft_kv_retain_ratio": model.draft_kv_retain_ratio,
                    "drafted_tokens": drafted_path,
                    "accepted_tokens": accepted_path,
                    "drafted_text": drafted_text,
                    "accepted_text": accepted_text,
                    "accept_length": int(accept_length)
                }

                log_file = f"without_skipping_token_log_compress_{model.draft_kv_compress}_ratio_{model.draft_kv_retain_ratio}.jsonl"
                with open(log_file, "a") as f:
                    import json
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logging.error(f"Error logging tokens: {e}")

        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        total_acc_num += accept_length_tree - 1
        if statistics.get("draft_kv_score_source") == "reuse" or (
                statistics.get("verify_kv_compress")
                and statistics.get("verify_kv_score_source") == "reuse"):
            apply_draft_attention_reuse_reward(
                model,
                accepted_draft_tokens=accept_length_tree - 1,
                drafted_tokens=step_draft_token_num,
                statistics=statistics,
            )

        should_stop = (
            _contains_eos_token(input_ids[0, input_len:].tolist(), eos_token_ids)
            or _should_stop_generation(input_ids, input_len, tokenizer, stop_patterns, stop_config)
            or new_token_num > max_new_tokens
        )
        update_verify_kv_dynamic_controller(
            statistics,
            accepted_total_tokens=accept_length_tree,
            accepted_draft_tokens=accept_length_tree - 1,
            drafted_tokens=step_draft_token_num,
            confidence_margin=verify_margin,
            confidence_min=verify_min_margin,
            allow_switch=not should_stop,
        )
        _update_local_adaptive_controller(
            statistics,
            model,
            local_adaptive_state,
            idx,
            accept_length_tree - 1,
            step_draft_token_num,
            allow_switch=not should_stop,
        )
        next_attn_skip_layers, next_mlp_skip_layers = _current_skip_layers_for_trace(model)
        _record_flops_step(statistics, model, {
            "record_type": "swift_step",
            "phase": statistics.get("flops_trace_phase", "eval"),
            "question_index": statistics.get("flops_trace_question_index"),
            "question_id": statistics.get("flops_trace_question_id"),
            "turn_index": statistics.get("flops_trace_turn_index"),
            "step": int(idx),
            "input_len_before_step": int(step_input_len_before),
            "new_token_num_before_step": int(step_new_token_num_before),
            "new_token_num_after_step": int(new_token_num),
            "max_new_tokens": int(max_new_tokens),
            "draft": draft_trace,
            "verify": verify_trace,
            "safe_commit": safe_commit_trace,
            "accept": {
                "accepted_total_tokens": int(accept_length_tree),
                "accepted_draft_tokens": int(accept_length_tree - 1),
                "drafted_tokens": int(step_draft_token_num),
                "token_acceptance_rate": (
                    float(max(0, accept_length_tree - 1) / step_draft_token_num)
                    if step_draft_token_num > 0
                    else 0.0
                ),
                "best_candidate": int(best_candidate.item() if hasattr(best_candidate, "item") else best_candidate),
                "confidence_margin": verify_margin,
                "confidence_min_margin": verify_min_margin,
            },
            "controller_after_step": {
                "draft_kv_retain_ratio": float(getattr(model, "draft_kv_retain_ratio", statistics.get("draft_kv_retain_ratio", 1.0))),
                "attn_skip_layers": next_attn_skip_layers,
                "mlp_skip_layers": next_mlp_skip_layers,
                "attn_skip_count": int(len(next_attn_skip_layers)),
                "mlp_skip_count": int(len(next_mlp_skip_layers)),
                "verify_scope_beta1": int(statistics.get("verify_kv_scope_beta1", 0) or 0),
                "verify_scope_beta2": int(statistics.get("verify_kv_scope_beta2", 0) or 0),
                "verify_dynamic_last_action": statistics.get("verify_kv_dynamic_last_action"),
            },
            "should_stop": bool(should_stop),
        })

        if should_stop:
            break

        # layer set optimization
        if (new_token_num > (statistics["context_window"] + 1) and statistics["optimization"]
                and idx % statistics["opt_interval"] == 0):
            swift_optimization(
                model,
                input_ids[:, input_len:],
                input_ids[:, :input_len],  # full_input_ids
                input_past_key_values_data,
                input_current_length_data,
                new_token_num,
                statistics,
                optimizer=optimizer,
                utility=utility)

        # swift drafting
        swift_logits, top1_prob = swift_draft(
            model,
            input_ids=sample_token,
            full_input_ids=input_ids,
            new_token_num=new_token_num,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            draft_token_num=fixed_draft_token_num,
            statistics=statistics,
        )
        pending_draft_trace = _capture_draft_trace(model, statistics, input_ids, top1_prob)
    if draft_token_num > 0:
        logging.info("token acceptance rate: {}".format(total_acc_num / draft_token_num))
    else:
        logging.info("token acceptance rate: 0.0")
    _finish_local_adaptive_sample(statistics, local_adaptive_state, total_acc_num, draft_token_num)
    _record_flops_sample_summary(
        statistics,
        input_len=input_len,
        new_token_num=new_token_num,
        step_count=idx + 1,
        accept_length_list=accept_length_list,
        draft_token_num=draft_token_num,
    )
    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num


def _has_layer_skip(model):
    if not hasattr(model, "get_skip_layers"):
        return False
    attn_skip, mlp_skip = model.get_skip_layers()
    return (len(attn_skip) + len(mlp_skip)) > 0


def _build_compressed_input_ids(full_input_ids, retain_ratio=1.0, min_retain_tokens=16, sink_len=4):
    full_len = full_input_ids.shape[1]
    keep_len = max(min_retain_tokens, int(full_len * retain_ratio))
    keep_len = min(keep_len, full_len)

    if keep_len == full_len:
        return full_input_ids

    if keep_len > sink_len and full_len > keep_len:
        kept_input_ids = torch.cat([
            full_input_ids[:, :sink_len],
            full_input_ids[:, -(keep_len - sink_len):]
        ], dim=1)
    else:
        kept_input_ids = full_input_ids[:, -keep_len:]

    return kept_input_ids


def draft_only_forward(input_ids, model, tokenizer, max_new_tokens, statistics=None, optimizer=None, utility=None,
                       logits_processor=None, max_steps=512, stop_config=None):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()

    model.draft_kv_compress = statistics.get("draft_kv_compress", False)
    model.draft_kv_retain_ratio = statistics.get("draft_kv_retain_ratio", 1.0)

    reset_swift_mode(model)
    model.model.swift_mask = None

    generated_ids = input_ids.clone()
    accept_length_list = []
    use_draft_mode = _has_layer_skip(model)
    input_len = input_ids.shape[1]
    stop_patterns = _compile_stop_patterns(stop_config)
    eos_token_ids = _get_eos_token_ids(model, tokenizer)

    with torch.inference_mode():
        for step in range(min(max_new_tokens, max_steps)):
            # Build the actual context seen by the decoder
            if model.draft_kv_compress and model.draft_kv_retain_ratio < 0.9999:
                context_ids = _build_compressed_input_ids(
                    generated_ids,
                    retain_ratio=model.draft_kv_retain_ratio,
                    min_retain_tokens=16,
                    sink_len=4,
                )
            else:
                context_ids = generated_ids

            model.model.swift_mask = None

            # No layer skip -> use the normal full model path
            # With layer skip -> use self_draft mode
            if use_draft_mode:
                with model.self_draft():
                    outputs = model.model(
                        input_ids=context_ids,
                        attention_mask=None,
                        past_key_values=None,
                    )
            else:
                outputs = model.model(
                    input_ids=context_ids,
                    attention_mask=None,
                    past_key_values=None,
                )

            logits = model.lm_head(outputs[0])
            last_logits = logits[:, -1]

            if logits_processor is not None:
                proc_logits = logits_processor(None, last_logits)
                probabilities = torch.nn.functional.softmax(proc_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
            else:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token.to(generated_ids.device)], dim=1)
            accept_length_list.append(1)

            if _contains_eos_token([next_token.item()], eos_token_ids):
                break
            if _should_stop_generation(generated_ids, input_len, tokenizer, stop_patterns, stop_config):
                break

    new_token_num = generated_ids.shape[1] - input_ids.shape[1]
    return generated_ids, new_token_num, new_token_num, accept_length_list, new_token_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for swift sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="The top-p for sampling.",
    )
    parser.add_argument(
        "--skip-ratio",
        type=float,
        default=0.45,
        help="The skipped layer ratio of swift.",
    )
    parser.add_argument(
        "--opt-interval",
        type=int,
        default=1,
        help="The interval of swift optimization.",
    )
    parser.add_argument(
        "--bayes-interval",
        type=int,
        default=25,
        help="The interval of bayesian optimization.",
    )
    parser.add_argument(
        "--max-opt-iter",
        type=int,
        default=1000,
        help="The maximum layer set optimization iteration.",
    )
    parser.add_argument(
        "--max-tolerance-iter",
        type=int,
        default=300,
        help="The maximum tolerance of layer set search iteration.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.95,
        help="The early stop threshold of layer set search.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32,
        help="The context window of swift.",
    )
    parser.add_argument(
        "--optimization",
        action="store_true",
        default=False,
        help="Layer set optimization.",
    )
    parser.add_argument(
        "--bayes",
        action="store_true",
        default=False,
        help="Bayes Optimization of Layer set.",
    )
    parser.add_argument(
        "--cache-hit",
        action="store_true",
        default=False,
        help="Whether to use cached SWIFT configuration.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-num",
        type=int,
        default=10,
        help="The number of samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="The sampling seed.",
    )
    parser.add_argument(
        "--draft-kv-compress",
        action="store_true",
        default=False,
        help="Whether to compress KV cache only during draft.",
    )

    parser.add_argument(
        "--draft-kv-retain-ratio",
        type=float,
        default=1.0,
        help="Retain ratio of KV cache during draft. 1.0 means no compression.",
    )
    parser.add_argument(
        "--draft-kv-cache-mode",
        type=str,
        default="copy",
        choices=["copy", "mask"],
        help="Draft KV compression implementation: physically copy selected KV, or keep full KV and mask unselected tokens.",
    )
    parser.add_argument(
        "--draft-kv-score-source",
        type=str,
        default="heuristic",
        choices=["heuristic", "observation", "reuse"],
        help="Token-importance source for draft KV compression: heuristic, observation forward, or reused previous draft attention.",
    )
    parser.add_argument(
        "--draft-kv-reuse-ema",
        type=float,
        default=0.7,
        help="EMA weight for previous reused draft attention scores. Used only when --draft-kv-score-source reuse.",
    )
    parser.add_argument(
        "--verify-kv-compress",
        action="store_true",
        default=False,
        help="Approximate mode: also compress KV during tree verification. This does not preserve exact target-model verification.",
    )
    parser.add_argument(
        "--verify-kv-retain-ratio",
        type=float,
        default=None,
        help="Retain ratio for approximate verification KV. Defaults to the current adaptive draft retain ratio.",
    )
    parser.add_argument(
        "--verify-kv-cache-mode",
        type=str,
        default="copy",
        choices=["copy", "mask"],
        help="Approximate verification KV implementation: compact copy or full-layout mask.",
    )
    parser.add_argument(
        "--verify-kv-score-source",
        type=str,
        default="semantic",
        choices=["heuristic", "observation", "reuse", "semantic", "scope"],
        help="Token-importance source for approximate verification KV compression. scope preserves prefill KV and compresses only decoding KV using verifier attention.",
    )
    parser.add_argument(
        "--verify-kv-bootstrap-full-steps",
        type=int,
        default=0,
        help="Use full verifier KV for the first N tree-verification steps of each sample, while still collecting verifier semantic attention for later compressed steps.",
    )
    parser.add_argument(
        "--verify-kv-scope-recent-size",
        type=int,
        default=None,
        help="Deprecated alias for --verify-kv-scope-beta2.",
    )
    parser.add_argument(
        "--verify-kv-scope-beta1",
        type=int,
        default=128,
        help="SCOPE beta1: decoding middle heavy-hitter budget for VERIFY_KV_SCORE_SOURCE=scope.",
    )
    parser.add_argument(
        "--verify-kv-scope-beta2",
        type=int,
        default=256,
        help="SCOPE beta2: recent decoding-token window retained by VERIFY_KV_SCORE_SOURCE=scope.",
    )
    parser.add_argument(
        "--verify-kv-safe-commit",
        action="store_true",
        default=False,
        help="After approximate verification, recompute accepted-token KV with full-context verifier before committing it to the main cache.",
    )
    parser.add_argument(
        "--verify-kv-dynamic",
        action="store_true",
        default=False,
        help="Dynamically adjust SCOPE verifier beta1 using acceptance quality and verifier confidence. Requires --verify-kv-score-source scope.",
    )
    parser.add_argument(
        "--verify-kv-dynamic-min-beta1",
        type=int,
        default=64,
        help="Smallest SCOPE beta1 allowed by --verify-kv-dynamic.",
    )
    parser.add_argument(
        "--verify-kv-dynamic-max-beta1",
        type=int,
        default=None,
        help="Largest SCOPE beta1 allowed by --verify-kv-dynamic. Defaults to the initial --verify-kv-scope-beta1.",
    )
    parser.add_argument(
        "--verify-kv-dynamic-step",
        type=int,
        default=32,
        help="Step size for changing SCOPE beta1 when --verify-kv-dynamic switches.",
    )
    parser.add_argument(
        "--verify-kv-dynamic-window",
        type=int,
        default=32,
        help="Rolling SWIFT-step window used by --verify-kv-dynamic.",
    )
    parser.add_argument(
        "--verify-kv-dynamic-min-observations",
        type=int,
        default=32,
        help="Minimum SWIFT steps before --verify-kv-dynamic may switch beta1.",
    )
    parser.add_argument(
        "--verify-kv-dynamic-acceptance-floor",
        type=float,
        default=0.88,
        help="Minimum rolling draft-token acceptance rate for more aggressive verifier KV compression.",
    )
    parser.add_argument(
        "--verify-kv-dynamic-mean-floor",
        type=float,
        default=3.0,
        help="Minimum rolling mean accepted tokens for more aggressive verifier KV compression.",
    )
    parser.add_argument(
        "--verify-kv-dynamic-confidence-floor",
        type=float,
        default=0.5,
        help="Minimum rolling verifier top1-top2 margin for more aggressive verifier KV compression.",
    )
    parser.add_argument(
        "--verify-kv-dynamic-confidence-low",
        type=float,
        default=0.25,
        help="Rolling verifier margin below this value triggers less aggressive verifier KV compression.",
    )
    parser.add_argument(
        "--verify-kv-dynamic-patience",
        type=int,
        default=2,
        help="Consecutive good/bad rolling windows required before --verify-kv-dynamic changes beta1.",
    )
    parser.add_argument(
        "--verify-kv-dynamic-cooldown",
        type=int,
        default=16,
        help="SWIFT steps to wait after a verifier beta1 switch.",
    )
    parser.add_argument(
        "--dynamic-retain-ratio",
        action="store_true",
        default=False,
        help="Dynamically choose draft KV retain ratio jointly with the skip-layer pattern during inference.",
    )
    parser.add_argument(
        "--retain-ratio-grid",
        type=str,
        default="1.0,0.9,0.8,0.7,0.6",
        help="Comma-separated retain ratios to search when --dynamic-retain-ratio is enabled.",
    )
    parser.add_argument(
        "--retain-target-score",
        type=float,
        default=None,
        help="Matchness target for absolute dynamic retain-ratio utility. Defaults to --max-score.",
    )
    parser.add_argument(
        "--retain-utility-mode",
        type=str,
        default="relative",
        choices=["relative", "additive", "absolute"],
        help="Utility used to compare dynamic retain-ratio candidates.",
    )
    parser.add_argument(
        "--retain-compression-weight",
        type=float,
        default=0.5,
        help="Weight for the compression-gain term in relative/additive dynamic retain-ratio utility.",
    )
    parser.add_argument(
        "--retain-score-tolerance",
        type=float,
        default=0.05,
        help="Allowed matchness drop from the best full-retain reference before applying relative utility penalty.",
    )
    parser.add_argument(
        "--retain-utility-lambda",
        type=float,
        default=1.0,
        help="Penalty applied to matchness deficit in dynamic retain-ratio utility.",
    )
    parser.add_argument(
        "--retain-ucb-c",
        type=float,
        default=0.3,
        help="UCB exploration weight for selecting retain ratios in dynamic mode.",
    )
    parser.add_argument(
        "--retain-warmup-rounds",
        type=int,
        default=50,
        help="Number of initial equal warmup probes to give each retain ratio.",
    )
    parser.add_argument(
        "--retain-filter-top-k",
        type=int,
        default=3,
        help="Number of retain ratios kept after equal warmup for candidate refinement.",
    )
    parser.add_argument(
        "--retain-refine-rounds",
        type=int,
        default=100,
        help="Number of refinement probes to give each retained candidate ratio.",
    )
    parser.add_argument(
        "--retain-final-tolerance",
        type=float,
        default=0.05,
        help="When choosing the final ratio, prefer the lowest retain ratio within this utility gap from the best.",
    )
    parser.add_argument(
        "--final-layer-refine-rounds",
        type=int,
        default=100,
        help="Number of layer-only refinement probes after the final retain ratio is selected.",
    )
    parser.add_argument(
        "--cosine-prefill-skip-layers",
        action="store_true",
        default=False,
        help="Choose draft skip layers from attention cosine similarity collected during the normal prefill.",
    )
    parser.add_argument(
        "--cosine-skip-mode",
        type=str,
        default="topk",
        choices=["topk", "threshold"],
        help="Use cosine ranking by skip budget or thresholding for prefill skip-layer selection.",
    )
    parser.add_argument(
        "--cosine-attn-alpha",
        type=float,
        default=0.985,
        help="Attention cosine threshold used when --cosine-skip-mode threshold is selected.",
    )
    parser.add_argument(
        "--cosine-max-skip-layers",
        type=int,
        default=None,
        help="Maximum number of attention layers to skip in cosine mode. Defaults to skip_ratio times eligible attention layers.",
    )
    parser.add_argument(
        "--cosine-keep-first-layers",
        type=int,
        default=1,
        help="Number of initial layers that cosine skip selection must preserve.",
    )
    parser.add_argument(
        "--cosine-keep-last-layers",
        type=int,
        default=2,
        help="Number of final layers that cosine skip selection must preserve.",
    )
    parser.add_argument(
        "--cosine-mlp-interval",
        type=int,
        default=0,
        help="Optionally skip every m-th MLP layer in cosine mode. 0 disables MLP skipping.",
    )
    parser.add_argument(
        "--adaptive-layer-controller",
        action="store_true",
        default=False,
        help="After an upward retain-ratio switch, restore one cosine-ranked attention layer if acceptance does not improve.",
    )
    parser.add_argument(
        "--adaptive-layer-fallback-window",
        type=int,
        default=16,
        help="Number of SWIFT decoding steps to wait after a ratio increase before restoring one skipped layer.",
    )
    parser.add_argument(
        "--adaptive-layer-improvement-delta",
        type=float,
        default=0.0,
        help="Minimum acceptance improvement required after a ratio increase to avoid restoring one skipped layer.",
    )
    parser.add_argument(
        "--adaptive-aggressive-controller",
        action="store_true",
        default=False,
        help="When local acceptance does not clearly regress, lower retain ratio and then add cosine-ranked skip layers.",
    )
    parser.add_argument(
        "--adaptive-min-retain-ratio",
        type=float,
        default=0.1,
        help="Lowest retain ratio added to the local adaptive ladder when aggressive control is enabled.",
    )
    parser.add_argument(
        "--adaptive-ratio-step",
        type=float,
        default=0.1,
        help="Step used to fill the aggressive local adaptive ratio ladder down to --adaptive-min-retain-ratio.",
    )
    parser.add_argument(
        "--adaptive-aggressive-tolerance",
        type=float,
        default=0.02,
        help="Allowed acceptance drop before aggressive down-shifts are considered unsafe.",
    )
    parser.add_argument(
        "--adaptive-aggressive-std-k",
        type=float,
        default=0.5,
        help="Running-std multiplier used with --adaptive-aggressive-tolerance for no-regression checks.",
    )
    parser.add_argument(
        "--adaptive-aggressive-patience",
        type=int,
        default=1,
        help="Consecutive no-regression windows required before lowering ratio or adding a skip layer.",
    )
    parser.add_argument(
        "--adaptive-max-extra-skip-layers",
        type=int,
        default=None,
        help="Optional cap on extra attention layers skipped by aggressive control within one question.",
    )
    parser.add_argument(
        "--adaptive-max-skip-layers",
        type=int,
        default=None,
        help="Optional absolute cap on final skipped attention layer count during adaptive layer increases.",
    )
    parser.add_argument(
        "--adaptive-final-controller",
        action="store_true",
        default=False,
        help="Use the dynamic-4-final controller: preserve mean accepted tokens, guard token acceptance/draft length, and choose ratio/layer actions jointly.",
    )
    parser.add_argument(
        "--final-target-mean-accepted",
        type=float,
        default=3.0,
        help="Target mean accepted tokens per SWIFT step for dynamic-4-final.",
    )
    parser.add_argument(
        "--final-bad-mean-accepted",
        type=float,
        default=2.5,
        help="Mean accepted tokens below this value triggers recovery in dynamic-4-final.",
    )
    parser.add_argument(
        "--final-severe-mean-accepted",
        type=float,
        default=2.1,
        help="Mean accepted tokens below this value triggers urgent recovery in dynamic-4-final.",
    )
    parser.add_argument(
        "--final-token-acceptance-floor",
        type=float,
        default=0.85,
        help="Minimum token acceptance guard for dynamic-4-final.",
    )
    parser.add_argument(
        "--final-more-skip-token-acceptance-floor",
        type=float,
        default=0.90,
        help="Token acceptance required before dynamic-4-final can add a skipped layer.",
    )
    parser.add_argument(
        "--final-draft-len-floor",
        type=float,
        default=2.0,
        help="Minimum average drafted tokens per step before dynamic-4-final recovers ratio/layers.",
    )
    parser.add_argument(
        "--final-more-skip-draft-len-floor",
        type=float,
        default=2.2,
        help="Average drafted tokens per step required before dynamic-4-final can add a skipped layer.",
    )
    parser.add_argument(
        "--final-stable-mean-margin",
        type=float,
        default=0.1,
        help="Mean accepted token margin above target required before dynamic-4-final becomes more aggressive.",
    )
    parser.add_argument(
        "--final-soft-max-skip-layers",
        type=int,
        default=17,
        help="Soft skipped-attention-layer cap for dynamic-4-final; probing above this needs stronger margins.",
    )
    parser.add_argument(
        "--final-hard-max-skip-layers",
        type=int,
        default=18,
        help="Hard skipped-attention-layer cap for dynamic-4-final.",
    )
    parser.add_argument(
        "--final-min-ratio-for-more-skip",
        type=float,
        default=0.4,
        help="Minimum retain ratio allowed when dynamic-4-final adds a skipped layer.",
    )
    parser.add_argument(
        "--final-low-ratio-guard",
        type=float,
        default=0.2,
        help="Retain-ratio value considered too low to pair with high skipped-layer counts in dynamic-4-final.",
    )
    parser.add_argument(
        "--final-low-ratio-guard-skip-layers",
        type=int,
        default=17,
        help="Skipped-layer count where dynamic-4-final blocks very low retain ratios.",
    )
    parser.add_argument(
        "--final-hard-layer-mean-floor",
        type=float,
        default=2.8,
        help="If hard skip cap is reached and mean accepted tokens falls below this, dynamic-4-final restores a layer.",
    )
    parser.add_argument(
        "--final-hard-probe-mean-margin",
        type=float,
        default=0.5,
        help="Extra mean accepted token margin required to probe above the soft skip cap.",
    )
    parser.add_argument(
        "--final-hard-probe-token-acceptance-floor",
        type=float,
        default=0.92,
        help="Token acceptance required to probe above the soft skip cap.",
    )
    parser.add_argument(
        "--final-hard-probe-draft-len-margin",
        type=float,
        default=0.3,
        help="Extra draft-length margin required to probe above the soft skip cap.",
    )
    parser.add_argument(
        "--final-ratio-down-gain-weight",
        type=float,
        default=1.0,
        help="Compression-gain weight for dynamic-4-final ratio-down candidates.",
    )
    parser.add_argument(
        "--final-layer-skip-gain-weight",
        type=float,
        default=3.0,
        help="Compression-gain weight for dynamic-4-final more-skip candidates.",
    )
    parser.add_argument(
        "--adaptive-final2-controller",
        action="store_true",
        default=False,
        help="Use dynamic-4-final2: adaptive mean-accepted bounds plus empirical ratio/layer config selection.",
    )
    parser.add_argument(
        "--final2-low-std-k",
        type=float,
        default=0.5,
        help="Running mean-accepted std multiplier below the reference mean that triggers recovery.",
    )
    parser.add_argument(
        "--final2-high-std-k",
        type=float,
        default=0.5,
        help="Running mean-accepted std multiplier above the reference mean that allows more aggressive compression.",
    )
    parser.add_argument(
        "--final2-mean-std-floor",
        type=float,
        default=0.10,
        help="Minimum std used for dynamic-4-final2 mean-accepted bounds and config LCB predictions.",
    )
    parser.add_argument(
        "--final2-min-config-observations",
        type=int,
        default=16,
        help="Minimum observations before a ratio/layer config uses empirical LCB instead of the cold-start prior.",
    )
    parser.add_argument(
        "--final2-prediction-beta",
        type=float,
        default=0.5,
        help="LCB uncertainty multiplier for empirical dynamic-4-final2 config predictions.",
    )
    parser.add_argument(
        "--final2-ratio-mean-slope",
        type=float,
        default=1.0,
        help="Prior mean-accepted slope per retain-ratio unit for unseen dynamic-4-final2 candidates.",
    )
    parser.add_argument(
        "--final2-layer-mean-slope",
        type=float,
        default=0.45,
        help="Prior mean-accepted penalty per additional skipped attention layer for unseen candidates.",
    )
    parser.add_argument(
        "--final2-cold-start-penalty",
        type=float,
        default=0.15,
        help="Conservative mean-accepted penalty for unseen dynamic-4-final2 candidates.",
    )
    parser.add_argument(
        "--final2-token-acceptance-floor",
        type=float,
        default=0.85,
        help="Minimum token acceptance guard before dynamic-4-final2 enters recovery.",
    )
    parser.add_argument(
        "--final2-more-aggressive-token-acceptance-floor",
        type=float,
        default=0.90,
        help="Token acceptance required before dynamic-4-final2 can move to a more compressed config.",
    )
    parser.add_argument(
        "--final2-draft-len-floor",
        type=float,
        default=2.0,
        help="Minimum drafted tokens per step before dynamic-4-final2 enters recovery.",
    )
    parser.add_argument(
        "--final2-more-aggressive-draft-len-floor",
        type=float,
        default=2.2,
        help="Drafted tokens per step required before dynamic-4-final2 can move to a more compressed config.",
    )
    parser.add_argument(
        "--final2-switch-cost",
        type=float,
        default=0.02,
        help="Score penalty for dynamic-4-final2 retain-ratio switches.",
    )
    parser.add_argument(
        "--final2-layer-switch-cost",
        type=float,
        default=None,
        help="Score penalty for dynamic-4-final2 layer switches. Defaults to --final2-switch-cost.",
    )
    parser.add_argument(
        "--final2-ratio-down-gain-weight",
        type=float,
        default=1.0,
        help="Compression-gain weight for dynamic-4-final2 ratio-down candidates.",
    )
    parser.add_argument(
        "--final2-layer-skip-gain-weight",
        type=float,
        default=2.0,
        help="Compression-gain weight for dynamic-4-final2 more-skip candidates.",
    )
    parser.add_argument(
        "--adaptive-cold-start",
        action="store_true",
        default=False,
        help="Run one prompt before evaluation to seed the adaptive ratio/layer config table. Named dynamic-6.",
    )
    parser.add_argument(
        "--adaptive-cold-start-mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "probe"],
        help="Cold-start strategy: dynamic runs one repeat-passage warmup with final2 enabled; probe tests fixed config pairs.",
    )
    parser.add_argument(
        "--adaptive-cold-start-prompt",
        type=str,
        default="auto",
        help="Prompt used once at startup to seed the adaptive config table.",
    )
    parser.add_argument(
        "--adaptive-cold-start-max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens generated by each cold-start config.",
    )
    parser.add_argument(
        "--adaptive-cold-start-max-steps",
        type=int,
        default=128,
        help="Maximum SWIFT decode steps per cold-start config.",
    )
    parser.add_argument(
        "--adaptive-cold-start-effective-count",
        type=int,
        default=16,
        help="Maximum pseudo-observations merged into the adaptive table per cold-start config.",
    )
    parser.add_argument(
        "--adaptive-cold-start-max-configs",
        type=int,
        default=12,
        help="Maximum ratio/layer configs probed during cold start.",
    )
    parser.add_argument(
        "--adaptive-cold-start-ratios",
        type=str,
        default="",
        help="Optional comma-separated retain ratios for cold start. Empty uses ratios near the initial ratio.",
    )
    parser.add_argument(
        "--adaptive-cold-start-skip-counts",
        type=str,
        default="",
        help="Optional comma-separated skipped-attention-layer counts for cold start. Empty probes around the current count.",
    )
    parser.add_argument(
        "--adaptive-cold-start-skip-delta",
        type=int,
        default=2,
        help="Layer-count delta used when cold-start skip counts are not explicitly provided.",
    )

    parser.add_argument(
        "--lyapunov-adaptive-controller",
        action="store_true",
        default=False,
        help="Use a Lyapunov virtual-queue objective for local retain-ratio/layer adaptation.",
    )
    parser.add_argument(
        "--lyapunov-acceptance-target",
        type=float,
        default=0.92,
        help="Target window acceptance used to update the Lyapunov virtual queue. <=0 uses the running reference acceptance.",
    )
    parser.add_argument(
        "--lyapunov-v",
        type=float,
        default=0.1,
        help="Weight on the compression penalty in the Lyapunov drift-plus-penalty objective.",
    )
    parser.add_argument(
        "--lyapunov-switch-cost",
        type=float,
        default=0.01,
        help="Cost added when the Lyapunov controller changes retain ratio.",
    )
    parser.add_argument(
        "--lyapunov-layer-switch-cost",
        type=float,
        default=None,
        help="Optional cost added when the Lyapunov controller changes skip-layer count. Defaults to --lyapunov-switch-cost.",
    )
    parser.add_argument(
        "--lyapunov-layer-penalty-weight",
        type=float,
        default=0.02,
        help="Static configuration penalty weight that favors more skipped attention layers without using speedup feedback.",
    )
    parser.add_argument(
        "--lyapunov-prediction-beta",
        type=float,
        default=0.5,
        help="Lower-confidence-bound multiplier for empirical acceptance prediction.",
    )
    parser.add_argument(
        "--lyapunov-ratio-acceptance-slope",
        type=float,
        default=0.2,
        help="Cold-start prior acceptance change per retain-ratio delta.",
    )
    parser.add_argument(
        "--lyapunov-layer-acceptance-slope",
        type=float,
        default=0.015,
        help="Cold-start prior acceptance drop per additional skipped attention layer.",
    )
    parser.add_argument(
        "--lyapunov-cold-start-penalty",
        type=float,
        default=0.03,
        help="Conservative penalty applied to unseen Lyapunov candidate configurations.",
    )
    parser.add_argument(
        "--optimize-with-compressed-draft-kv",
        dest="optimize_with_compressed_draft_kv",
        action="store_true",
        default=True,
        help="Use compressed draft KV cache when scoring layer-skip optimization.",
    )
    parser.add_argument(
        "--no-optimize-with-compressed-draft-kv",
        dest="optimize_with_compressed_draft_kv",
        action="store_false",
        help="Use the uncompressed prompt KV cache when scoring layer-skip optimization.",
    )
    parser.add_argument(
        "--draft-token-num",
        type=int,
        default=None,
        help="Fixed number of tokens to draft each SWIFT step. If unset, use stop_threshold.",
    )
    parser.add_argument(
        "--log-draft-tokens",
        action="store_true",
        default=False,
        help="Log drafted/accepted tokens for every SWIFT step. Disabled by default because it adds decode and file I/O overhead.",
    )
    parser.add_argument(
        "--flops-trace",
        action="store_true",
        default=False,
        help="Write a JSONL trace with per-step KV lengths, layer skips, acceptance, and estimated FLOPs inputs.",
    )
    parser.add_argument(
        "--flops-trace-file",
        type=str,
        default=None,
        help="Output JSONL path for --flops-trace. Defaults to the answer file with .flops_trace.jsonl suffix.",
    )
    parser.add_argument(
        "--draft-only",
        action="store_true",
        default=False,
        help="Use the draft model natively as the main model for decoding.",
    )
    parser.add_argument(
        "--skip-layer-cache-file",
        type=str,
        default="outputs/skip_layer_cache.json",
        help="File used to save/load benchmark-specific skip-layer sets.",
    )
    parser.add_argument(
        "--skip-layer-cache-key",
        type=str,
        default=None,
        help="Optional explicit key for the skip-layer cache.",
    )
    parser.add_argument(
        "--save-skip-layer-cache",
        action="store_true",
        default=False,
        help="Save the final optimized skip-layer set after evaluation.",
    )
    parser.add_argument(
        "--load-skip-layer-cache",
        action="store_true",
        default=False,
        help="Load a skip-layer set and disable layer-set optimization.",
    )
    parser.add_argument(
        "--selected-swift-config-file",
        type=str,
        default="outputs/selected_swift_config.json",
        help="File containing one selected retain-ratio/skip-layer config per benchmark.",
    )
    parser.add_argument(
        "--load-selected-swift-config",
        action="store_true",
        default=False,
        help="Load the selected benchmark-level retain-ratio/skip-layer config and disable optimization.",
    )
    parser.add_argument(
        "--local-adaptive-controller",
        action="store_true",
        default=False,
        help="Adapt draft KV retain ratio within each question using local acceptance statistics.",
    )
    parser.add_argument(
        "--adaptive-ratio-ladder",
        type=str,
        default="0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated retain-ratio ladder for local adaptive control.",
    )
    parser.add_argument(
        "--adaptive-window",
        type=int,
        default=16,
        help="Sliding window size, in SWIFT decoding steps, for local adaptive control.",
    )
    parser.add_argument(
        "--adaptive-min-observations",
        type=int,
        default=24,
        help="Minimum observed SWIFT decoding steps before local adaptive switching is enabled.",
    )
    parser.add_argument(
        "--adaptive-std-k",
        type=float,
        default=0.5,
        help="Fallback number of global running standard deviations used as the adaptive switch threshold.",
    )
    parser.add_argument(
        "--adaptive-up-std-k",
        type=float,
        default=None,
        help="Global running standard-deviation multiplier used when raising retain ratio.",
    )
    parser.add_argument(
        "--adaptive-down-std-k",
        type=float,
        default=None,
        help="Global running standard-deviation multiplier used when lowering retain ratio.",
    )
    parser.add_argument(
        "--adaptive-std-floor",
        type=float,
        default=0.05,
        help="Lower bound for the running acceptance standard deviation.",
    )
    parser.add_argument(
        "--adaptive-patience",
        type=int,
        default=1,
        help="Consecutive low/high windows required before changing retain ratio.",
    )
    parser.add_argument(
        "--adaptive-cooldown",
        type=int,
        default=8,
        help="Number of SWIFT decoding steps to wait after an adaptive ratio switch.",
    )

    args = parser.parse_args()
    if args.draft_token_num is not None and args.draft_token_num <= 0:
        parser.error("--draft-token-num must be a positive integer.")
    if args.draft_kv_retain_ratio <= 0.0 or args.draft_kv_retain_ratio > 1.0:
        parser.error("--draft-kv-retain-ratio must be in the range (0, 1].")
    if args.draft_kv_reuse_ema < 0.0 or args.draft_kv_reuse_ema > 1.0:
        parser.error("--draft-kv-reuse-ema must be in the range [0, 1].")
    if args.verify_kv_retain_ratio is not None and (
            args.verify_kv_retain_ratio <= 0.0 or args.verify_kv_retain_ratio > 1.0):
        parser.error("--verify-kv-retain-ratio must be in the range (0, 1].")
    if args.verify_kv_bootstrap_full_steps < 0:
        parser.error("--verify-kv-bootstrap-full-steps must be non-negative.")
    if args.verify_kv_scope_recent_size is not None and args.verify_kv_scope_recent_size < 0:
        parser.error("--verify-kv-scope-recent-size must be non-negative.")
    if args.verify_kv_scope_beta1 < 0:
        parser.error("--verify-kv-scope-beta1 must be non-negative.")
    if args.verify_kv_scope_beta2 < 0:
        parser.error("--verify-kv-scope-beta2 must be non-negative.")
    if args.verify_kv_scope_recent_size is not None:
        args.verify_kv_scope_beta2 = args.verify_kv_scope_recent_size
    if args.verify_kv_dynamic:
        if not args.verify_kv_compress:
            parser.error("--verify-kv-dynamic requires --verify-kv-compress.")
        if args.verify_kv_score_source != "scope":
            parser.error("--verify-kv-dynamic requires --verify-kv-score-source scope.")
        if args.verify_kv_dynamic_max_beta1 is None:
            args.verify_kv_dynamic_max_beta1 = args.verify_kv_scope_beta1
        if args.verify_kv_dynamic_min_beta1 < 0:
            parser.error("--verify-kv-dynamic-min-beta1 must be non-negative.")
        if args.verify_kv_dynamic_max_beta1 < args.verify_kv_dynamic_min_beta1:
            parser.error("--verify-kv-dynamic-max-beta1 must be >= --verify-kv-dynamic-min-beta1.")
        for name, value in [
            ("--verify-kv-dynamic-step", args.verify_kv_dynamic_step),
            ("--verify-kv-dynamic-window", args.verify_kv_dynamic_window),
            ("--verify-kv-dynamic-min-observations", args.verify_kv_dynamic_min_observations),
            ("--verify-kv-dynamic-patience", args.verify_kv_dynamic_patience),
        ]:
            if value <= 0:
                parser.error(f"{name} must be a positive integer.")
        if args.verify_kv_dynamic_cooldown < 0:
            parser.error("--verify-kv-dynamic-cooldown must be non-negative.")
        for name, value in [
            ("--verify-kv-dynamic-acceptance-floor", args.verify_kv_dynamic_acceptance_floor),
        ]:
            if value < 0.0 or value > 1.0:
                parser.error(f"{name} must be in the range [0, 1].")
        for name, value in [
            ("--verify-kv-dynamic-mean-floor", args.verify_kv_dynamic_mean_floor),
            ("--verify-kv-dynamic-confidence-floor", args.verify_kv_dynamic_confidence_floor),
            ("--verify-kv-dynamic-confidence-low", args.verify_kv_dynamic_confidence_low),
        ]:
            if value < 0.0:
                parser.error(f"{name} must be non-negative.")
        if args.verify_kv_dynamic_confidence_low > args.verify_kv_dynamic_confidence_floor:
            parser.error("--verify-kv-dynamic-confidence-low must be <= --verify-kv-dynamic-confidence-floor.")
    elif args.verify_kv_dynamic_max_beta1 is None:
        args.verify_kv_dynamic_max_beta1 = args.verify_kv_scope_beta1
    if args.retain_warmup_rounds <= 0:
        parser.error("--retain-warmup-rounds must be a positive integer.")
    if args.retain_filter_top_k <= 0:
        parser.error("--retain-filter-top-k must be a positive integer.")
    if args.retain_refine_rounds < 0:
        parser.error("--retain-refine-rounds must be non-negative.")
    if args.retain_final_tolerance < 0.0:
        parser.error("--retain-final-tolerance must be non-negative.")
    if args.final_layer_refine_rounds < 0:
        parser.error("--final-layer-refine-rounds must be non-negative.")
    if args.adaptive_window <= 0:
        parser.error("--adaptive-window must be a positive integer.")
    if args.adaptive_min_observations <= 0:
        parser.error("--adaptive-min-observations must be a positive integer.")
    if args.adaptive_std_k < 0.0:
        parser.error("--adaptive-std-k must be non-negative.")
    if args.adaptive_up_std_k is None:
        args.adaptive_up_std_k = args.adaptive_std_k
    if args.adaptive_down_std_k is None:
        args.adaptive_down_std_k = args.adaptive_std_k
    if args.adaptive_up_std_k < 0.0:
        parser.error("--adaptive-up-std-k must be non-negative.")
    if args.adaptive_down_std_k < 0.0:
        parser.error("--adaptive-down-std-k must be non-negative.")
    if args.adaptive_std_floor < 0.0:
        parser.error("--adaptive-std-floor must be non-negative.")
    if args.adaptive_patience <= 0:
        parser.error("--adaptive-patience must be a positive integer.")
    if args.adaptive_cooldown < 0:
        parser.error("--adaptive-cooldown must be non-negative.")
    if args.cosine_attn_alpha < -1.0 or args.cosine_attn_alpha > 1.0:
        parser.error("--cosine-attn-alpha must be in the range [-1, 1].")
    if args.cosine_max_skip_layers is not None and args.cosine_max_skip_layers < 0:
        parser.error("--cosine-max-skip-layers must be non-negative when set.")
    if args.cosine_keep_first_layers < 0:
        parser.error("--cosine-keep-first-layers must be non-negative.")
    if args.cosine_keep_last_layers < 0:
        parser.error("--cosine-keep-last-layers must be non-negative.")
    if args.cosine_mlp_interval < 0:
        parser.error("--cosine-mlp-interval must be non-negative.")
    if args.adaptive_layer_fallback_window <= 0:
        parser.error("--adaptive-layer-fallback-window must be a positive integer.")
    if args.adaptive_layer_improvement_delta < 0.0:
        parser.error("--adaptive-layer-improvement-delta must be non-negative.")
    if args.adaptive_min_retain_ratio <= 0.0 or args.adaptive_min_retain_ratio > 1.0:
        parser.error("--adaptive-min-retain-ratio must be in the range (0, 1].")
    if args.adaptive_ratio_step <= 0.0 or args.adaptive_ratio_step > 1.0:
        parser.error("--adaptive-ratio-step must be in the range (0, 1].")
    if args.adaptive_aggressive_tolerance < 0.0:
        parser.error("--adaptive-aggressive-tolerance must be non-negative.")
    if args.adaptive_aggressive_std_k < 0.0:
        parser.error("--adaptive-aggressive-std-k must be non-negative.")
    if args.adaptive_aggressive_patience <= 0:
        parser.error("--adaptive-aggressive-patience must be a positive integer.")
    if args.adaptive_max_extra_skip_layers is not None and args.adaptive_max_extra_skip_layers < 0:
        parser.error("--adaptive-max-extra-skip-layers must be non-negative when set.")
    if args.adaptive_max_skip_layers is not None and args.adaptive_max_skip_layers < 0:
        parser.error("--adaptive-max-skip-layers must be non-negative when set.")
    if args.final_target_mean_accepted <= 1.0:
        parser.error("--final-target-mean-accepted must be > 1.0.")
    if args.final_bad_mean_accepted <= 1.0:
        parser.error("--final-bad-mean-accepted must be > 1.0.")
    if args.final_severe_mean_accepted <= 1.0:
        parser.error("--final-severe-mean-accepted must be > 1.0.")
    if args.final_severe_mean_accepted > args.final_bad_mean_accepted:
        parser.error("--final-severe-mean-accepted must be <= --final-bad-mean-accepted.")
    if args.final_bad_mean_accepted > args.final_target_mean_accepted:
        parser.error("--final-bad-mean-accepted must be <= --final-target-mean-accepted.")
    for name, value in [
        ("--final-token-acceptance-floor", args.final_token_acceptance_floor),
        ("--final-more-skip-token-acceptance-floor", args.final_more_skip_token_acceptance_floor),
        ("--final-low-ratio-guard", args.final_low_ratio_guard),
        ("--final-min-ratio-for-more-skip", args.final_min_ratio_for_more_skip),
    ]:
        if value < 0.0 or value > 1.0:
            parser.error(f"{name} must be in the range [0, 1].")
    if args.final_more_skip_token_acceptance_floor < args.final_token_acceptance_floor:
        parser.error("--final-more-skip-token-acceptance-floor must be >= --final-token-acceptance-floor.")
    if args.final_hard_probe_token_acceptance_floor < 0.0 or args.final_hard_probe_token_acceptance_floor > 1.0:
        parser.error("--final-hard-probe-token-acceptance-floor must be in the range [0, 1].")
    for name, value in [
        ("--final-draft-len-floor", args.final_draft_len_floor),
        ("--final-more-skip-draft-len-floor", args.final_more_skip_draft_len_floor),
        ("--final-stable-mean-margin", args.final_stable_mean_margin),
        ("--final-hard-layer-mean-floor", args.final_hard_layer_mean_floor),
        ("--final-hard-probe-mean-margin", args.final_hard_probe_mean_margin),
        ("--final-hard-probe-token-acceptance-floor", args.final_hard_probe_token_acceptance_floor),
        ("--final-hard-probe-draft-len-margin", args.final_hard_probe_draft_len_margin),
        ("--final-ratio-down-gain-weight", args.final_ratio_down_gain_weight),
        ("--final-layer-skip-gain-weight", args.final_layer_skip_gain_weight),
    ]:
        if value < 0.0:
            parser.error(f"{name} must be non-negative.")
    if args.final_more_skip_draft_len_floor < args.final_draft_len_floor:
        parser.error("--final-more-skip-draft-len-floor must be >= --final-draft-len-floor.")
    if args.final_soft_max_skip_layers < 0 or args.final_hard_max_skip_layers < 0:
        parser.error("--final-soft-max-skip-layers and --final-hard-max-skip-layers must be non-negative.")
    if args.final_soft_max_skip_layers > args.final_hard_max_skip_layers:
        parser.error("--final-soft-max-skip-layers must be <= --final-hard-max-skip-layers.")
    if args.final_low_ratio_guard_skip_layers < 0:
        parser.error("--final-low-ratio-guard-skip-layers must be non-negative.")
    if args.final2_min_config_observations <= 0:
        parser.error("--final2-min-config-observations must be a positive integer.")
    if args.final2_layer_switch_cost is None:
        args.final2_layer_switch_cost = args.final2_switch_cost
    for name, value in [
        ("--final2-low-std-k", args.final2_low_std_k),
        ("--final2-high-std-k", args.final2_high_std_k),
        ("--final2-mean-std-floor", args.final2_mean_std_floor),
        ("--final2-prediction-beta", args.final2_prediction_beta),
        ("--final2-ratio-mean-slope", args.final2_ratio_mean_slope),
        ("--final2-layer-mean-slope", args.final2_layer_mean_slope),
        ("--final2-cold-start-penalty", args.final2_cold_start_penalty),
        ("--final2-draft-len-floor", args.final2_draft_len_floor),
        ("--final2-more-aggressive-draft-len-floor", args.final2_more_aggressive_draft_len_floor),
        ("--final2-switch-cost", args.final2_switch_cost),
        ("--final2-layer-switch-cost", args.final2_layer_switch_cost),
        ("--final2-ratio-down-gain-weight", args.final2_ratio_down_gain_weight),
        ("--final2-layer-skip-gain-weight", args.final2_layer_skip_gain_weight),
    ]:
        if value < 0.0:
            parser.error(f"{name} must be non-negative.")
    for name, value in [
        ("--final2-token-acceptance-floor", args.final2_token_acceptance_floor),
        ("--final2-more-aggressive-token-acceptance-floor", args.final2_more_aggressive_token_acceptance_floor),
    ]:
        if value < 0.0 or value > 1.0:
            parser.error(f"{name} must be in the range [0, 1].")
    if args.final2_more_aggressive_token_acceptance_floor < args.final2_token_acceptance_floor:
        parser.error("--final2-more-aggressive-token-acceptance-floor must be >= --final2-token-acceptance-floor.")
    if args.final2_more_aggressive_draft_len_floor < args.final2_draft_len_floor:
        parser.error("--final2-more-aggressive-draft-len-floor must be >= --final2-draft-len-floor.")
    for name, value in [
        ("--adaptive-cold-start-max-new-tokens", args.adaptive_cold_start_max_new_tokens),
        ("--adaptive-cold-start-max-steps", args.adaptive_cold_start_max_steps),
        ("--adaptive-cold-start-effective-count", args.adaptive_cold_start_effective_count),
        ("--adaptive-cold-start-max-configs", args.adaptive_cold_start_max_configs),
    ]:
        if value <= 0:
            parser.error(f"{name} must be a positive integer.")
    if args.adaptive_cold_start_skip_delta < 0:
        parser.error("--adaptive-cold-start-skip-delta must be non-negative.")
    try:
        _parse_optional_float_list(args.adaptive_cold_start_ratios)
        _parse_optional_int_list(args.adaptive_cold_start_skip_counts)
    except ValueError as exc:
        parser.error(f"Invalid adaptive cold-start list: {exc}")
    if args.lyapunov_acceptance_target > 1.0:
        parser.error("--lyapunov-acceptance-target must be <= 1.0; use <=0 to track the running reference acceptance.")
    if args.lyapunov_v < 0.0:
        parser.error("--lyapunov-v must be non-negative.")
    if args.lyapunov_switch_cost < 0.0:
        parser.error("--lyapunov-switch-cost must be non-negative.")
    if args.lyapunov_layer_switch_cost is None:
        args.lyapunov_layer_switch_cost = args.lyapunov_switch_cost
    if args.lyapunov_layer_switch_cost < 0.0:
        parser.error("--lyapunov-layer-switch-cost must be non-negative.")
    if args.lyapunov_layer_penalty_weight < 0.0:
        parser.error("--lyapunov-layer-penalty-weight must be non-negative.")
    if args.lyapunov_prediction_beta < 0.0:
        parser.error("--lyapunov-prediction-beta must be non-negative.")
    if args.lyapunov_ratio_acceptance_slope < 0.0:
        parser.error("--lyapunov-ratio-acceptance-slope must be non-negative.")
    if args.lyapunov_layer_acceptance_slope < 0.0:
        parser.error("--lyapunov-layer-acceptance-slope must be non-negative.")
    if args.lyapunov_cold_start_penalty < 0.0:
        parser.error("--lyapunov-cold-start-penalty must be non-negative.")
    try:
        args.retain_ratio_grid_values = parse_retain_ratio_grid(
            args.retain_ratio_grid,
            initial_ratio=args.draft_kv_retain_ratio if args.dynamic_retain_ratio else None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.retain_ratio_grid = format_retain_ratio_grid(args.retain_ratio_grid_values)
    if args.retain_target_score is None:
        args.retain_target_score = args.max_score
    if args.retain_target_score < 0.0 or args.retain_target_score > 1.0:
        parser.error("--retain-target-score must be in the range [0, 1].")
    if args.retain_utility_lambda < 0.0:
        parser.error("--retain-utility-lambda must be non-negative.")
    if args.retain_compression_weight < 0.0:
        parser.error("--retain-compression-weight must be non-negative.")
    if args.retain_score_tolerance < 0.0:
        parser.error("--retain-score-tolerance must be non-negative.")
    if args.retain_ucb_c < 0.0:
        parser.error("--retain-ucb-c must be non-negative.")
    if args.dynamic_retain_ratio and not args.optimization:
        parser.error("--dynamic-retain-ratio requires --optimization.")
    if args.dynamic_retain_ratio and args.draft_only:
        parser.error("--dynamic-retain-ratio is not supported with --draft-only.")
    if args.dynamic_retain_ratio and not args.draft_kv_compress:
        parser.error("--dynamic-retain-ratio requires --draft-kv-compress.")
    if args.dynamic_retain_ratio and not args.optimize_with_compressed_draft_kv:
        parser.error("--dynamic-retain-ratio requires --optimize-with-compressed-draft-kv.")
    if args.dynamic_retain_ratio and (args.load_skip_layer_cache or args.cache_hit):
        parser.error("--dynamic-retain-ratio cannot be combined with cached skip-layer loading.")
    if args.cosine_prefill_skip_layers and args.dynamic_retain_ratio:
        parser.error("--cosine-prefill-skip-layers is intended to replace dynamic retain-ratio search in this path.")
    if args.cosine_prefill_skip_layers and args.draft_only:
        parser.error("--cosine-prefill-skip-layers is not supported with --draft-only.")
    if args.cosine_prefill_skip_layers and args.optimization:
        parser.error("--cosine-prefill-skip-layers cannot be combined with --optimization trials.")
    if args.adaptive_layer_controller and not args.local_adaptive_controller:
        parser.error("--adaptive-layer-controller requires --local-adaptive-controller.")
    if args.adaptive_layer_controller and not args.cosine_prefill_skip_layers:
        parser.error("--adaptive-layer-controller requires --cosine-prefill-skip-layers.")
    if args.adaptive_aggressive_controller and not args.local_adaptive_controller:
        parser.error("--adaptive-aggressive-controller requires --local-adaptive-controller.")
    if args.adaptive_final_controller and not args.local_adaptive_controller:
        parser.error("--adaptive-final-controller requires --local-adaptive-controller.")
    if args.adaptive_final_controller and not args.cosine_prefill_skip_layers:
        parser.error("--adaptive-final-controller requires --cosine-prefill-skip-layers.")
    if args.adaptive_final2_controller and not args.local_adaptive_controller:
        parser.error("--adaptive-final2-controller requires --local-adaptive-controller.")
    if args.adaptive_final2_controller and not args.cosine_prefill_skip_layers:
        parser.error("--adaptive-final2-controller requires --cosine-prefill-skip-layers.")
    if args.adaptive_cold_start and not args.local_adaptive_controller:
        parser.error("--adaptive-cold-start requires --local-adaptive-controller.")
    if args.adaptive_cold_start and not args.adaptive_final2_controller:
        parser.error("--adaptive-cold-start is intended for the final2 adaptive table and requires --adaptive-final2-controller.")
    if args.adaptive_cold_start and not args.cosine_prefill_skip_layers:
        parser.error("--adaptive-cold-start requires --cosine-prefill-skip-layers.")
    if args.lyapunov_adaptive_controller and not args.local_adaptive_controller:
        parser.error("--lyapunov-adaptive-controller requires --local-adaptive-controller.")
    enabled_policy_count = sum([
        bool(args.lyapunov_adaptive_controller),
        bool(args.adaptive_aggressive_controller),
        bool(args.adaptive_final_controller),
        bool(args.adaptive_final2_controller),
    ])
    if enabled_policy_count > 1:
        parser.error("--adaptive-final2-controller, --adaptive-final-controller, --adaptive-aggressive-controller, and --lyapunov-adaptive-controller are separate modes and cannot be combined.")
    if args.save_skip_layer_cache and args.load_skip_layer_cache:
        parser.error("--save-skip-layer-cache and --load-skip-layer-cache are mutually exclusive.")
    if args.load_selected_swift_config:
        if args.dynamic_retain_ratio:
            parser.error("--load-selected-swift-config cannot be combined with --dynamic-retain-ratio.")
        if args.load_skip_layer_cache or args.cache_hit or args.save_skip_layer_cache:
            parser.error("--load-selected-swift-config cannot be combined with other skip-layer cache modes.")
        if not args.draft_kv_compress:
            parser.error("--load-selected-swift-config requires --draft-kv-compress.")
    if args.local_adaptive_controller:
        if args.draft_only:
            parser.error("--local-adaptive-controller is not supported with --draft-only.")
        if not args.draft_kv_compress:
            parser.error("--local-adaptive-controller requires --draft-kv-compress.")
        if not (args.dynamic_retain_ratio or args.load_selected_swift_config or args.cosine_prefill_skip_layers):
            parser.error("--local-adaptive-controller requires dynamic retain-ratio, selected config, or cosine prefill skip layers.")

    selected_swift_config = None
    if args.load_selected_swift_config:
        selected_swift_config = get_selected_swift_config(
            args.selected_swift_config_file,
            args.model_id,
            args.task_name,
        )
        if selected_swift_config is None:
            raise FileNotFoundError(
                f"Selected SWIFT config not found for task '{args.task_name}' "
                f"in {args.selected_swift_config_file}."
            )
        args.draft_kv_retain_ratio = float(selected_swift_config["draft_kv_retain_ratio"])
        args.skip_ratio = float(selected_swift_config.get("skip_ratio", args.skip_ratio))
        args.optimization, args.bayes = False, False

    try:
        args.adaptive_ratio_ladder_values = parse_retain_ratio_grid(
            args.adaptive_ratio_ladder,
            initial_ratio=args.draft_kv_retain_ratio if args.local_adaptive_controller else None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if (
        args.adaptive_aggressive_controller
        or args.lyapunov_adaptive_controller
        or args.adaptive_final_controller
        or args.adaptive_final2_controller
    ):
        args.adaptive_ratio_ladder_values = extend_aggressive_adaptive_ratio_ladder(
            args.adaptive_ratio_ladder_values,
            min_ratio=args.adaptive_min_retain_ratio,
            step=args.adaptive_ratio_step,
        )
    args.adaptive_ratio_ladder = format_retain_ratio_grid(args.adaptive_ratio_ladder_values)

    args.skip_layer_cache_key = args.skip_layer_cache_key or build_skip_layer_cache_key(args)

    retain_ratio_name = retain_ratio_run_name(args)
    cosine_name_suffix = ""
    kv_score_name_suffix = "-kvsrc-" + args.draft_kv_score_source if args.draft_kv_compress else ""
    verify_kv_name_suffix = ""
    if args.verify_kv_compress:
        verify_score_tags = {
            "semantic": "sem",
            "scope": "s",
            "observation": "obs",
            "heuristic": "heu",
            "reuse": "reuse",
        }
        verify_kv_name_suffix = "-v" + verify_score_tags.get(
            args.verify_kv_score_source,
            args.verify_kv_score_source,
        )
        if args.verify_kv_score_source == "scope":
            verify_kv_name_suffix += str(args.verify_kv_scope_beta1) + "x" + str(args.verify_kv_scope_beta2)
            if args.verify_kv_dynamic:
                verify_kv_name_suffix += "-d" + str(args.verify_kv_dynamic_min_beta1)
        elif args.verify_kv_retain_ratio is not None:
            verify_kv_name_suffix += "-r" + str(args.verify_kv_retain_ratio)
        if args.verify_kv_bootstrap_full_steps > 0:
            verify_kv_name_suffix += "-b" + str(args.verify_kv_bootstrap_full_steps)
        if args.verify_kv_safe_commit:
            verify_kv_name_suffix += "-sc"
    args.model_name = (args.model_id + "-swift-" + str(args.dtype)+ "-temp-" + str(args.temperature)
                       + "-top-p-" + str(args.top_p) + "-seed-" + str(args.seed) + "-max_new_tokens-" + str(args.max_new_tokens)+ "-opt_interval-" + str(args.opt_interval)
                    #    + "-bayes_interval-" + str(args.bayes_interval) + "-max_opt-" + str(args.max_opt_iter) + "-max_tolerance-" + str(args.max_tolerance_iter)
                       + "-max_score-" + str(args.max_score) + "-context_window-" + str(args.context_window) + "-skip_ratio-" + str(args.skip_ratio) + "-draft_kv_retain_ratio-" + retain_ratio_name
                       + "-opt_compressed_draft_kv-" + str(args.optimize_with_compressed_draft_kv)
                       + cosine_name_suffix
                       + kv_score_name_suffix
                       + verify_kv_name_suffix
                       + ("-draft_token_num-" + str(args.draft_token_num) if args.draft_token_num is not None else "")
                       + ("-draft_only" if args.draft_only else ""))
    answer_file = args.answer_file or f"outputs/{args.task_name}/{args.task_name}_{args.data_num}/without_layerskip_draft_model_answer/{args.model_id}/{args.model_name}.jsonl"
    set_logger()

    print(f"Output to {answer_file}")

    torch.nn.Linear.reset_parameters = lambda x: None

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map={"": 0})

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature, top_p=args.top_p)
    else:
        logits_processor = None

    if args.load_selected_swift_config:
        _attn_skip_layer_id_set = selected_swift_config["attention"]
        _mlp_skip_layer_id_set = selected_swift_config["mlp"]
        args.optimization, args.bayes = False, False
    elif args.load_skip_layer_cache:
        cached_skip_layers = get_skip_layer_cache(args.skip_layer_cache_file, args.skip_layer_cache_key)
        if cached_skip_layers is None:
            raise FileNotFoundError(
                f"Skip-layer cache not found for key '{args.skip_layer_cache_key}' "
                f"in {args.skip_layer_cache_file}."
            )

        _attn_skip_layer_id_set, _mlp_skip_layer_id_set = cached_skip_layers
        args.optimization, args.bayes = False, False
    elif args.cache_hit:
        # Load the cached layer set configuration
        args.optimization, args.bayes=False, False
        _attn_skip_layer_id_set, _mlp_skip_layer_id_set = get_cache_configuration(model_name=args.model_id,
                                                                                  task_name=args.task_name)
    else:
        # Unified layer set initialization
        # with layer skip
        _attn_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)  # keep the first and last layer
        _mlp_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)
        # without layer skip
        # _attn_skip_layer_id_set = []  
        # _mlp_skip_layer_id_set = []

    model.set_skip_layers(_attn_skip_layer_id_set, _mlp_skip_layer_id_set)

    # Bayes Optimization Settings
    if args.dynamic_retain_ratio:
        optimizer = {
            retain_ratio: build_layer_optimizer(model.config.num_hidden_layers, random_state=idx + 1)
            for idx, retain_ratio in enumerate(args.retain_ratio_grid_values)
        }
    else:
        optimizer = build_layer_optimizer(model.config.num_hidden_layers, random_state=1)
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    statistics = {"origin_score": 0, "opt_iter": 0, "tolerance_iter": 0,
                  "task_name": args.task_name,
                  "skip_ratio": args.skip_ratio, "acceptance_rate_list": [], "opt_interval": args.opt_interval,
                  "bayes_interval": args.bayes_interval, "max_opt_iter": args.max_opt_iter,
                  "max_tolerance_iter": args.max_tolerance_iter, "max_score": args.max_score,
                  "context_window": args.context_window, "optimization": args.optimization, "bayes": args.bayes,
                  "draft_kv_compress": args.draft_kv_compress, "draft_kv_retain_ratio": args.draft_kv_retain_ratio,
                  "draft_kv_cache_mode": args.draft_kv_cache_mode,
                  "draft_kv_score_source": args.draft_kv_score_source,
                  "draft_kv_reuse_ema": args.draft_kv_reuse_ema,
                  "verify_kv_compress": args.verify_kv_compress,
                  "verify_kv_retain_ratio": args.verify_kv_retain_ratio,
                  "verify_kv_cache_mode": args.verify_kv_cache_mode,
                  "verify_kv_score_source": args.verify_kv_score_source,
                  "verify_kv_bootstrap_full_steps": args.verify_kv_bootstrap_full_steps,
                  "verify_kv_scope_recent_size": args.verify_kv_scope_beta2,
                  "verify_kv_scope_beta1": args.verify_kv_scope_beta1,
                  "verify_kv_scope_beta2": args.verify_kv_scope_beta2,
                  "verify_kv_scope_score_full_only": True,
                  "verify_kv_safe_commit": args.verify_kv_safe_commit,
                  "verify_kv_dynamic": args.verify_kv_dynamic,
                  "verify_kv_dynamic_initial_beta1": args.verify_kv_scope_beta1,
                  "verify_kv_dynamic_current_beta1": args.verify_kv_scope_beta1,
                  "verify_kv_dynamic_min_beta1": args.verify_kv_dynamic_min_beta1,
                  "verify_kv_dynamic_max_beta1": args.verify_kv_dynamic_max_beta1,
                  "verify_kv_dynamic_step": args.verify_kv_dynamic_step,
                  "verify_kv_dynamic_window": args.verify_kv_dynamic_window,
                  "verify_kv_dynamic_min_observations": args.verify_kv_dynamic_min_observations,
                  "verify_kv_dynamic_acceptance_floor": args.verify_kv_dynamic_acceptance_floor,
                  "verify_kv_dynamic_mean_floor": args.verify_kv_dynamic_mean_floor,
                  "verify_kv_dynamic_confidence_floor": args.verify_kv_dynamic_confidence_floor,
                  "verify_kv_dynamic_confidence_low": args.verify_kv_dynamic_confidence_low,
                  "verify_kv_dynamic_patience": args.verify_kv_dynamic_patience,
                  "verify_kv_dynamic_cooldown": args.verify_kv_dynamic_cooldown,
                  "optimize_with_compressed_draft_kv": args.optimize_with_compressed_draft_kv,
                  "draft_token_num": args.draft_token_num,
                  "log_draft_tokens": args.log_draft_tokens,
                  "flops_trace_enabled": args.flops_trace,
                  "flops_trace_file": args.flops_trace_file,
                  "flops_trace_phase": "eval",
                  "cosine_prefill_skip_layers": args.cosine_prefill_skip_layers,
                  "cosine_skip_mode": args.cosine_skip_mode,
                  "cosine_attn_alpha": args.cosine_attn_alpha,
                  "cosine_max_skip_layers": args.cosine_max_skip_layers,
                  "cosine_keep_first_layers": args.cosine_keep_first_layers,
                  "cosine_keep_last_layers": args.cosine_keep_last_layers,
                  "cosine_mlp_interval": args.cosine_mlp_interval,
                  "adaptive_layer_controller": args.adaptive_layer_controller,
                  "adaptive_layer_fallback_window": args.adaptive_layer_fallback_window,
                  "adaptive_layer_improvement_delta": args.adaptive_layer_improvement_delta,
                  "adaptive_aggressive_controller": args.adaptive_aggressive_controller,
                  "adaptive_min_retain_ratio": args.adaptive_min_retain_ratio,
                  "adaptive_ratio_step": args.adaptive_ratio_step,
                  "adaptive_aggressive_tolerance": args.adaptive_aggressive_tolerance,
                  "adaptive_aggressive_std_k": args.adaptive_aggressive_std_k,
                  "adaptive_aggressive_patience": args.adaptive_aggressive_patience,
                  "adaptive_max_extra_skip_layers": args.adaptive_max_extra_skip_layers,
                  "adaptive_max_skip_layers": args.adaptive_max_skip_layers,
                  "adaptive_final_controller": args.adaptive_final_controller,
                  "final_target_mean_accepted": args.final_target_mean_accepted,
                  "final_bad_mean_accepted": args.final_bad_mean_accepted,
                  "final_severe_mean_accepted": args.final_severe_mean_accepted,
                  "final_token_acceptance_floor": args.final_token_acceptance_floor,
                  "final_more_skip_token_acceptance_floor": args.final_more_skip_token_acceptance_floor,
                  "final_draft_len_floor": args.final_draft_len_floor,
                  "final_more_skip_draft_len_floor": args.final_more_skip_draft_len_floor,
                  "final_stable_mean_margin": args.final_stable_mean_margin,
                  "final_soft_max_skip_layers": args.final_soft_max_skip_layers,
                  "final_hard_max_skip_layers": args.final_hard_max_skip_layers,
                  "final_min_ratio_for_more_skip": args.final_min_ratio_for_more_skip,
                  "final_low_ratio_guard": args.final_low_ratio_guard,
                  "final_low_ratio_guard_skip_layers": args.final_low_ratio_guard_skip_layers,
                  "final_hard_layer_mean_floor": args.final_hard_layer_mean_floor,
                  "final_hard_probe_mean_margin": args.final_hard_probe_mean_margin,
                  "final_hard_probe_token_acceptance_floor": args.final_hard_probe_token_acceptance_floor,
                  "final_hard_probe_draft_len_margin": args.final_hard_probe_draft_len_margin,
                  "final_ratio_down_gain_weight": args.final_ratio_down_gain_weight,
                  "final_layer_skip_gain_weight": args.final_layer_skip_gain_weight,
                  "adaptive_final2_controller": args.adaptive_final2_controller,
                  "final2_low_std_k": args.final2_low_std_k,
                  "final2_high_std_k": args.final2_high_std_k,
                  "final2_mean_std_floor": args.final2_mean_std_floor,
                  "final2_min_config_observations": args.final2_min_config_observations,
                  "final2_prediction_beta": args.final2_prediction_beta,
                  "final2_ratio_mean_slope": args.final2_ratio_mean_slope,
                  "final2_layer_mean_slope": args.final2_layer_mean_slope,
                  "final2_cold_start_penalty": args.final2_cold_start_penalty,
                  "final2_token_acceptance_floor": args.final2_token_acceptance_floor,
                  "final2_more_aggressive_token_acceptance_floor": args.final2_more_aggressive_token_acceptance_floor,
                  "final2_draft_len_floor": args.final2_draft_len_floor,
                  "final2_more_aggressive_draft_len_floor": args.final2_more_aggressive_draft_len_floor,
                  "final2_switch_cost": args.final2_switch_cost,
                  "final2_layer_switch_cost": args.final2_layer_switch_cost,
                  "final2_ratio_down_gain_weight": args.final2_ratio_down_gain_weight,
                  "final2_layer_skip_gain_weight": args.final2_layer_skip_gain_weight,
                  "adaptive_cold_start": args.adaptive_cold_start,
                  "adaptive_cold_start_mode": args.adaptive_cold_start_mode,
                  "adaptive_cold_start_prompt": args.adaptive_cold_start_prompt,
                  "adaptive_cold_start_max_new_tokens": args.adaptive_cold_start_max_new_tokens,
                  "adaptive_cold_start_max_steps": args.adaptive_cold_start_max_steps,
                  "adaptive_cold_start_effective_count": args.adaptive_cold_start_effective_count,
                  "adaptive_cold_start_max_configs": args.adaptive_cold_start_max_configs,
                  "adaptive_cold_start_ratios": args.adaptive_cold_start_ratios,
                  "adaptive_cold_start_skip_counts": args.adaptive_cold_start_skip_counts,
                  "adaptive_cold_start_skip_delta": args.adaptive_cold_start_skip_delta,
                  "lyapunov_adaptive_controller": args.lyapunov_adaptive_controller,
                  "lyapunov_acceptance_target": args.lyapunov_acceptance_target,
                  "lyapunov_v": args.lyapunov_v,
                  "lyapunov_switch_cost": args.lyapunov_switch_cost,
                  "lyapunov_layer_switch_cost": args.lyapunov_layer_switch_cost,
                  "lyapunov_layer_penalty_weight": args.lyapunov_layer_penalty_weight,
                  "lyapunov_prediction_beta": args.lyapunov_prediction_beta,
                  "lyapunov_ratio_acceptance_slope": args.lyapunov_ratio_acceptance_slope,
                  "lyapunov_layer_acceptance_slope": args.lyapunov_layer_acceptance_slope,
                  "lyapunov_cold_start_penalty": args.lyapunov_cold_start_penalty,
                  "dynamic_retain_ratio": args.dynamic_retain_ratio,
                  "retain_ratio_grid": args.retain_ratio_grid_values,
                  "retain_target_score": args.retain_target_score,
                  "retain_utility_mode": args.retain_utility_mode,
                  "retain_compression_weight": args.retain_compression_weight,
                  "retain_score_tolerance": args.retain_score_tolerance,
                  "retain_utility_lambda": args.retain_utility_lambda,
                  "retain_ucb_c": args.retain_ucb_c,
                  "retain_warmup_rounds": args.retain_warmup_rounds,
                  "retain_filter_top_k": args.retain_filter_top_k,
                  "retain_refine_rounds": args.retain_refine_rounds,
                  "retain_final_tolerance": args.retain_final_tolerance,
                  "final_layer_refine_rounds": args.final_layer_refine_rounds,
                  "local_adaptive_controller": args.local_adaptive_controller,
                  "adaptive_initial_retain_ratio": args.draft_kv_retain_ratio,
                  "adaptive_ratio_ladder": args.adaptive_ratio_ladder_values,
                  "adaptive_window": args.adaptive_window,
                  "adaptive_min_observations": args.adaptive_min_observations,
                  "adaptive_reference_mode": "global",
                  "adaptive_std_k": args.adaptive_std_k,
                  "adaptive_up_std_k": args.adaptive_up_std_k,
                  "adaptive_down_std_k": args.adaptive_down_std_k,
                  "adaptive_std_floor": args.adaptive_std_floor,
                  "adaptive_patience": args.adaptive_patience,
                  "adaptive_cooldown": args.adaptive_cooldown}

    _initialize_flops_trace_file(statistics, model, args, answer_file)
    if statistics.get("flops_trace_enabled", False):
        print(f"FLOPs trace to {statistics.get('flops_trace_file')}")

    # Run dynamic-6 warmup once before the benchmark stream, never per question.
    run_adaptive_cold_start(model, tokenizer, statistics, logits_processor=logits_processor)

    forward_f = draft_only_forward if args.draft_only else swift_forward
    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=forward_f,
        model_id=args.model_id,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        task_name=args.task_name,
        data_num=args.data_num,
        seed=args.seed,
        optimizer=optimizer,
        utility=utility,
        statistics=statistics,
        logits_processor=logits_processor,
    )

    if args.save_skip_layer_cache:
        best_attn_skip_layer_id_set, best_mlp_skip_layer_id_set = model.get_skip_layers()
        save_skip_layer_cache(
            args.skip_layer_cache_file,
            args.skip_layer_cache_key,
            best_attn_skip_layer_id_set,
            best_mlp_skip_layer_id_set,
            metadata={
                "model_id": args.model_id,
                "task_name": args.task_name,
                "data_num": args.data_num,
                "seed": args.seed,
                "dtype": args.dtype,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "opt_interval": args.opt_interval,
                "bayes_interval": args.bayes_interval,
                "max_opt_iter": args.max_opt_iter,
                "max_tolerance_iter": args.max_tolerance_iter,
                "max_score": args.max_score,
                "context_window": args.context_window,
                "skip_ratio": args.skip_ratio,
                "draft_token_num": args.draft_token_num,
                "draft_kv_retain_ratio": statistics.get("draft_kv_retain_ratio", args.draft_kv_retain_ratio),
                "dynamic_retain_ratio": args.dynamic_retain_ratio,
                "retain_ratio_grid": args.retain_ratio_grid_values,
                "retain_target_score": args.retain_target_score,
                "retain_utility_mode": args.retain_utility_mode,
                "retain_compression_weight": args.retain_compression_weight,
                "retain_score_tolerance": args.retain_score_tolerance,
                "retain_utility_lambda": args.retain_utility_lambda,
                "retain_ucb_c": args.retain_ucb_c,
                "retain_warmup_rounds": args.retain_warmup_rounds,
                "retain_filter_top_k": args.retain_filter_top_k,
                "retain_refine_rounds": args.retain_refine_rounds,
                "retain_final_tolerance": args.retain_final_tolerance,
                "final_layer_refine_rounds": args.final_layer_refine_rounds,
                "retain_stage": statistics.get("retain_stage"),
                "retain_candidate_ratios": statistics.get("retain_candidate_ratios"),
                "retain_final_ratio": statistics.get("retain_final_ratio"),
                "retain_ratio_state": statistics.get("retain_ratio_state", {}),
                "best_retain_ratio": statistics.get("best_retain_ratio"),
                "best_retain_score": statistics.get("best_retain_score"),
                "best_retain_utility": statistics.get("best_retain_utility"),
                "optimize_with_compressed_draft_kv": args.optimize_with_compressed_draft_kv,
                "origin_score": statistics["origin_score"],
                "opt_iter": statistics["opt_iter"],
            },
        )
