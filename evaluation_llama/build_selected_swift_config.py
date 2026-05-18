"""Build one selected SWIFT config per benchmark from fixed-ratio result files."""

import argparse
import json
import os
import re
from pathlib import Path


RATIO_RE = re.compile(r"draft_kv_retain_ratio-([0-9.]+)-opt_compressed_draft_kv-True")


def _parse_tasks(value):
    if value is None:
        return None
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    return set(tasks) if tasks else None


def _read_summary(path):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return None

    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "Token acceptance rate" in record:
            return record
    return None


def _candidate_from_path(path):
    ratio_match = RATIO_RE.search(path.name)
    if ratio_match is None:
        return None

    parts = path.parts
    try:
        output_index = parts.index("outputs")
    except ValueError:
        output_index = 0

    if len(parts) <= output_index + 2:
        return None

    task_name = parts[output_index + 1]
    data_dir = parts[output_index + 2]
    data_prefix = f"{task_name}_"
    if not data_dir.startswith(data_prefix):
        return None

    try:
        data_num = int(data_dir[len(data_prefix):])
    except ValueError:
        return None

    model_id = path.parent.name
    summary = _read_summary(path)
    if summary is None:
        return None

    if "Best Attn Layer Set" not in summary or "Best MLP Layer Set" not in summary:
        return None

    retain_ratio = float(ratio_match.group(1))
    return {
        "task_name": task_name,
        "data_num": data_num,
        "model_id": model_id,
        "draft_kv_retain_ratio": retain_ratio,
        "skip_ratio": float(summary.get("Best Skip Ratio", 0.0)),
        "attention": [int(x) for x in summary["Best Attn Layer Set"]],
        "mlp": [int(x) for x in summary["Best MLP Layer Set"]],
        "mean_accepted_tokens": float(summary.get("Mean accepted tokens", 0.0)),
        "token_acceptance_rate": float(summary.get("Token acceptance rate", 0.0)),
        "source_file": str(path),
    }


def _candidate_sort_key(metric):
    def key(candidate):
        primary = float(candidate.get(metric, 0.0))
        secondary = float(candidate.get("mean_accepted_tokens", 0.0))
        return (-primary, -secondary, float(candidate["draft_kv_retain_ratio"]))

    return key


def _compact_candidate(candidate, metric):
    return {
        "draft_kv_retain_ratio": candidate["draft_kv_retain_ratio"],
        metric: candidate[metric],
        "mean_accepted_tokens": candidate["mean_accepted_tokens"],
        "source_file": candidate["source_file"],
    }


def _select_candidate(candidates, metric):
    ranked = sorted(candidates, key=_candidate_sort_key(metric))
    if len(ranked) < 2:
        return ranked[0], ranked, 1

    best = ranked[0]
    second = ranked[1]
    if second["draft_kv_retain_ratio"] < best["draft_kv_retain_ratio"]:
        return second, ranked, 2
    return best, ranked, 1


def _group_candidates(candidates, model_id, tasks, data_num):
    grouped = {}
    for candidate in candidates:
        if model_id is not None and candidate["model_id"] != model_id:
            continue
        if tasks is not None and candidate["task_name"] not in tasks:
            continue
        if data_num is not None and candidate["data_num"] != data_num:
            continue
        key = (candidate["task_name"], candidate["data_num"])
        grouped.setdefault(key, []).append(candidate)
    return grouped


def _choose_data_group(grouped):
    by_task = {}
    for (task_name, data_num), candidates in grouped.items():
        current = by_task.get(task_name)
        if current is None or data_num > current[0]:
            by_task[task_name] = (data_num, candidates)
    return by_task


def build_config(args):
    output_root = Path(args.outputs_dir)
    tasks = _parse_tasks(args.tasks)
    candidates = []
    for path in output_root.rglob("*.jsonl"):
        if "opt_compressed_draft_kv-True" not in path.name:
            continue
        candidate = _candidate_from_path(path)
        if candidate is not None:
            candidates.append(candidate)

    grouped = _group_candidates(candidates, args.model_id, tasks, args.data_num)
    selected_groups = _choose_data_group(grouped)

    config = {
        "metadata": {
            "created_by": "evaluation_llama/build_selected_swift_config.py",
            "outputs_dir": args.outputs_dir,
            "metric": args.metric,
            "selection_rule": (
                "Rank fixed-ratio opt_compressed_draft_kv=True runs by metric. "
                "If the second-best retain ratio is lower than the best retain ratio, "
                "select the second-best candidate; otherwise select the best candidate."
            ),
        },
        "benchmarks": {},
    }

    for task_name in sorted(selected_groups):
        _, task_candidates = selected_groups[task_name]
        selected, ranked, selected_rank = _select_candidate(task_candidates, args.metric)
        entry = {
            "model_id": selected["model_id"],
            "task_name": selected["task_name"],
            "data_num": selected["data_num"],
            "draft_kv_retain_ratio": selected["draft_kv_retain_ratio"],
            "skip_ratio": selected["skip_ratio"],
            "attention": selected["attention"],
            "mlp": selected["mlp"],
            "token_acceptance_rate": selected["token_acceptance_rate"],
            "mean_accepted_tokens": selected["mean_accepted_tokens"],
            "selected_rank": selected_rank,
            "source_file": selected["source_file"],
            "best_candidate": _compact_candidate(ranked[0], args.metric),
            "second_candidate": _compact_candidate(ranked[1], args.metric) if len(ranked) > 1 else None,
        }
        config["benchmarks"][task_name] = entry

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output-file", default="outputs/selected_swift_config.json")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated task list. Defaults to all tasks found.")
    parser.add_argument("--data-num", type=int, default=None, help="Only use this data_num. Defaults to largest per task.")
    parser.add_argument(
        "--metric",
        choices=["token_acceptance_rate", "mean_accepted_tokens"],
        default="token_acceptance_rate",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    config = build_config(args)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    output_file = Path(args.output_file)
    output_dir = output_file.parent
    if str(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Saved selected SWIFT config to {output_file}.")


if __name__ == "__main__":
    main()
