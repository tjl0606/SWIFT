# 0.99 with layer skip
# python evaluation_llama/eval_rouge.py \
#     --answer-file /home/tjlin/KV_SSD/SWIFT/outputs/cnndm/cnndm_100/draft_model_answer/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-swift-float16-temp-0.0-top-p-0.85-seed-2024-max_new_tokens-512-opt_interval-1-bayes_interval-25-max_opt-1000-max_tolerance-300-max_score-0.93-context_window-50-skip_ratio-0.45-draft_only.jsonl \
#     --seed 2024 \
#     --data-num 100

# 0.99 without layer skip
# python evaluation_llama/eval_rouge.py \
#     --answer-file /home/tjlin/KV_SSD/SWIFT/outputs/cnndm/cnndm_100/without_layerskip_draft_model_answer/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-swift-float16-temp-0.0-top-p-0.85-seed-2024-max_new_tokens-512-opt_interval-1-bayes_interval-25-max_opt-1000-max_tolerance-300-max_score-0.93-context_window-50-skip_ratio-0.45-draft_kv_retain_ratio-1.0-draft_only.jsonl \
#     --seed 2024 \
#     --data-num 100

# original llama answer
python evaluation_llama/eval_rouge.py \
    --answer-file /home/tjlin/KV_SSD/SWIFT/test/cnndm/cnndm_100/model_answer/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-vanilla-float16-temp-0.0-top-p-0.85-seed-2024-max_new_tokens-512.jsonl \
    --seed 2024 \
    --data-num 100