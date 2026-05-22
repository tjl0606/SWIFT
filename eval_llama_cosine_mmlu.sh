#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/home/tjlin/models/Meta-Llama-3-8B-Instruct}
MODEL_NAME=${MODEL_NAME:-Meta-Llama-3-8B-Instruct}
TEMP=${TEMP:-0.0}
TOP_P=${TOP_P:-1.0}
SEED=${SEED:-2024}
GPU_DEVICES=${GPU_DEVICES:-0}
torch_dtype=${torch_dtype:-bfloat16}

TASK_NAME=${TASK_NAME:-mmlu}
DATA_NUM=${DATA_NUM:-1000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
CONTEXT_WINDOW=${CONTEXT_WINDOW:-50}
OPT_INTERVAL=${OPT_INTERVAL:-1}
BAYES_INTERVAL=${BAYES_INTERVAL:-25}
MAX_OPT_ITER=${MAX_OPT_ITER:-1100}
MAX_TOLERANCE_ITER=${MAX_TOLERANCE_ITER:-300}
MAX_SCORE=${MAX_SCORE:-0.93}
SKIP_RATIO=${SKIP_RATIO:-0.45}
DRAFT_TOKEN_NUM=${DRAFT_TOKEN_NUM-}

case "${TASK_NAME}" in
  mmlu)
    DEFAULT_RETAIN_RATIO=0.6
    ;;
  gsm8k)
    DEFAULT_RETAIN_RATIO=0.7
    ;;
  *)
    echo "This cosine-prefill script currently supports TASK_NAME=mmlu or TASK_NAME=gsm8k." >&2
    exit 1
    ;;
esac

RETAIN_RATIO=${RETAIN_RATIO:-${DEFAULT_RETAIN_RATIO}}
ADAPTIVE_RATIO_LADDER=${ADAPTIVE_RATIO_LADDER:-0.6,0.7,0.8,0.9,1.0}
ADAPTIVE_WINDOW=${ADAPTIVE_WINDOW:-16}
ADAPTIVE_MIN_OBSERVATIONS=${ADAPTIVE_MIN_OBSERVATIONS:-24}
ADAPTIVE_STD_K=${ADAPTIVE_STD_K:-0.5}
ADAPTIVE_UP_STD_K=${ADAPTIVE_UP_STD_K:-${ADAPTIVE_STD_K}}
ADAPTIVE_DOWN_STD_K=${ADAPTIVE_DOWN_STD_K:-1.0}
ADAPTIVE_STD_FLOOR=${ADAPTIVE_STD_FLOOR:-0.05}
ADAPTIVE_PATIENCE=${ADAPTIVE_PATIENCE:-1}
ADAPTIVE_COOLDOWN=${ADAPTIVE_COOLDOWN:-8}

COSINE_SKIP_MODE=${COSINE_SKIP_MODE:-topk}
COSINE_ATTN_ALPHA=${COSINE_ATTN_ALPHA:-0.985}
COSINE_MAX_SKIP_LAYERS=${COSINE_MAX_SKIP_LAYERS-}
COSINE_KEEP_FIRST_LAYERS=${COSINE_KEEP_FIRST_LAYERS:-1}
COSINE_KEEP_LAST_LAYERS=${COSINE_KEEP_LAST_LAYERS:-2}
COSINE_MLP_INTERVAL=${COSINE_MLP_INTERVAL:-0}
LOCAL_ADAPTIVE_CONTROLLER=${LOCAL_ADAPTIVE_CONTROLLER:-1}
ADAPTIVE_LAYER_CONTROLLER=${ADAPTIVE_LAYER_CONTROLLER:-1}
ADAPTIVE_LAYER_FALLBACK_WINDOW=${ADAPTIVE_LAYER_FALLBACK_WINDOW:-16}
ADAPTIVE_LAYER_IMPROVEMENT_DELTA=${ADAPTIVE_LAYER_IMPROVEMENT_DELTA:-0.0}

case "${ADAPTIVE_LAYER_CONTROLLER,,}" in
  1|true|yes|on)
    ADAPTIVE_LAYER_ARG="--adaptive-layer-controller --adaptive-layer-fallback-window ${ADAPTIVE_LAYER_FALLBACK_WINDOW} --adaptive-layer-improvement-delta ${ADAPTIVE_LAYER_IMPROVEMENT_DELTA}"
    ADAPTIVE_LAYER_SUFFIX=""
    ADAPTIVE_LAYER_STATUS="enabled"
    ;;
  0|false|no|off)
    ADAPTIVE_LAYER_ARG=""
    ADAPTIVE_LAYER_SUFFIX="-no-layer-fallback"
    ADAPTIVE_LAYER_STATUS="disabled"
    ;;
  *)
    echo "ADAPTIVE_LAYER_CONTROLLER must be one of 1/0, true/false, yes/no, on/off." >&2
    exit 1
    ;;
esac

case "${LOCAL_ADAPTIVE_CONTROLLER,,}" in
  1|true|yes|on)
    LOCAL_ADAPTIVE_ARG="--local-adaptive-controller --adaptive-ratio-ladder ${ADAPTIVE_RATIO_LADDER} --adaptive-window ${ADAPTIVE_WINDOW} --adaptive-min-observations ${ADAPTIVE_MIN_OBSERVATIONS} --adaptive-std-k ${ADAPTIVE_STD_K} --adaptive-up-std-k ${ADAPTIVE_UP_STD_K} --adaptive-down-std-k ${ADAPTIVE_DOWN_STD_K} --adaptive-std-floor ${ADAPTIVE_STD_FLOOR} --adaptive-patience ${ADAPTIVE_PATIENCE} --adaptive-cooldown ${ADAPTIVE_COOLDOWN} ${ADAPTIVE_LAYER_ARG}"
    ADAPTIVE_MODE_SUFFIX="${ADAPTIVE_LAYER_SUFFIX}"
    LOCAL_ADAPTIVE_STATUS="enabled"
    ;;
  0|false|no|off)
    LOCAL_ADAPTIVE_ARG=""
    ADAPTIVE_MODE_SUFFIX="-cosine-only"
    LOCAL_ADAPTIVE_STATUS="disabled"
    ADAPTIVE_LAYER_STATUS="disabled"
    ;;
  *)
    echo "LOCAL_ADAPTIVE_CONTROLLER must be one of 1/0, true/false, yes/no, on/off." >&2
    exit 1
    ;;
esac

DRAFT_TOKEN_ARG=""
DRAFT_TOKEN_SUFFIX=""
if [ -n "${DRAFT_TOKEN_NUM}" ]; then
  DRAFT_TOKEN_ARG="--draft-token-num ${DRAFT_TOKEN_NUM}"
  DRAFT_TOKEN_SUFFIX="-draft_token_num-${DRAFT_TOKEN_NUM}"
fi

COSINE_MAX_SKIP_ARG=""
COSINE_MAX_SKIP_SUFFIX=""
if [ -n "${COSINE_MAX_SKIP_LAYERS}" ]; then
  COSINE_MAX_SKIP_ARG="--cosine-max-skip-layers ${COSINE_MAX_SKIP_LAYERS}"
  COSINE_MAX_SKIP_SUFFIX="-cosine_max_skip_layers-${COSINE_MAX_SKIP_LAYERS}"
fi

MODEL_RUN_NAME="${MODEL_NAME}-swift-${torch_dtype}-temp-${TEMP}-top-p-${TOP_P}-seed-${SEED}-max_new_tokens-${MAX_NEW_TOKENS}-opt_interval-${OPT_INTERVAL}-max_score-${MAX_SCORE}-context_window-${CONTEXT_WINDOW}-skip_ratio-${SKIP_RATIO}-draft_kv_retain_ratio-dynamic-4${ADAPTIVE_MODE_SUFFIX}-opt_compressed_draft_kv-True${DRAFT_TOKEN_SUFFIX}"
ANSWER_FILE="outputs/${TASK_NAME}/${TASK_NAME}_${DATA_NUM}/without_layerskip_draft_model_answer/${MODEL_NAME}/${MODEL_RUN_NAME}.jsonl"

mkdir -p "$(dirname "${ANSWER_FILE}")"

echo "Running cosine-prefill SWIFT on ${TASK_NAME} with retain ratio ${RETAIN_RATIO}, DATA_NUM=${DATA_NUM}"
echo "Local adaptive controller: ${LOCAL_ADAPTIVE_STATUS}"
echo "Adaptive layer fallback: ${ADAPTIVE_LAYER_STATUS}"
echo "Output to ${ANSWER_FILE}"

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_swift \
  --model-path "${MODEL_PATH}" \
  --model-id "${MODEL_NAME}" \
  --answer-file "${ANSWER_FILE}" \
  --temperature "${TEMP}" \
  --top-p "${TOP_P}" \
  --dtype "${torch_dtype}" \
  --task-name "${TASK_NAME}" \
  --data-num "${DATA_NUM}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --seed "${SEED}" \
  --context-window "${CONTEXT_WINDOW}" \
  --opt-interval "${OPT_INTERVAL}" \
  --bayes-interval "${BAYES_INTERVAL}" \
  --max-opt-iter "${MAX_OPT_ITER}" \
  --max-tolerance-iter "${MAX_TOLERANCE_ITER}" \
  --max-score "${MAX_SCORE}" \
  --skip-ratio "${SKIP_RATIO}" \
  ${DRAFT_TOKEN_ARG} \
  --draft-kv-compress \
  --draft-kv-retain-ratio "${RETAIN_RATIO}" \
  --optimize-with-compressed-draft-kv \
  --cosine-prefill-skip-layers \
  --cosine-skip-mode "${COSINE_SKIP_MODE}" \
  --cosine-attn-alpha "${COSINE_ATTN_ALPHA}" \
  ${COSINE_MAX_SKIP_ARG} \
  --cosine-keep-first-layers "${COSINE_KEEP_FIRST_LAYERS}" \
  --cosine-keep-last-layers "${COSINE_KEEP_LAST_LAYERS}" \
  --cosine-mlp-interval "${COSINE_MLP_INTERVAL}" \
  ${LOCAL_ADAPTIVE_ARG}
