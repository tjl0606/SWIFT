#!/usr/bin/env bash

# MODEL_PATH=/data/models/Llama-2-13b-hf # Llama-2-13b-hf, CodeLlama-13b-hf, Llama-2-13b-chat-hf
# MODEL_NAME=llama-2-13b # llama-2-13b, codellama-13b, llama-2-13b-chat
MODEL_PATH=${MODEL_PATH:-/home/tjlin/models/Meta-Llama-3-8B-Instruct}
MODEL_NAME=${MODEL_NAME:-Meta-Llama-3-8B-Instruct}
TEMP=${TEMP:-0.0} # 0.2 for general tasks and 0.6 for code generation
# TOP_P=0.85 # 0.85 for general tasks and 0.95 for code generation
SEED=${SEED:-2024}
GPU_DEVICES=${GPU_DEVICES:-0}
# MAX_NEW_TOKENS=512

# SWIFT Hyperparameters
OPT_INTERVAL=${OPT_INTERVAL:-1}
BAYES_INTERVAL=${BAYES_INTERVAL:-25}
MAX_OPT_ITER=${MAX_OPT_ITER:-1000}
MAX_TOLERANCE_ITER=${MAX_TOLERANCE_ITER:-300}
MAX_SCORE=${MAX_SCORE:-0.93}
BASE_CONTEXT_WINDOW=${BASE_CONTEXT_WINDOW:-50}
CONTEXT_WINDOW=${BASE_CONTEXT_WINDOW}
SKIP_RATIO=${SKIP_RATIO:-0.45}
DRAFT_TOKEN_NUM=${DRAFT_TOKEN_NUM-} # e.g. 8; leave empty to use stop_threshold
OPTIMIZE_WITH_COMPRESSED_DRAFT_KV=${OPTIMIZE_WITH_COMPRESSED_DRAFT_KV:-1} # 1: optimize layer skips with compressed KV, 0: use uncompressed KV
DYNAMIC_RETAIN_RATIO=${DYNAMIC_RETAIN_RATIO:-1}
RETAIN_RATIO=${RETAIN_RATIO:-1.0}
RETAIN_RATIO_GRID=${RETAIN_RATIO_GRID:-1.0,0.9,0.8,0.7,0.6}
RETAIN_UTILITY_MODE=${RETAIN_UTILITY_MODE:-relative}
RETAIN_COMPRESSION_WEIGHT=${RETAIN_COMPRESSION_WEIGHT:-0.5}
RETAIN_SCORE_TOLERANCE=${RETAIN_SCORE_TOLERANCE:-0.05}
RETAIN_UTILITY_LAMBDA=${RETAIN_UTILITY_LAMBDA:-1.0}
RETAIN_UCB_C=${RETAIN_UCB_C:-0.3}
RETAIN_WARMUP_ROUNDS=${RETAIN_WARMUP_ROUNDS:-50}
RETAIN_FILTER_TOP_K=${RETAIN_FILTER_TOP_K:-3}
RETAIN_REFINE_ROUNDS=${RETAIN_REFINE_ROUNDS:-100}
RETAIN_FINAL_TOLERANCE=${RETAIN_FINAL_TOLERANCE:-0.05}
FINAL_LAYER_REFINE_ROUNDS=${FINAL_LAYER_REFINE_ROUNDS:-500}

TASK_NAME=${TASK_NAME:-gsm8k} # gsm8k, mmlu, triviaqa, natural_questions, cnndm, humaneval, samsum
DATA_NUM=${DATA_NUM:-1000}
TOP_P=${TOP_P:-1.0}

torch_dtype=${torch_dtype:-bfloat16} # ["float32", "float64", "float16", "bfloat16"]

set_task_generation_params() {
  case "${TASK_NAME}" in
    triviaqa|natural_questions)
      MAX_NEW_TOKENS=32
      ;;
    gsm8k|mmlu)
      MAX_NEW_TOKENS=256
      ;;
    samsum)
      MAX_NEW_TOKENS=64
      if [ "${DATA_NUM}" -gt 819 ]; then
        DATA_NUM=819
      fi
      ;;
    cnndm|humaneval)
      MAX_NEW_TOKENS=512
      ;;
    *)
      echo "Unsupported TASK_NAME: ${TASK_NAME}" >&2
      exit 1
      ;;
  esac

  CONTEXT_WINDOW=${BASE_CONTEXT_WINDOW}
  if [ "${MAX_NEW_TOKENS}" -le 32 ]; then
    CONTEXT_WINDOW=16
  fi
}

DRAFT_TOKEN_ARG=""
if [ -n "${DRAFT_TOKEN_NUM}" ]; then
  DRAFT_TOKEN_ARG="--draft-token-num ${DRAFT_TOKEN_NUM}"
fi

if [ "${DYNAMIC_RETAIN_RATIO}" = "1" ]; then
  OPTIMIZE_WITH_COMPRESSED_DRAFT_KV=1
  if [[ ",${RETAIN_RATIO_GRID}," != *",${RETAIN_RATIO},"* ]]; then
    RETAIN_RATIO_GRID="${RETAIN_RATIO},${RETAIN_RATIO_GRID}"
  fi
fi

OPTIMIZATION_KV_ARG="--optimize-with-compressed-draft-kv"
OPTIMIZATION_KV_NAME="True"
if [ "${OPTIMIZE_WITH_COMPRESSED_DRAFT_KV}" = "0" ]; then
  OPTIMIZATION_KV_ARG="--no-optimize-with-compressed-draft-kv"
  OPTIMIZATION_KV_NAME="False"
fi

DYNAMIC_RETAIN_ARG=""
if [ "${DYNAMIC_RETAIN_RATIO}" = "1" ]; then
  DYNAMIC_RETAIN_ARG="--dynamic-retain-ratio --retain-ratio-grid ${RETAIN_RATIO_GRID} --retain-target-score ${MAX_SCORE} --retain-utility-mode ${RETAIN_UTILITY_MODE} --retain-compression-weight ${RETAIN_COMPRESSION_WEIGHT} --retain-score-tolerance ${RETAIN_SCORE_TOLERANCE} --retain-utility-lambda ${RETAIN_UTILITY_LAMBDA} --retain-ucb-c ${RETAIN_UCB_C} --retain-warmup-rounds ${RETAIN_WARMUP_ROUNDS} --retain-filter-top-k ${RETAIN_FILTER_TOP_K} --retain-refine-rounds ${RETAIN_REFINE_ROUNDS} --retain-final-tolerance ${RETAIN_FINAL_TOLERANCE} --final-layer-refine-rounds ${FINAL_LAYER_REFINE_ROUNDS}"
fi

SKIP_LAYER_CACHE_FILE="outputs/skip_layer_cache.json"
LOAD_SKIP_LAYER_CACHE_ARG=""
SAVE_SKIP_LAYER_CACHE_ARG=""
if [ "${OPTIMIZE_WITH_COMPRESSED_DRAFT_KV}" = "0" ]; then
  LOAD_SKIP_LAYER_CACHE_ARG="--skip-layer-cache-file ${SKIP_LAYER_CACHE_FILE} --load-skip-layer-cache"
  SAVE_SKIP_LAYER_CACHE_ARG="--skip-layer-cache-file ${SKIP_LAYER_CACHE_FILE} --save-skip-layer-cache"
fi

baseline_answer_file() {
  MODEL_RUN_NAME="${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP}-top-p-${TOP_P}-seed-${SEED}-max_new_tokens-${MAX_NEW_TOKENS}"
  echo "test/${TASK_NAME}/${TASK_NAME}_${DATA_NUM}/model_answer/${MODEL_NAME}/${MODEL_RUN_NAME}.jsonl"
}

swift_answer_file() {
  RETAIN_RATIO_ARG=$1
  RETAIN_RATIO_NAME=${RETAIN_RATIO_ARG}
  if [ "${DYNAMIC_RETAIN_RATIO}" = "1" ]; then
    RETAIN_RATIO_NAME="dynamic-2"
  fi

  DRAFT_TOKEN_SUFFIX=""
  if [ -n "${DRAFT_TOKEN_NUM}" ]; then
    DRAFT_TOKEN_SUFFIX="-draft_token_num-${DRAFT_TOKEN_NUM}"
  fi

  MODEL_RUN_NAME="${MODEL_NAME}-swift-${torch_dtype}-temp-${TEMP}-top-p-${TOP_P}-seed-${SEED}-max_new_tokens-${MAX_NEW_TOKENS}-opt_interval-${OPT_INTERVAL}-max_score-${MAX_SCORE}-context_window-${CONTEXT_WINDOW}-skip_ratio-${SKIP_RATIO}-draft_kv_retain_ratio-${RETAIN_RATIO_NAME}-opt_compressed_draft_kv-${OPTIMIZATION_KV_NAME}${DRAFT_TOKEN_SUFFIX}"
  echo "outputs/${TASK_NAME}/${TASK_NAME}_${DATA_NUM}/without_layerskip_draft_model_answer/${MODEL_NAME}/${MODEL_RUN_NAME}.jsonl"
}

run_quality_eval() {
  ANSWER_FILE=$1
  if [ "${TASK_NAME}" = "samsum" ]; then
    python evaluation_llama/eval_rouge.py \
      --task-name "${TASK_NAME}" \
      --answer-file "${ANSWER_FILE}" \
      --seed "${SEED}" \
      --data-num "${DATA_NUM}"
  fi
  return 0
}

run_baseline_eval() {
  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_baseline --model-path $MODEL_PATH --model-id ${MODEL_NAME} --max-new-tokens ${MAX_NEW_TOKENS} --task-name ${TASK_NAME} --data-num ${DATA_NUM} --temperature $TEMP --top-p ${TOP_P} --seed ${SEED} --dtype $torch_dtype
  run_quality_eval "$(baseline_answer_file)"
}

run_swift_eval() {
  RETAIN_RATIO_ARG=$1
  CACHE_ARG=$2
  DO_QUALITY_EVAL=${3:-1}
  ANSWER_FILE=$(swift_answer_file ${RETAIN_RATIO_ARG})

  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_swift --model-path $MODEL_PATH --model-id ${MODEL_NAME} \
    --answer-file "${ANSWER_FILE}" \
    --temperature $TEMP --top-p ${TOP_P} --dtype $torch_dtype --task-name ${TASK_NAME} --data-num ${DATA_NUM} --max-new-tokens ${MAX_NEW_TOKENS} \
    --seed $SEED --context-window ${CONTEXT_WINDOW} --opt-interval ${OPT_INTERVAL} --bayes-interval ${BAYES_INTERVAL} --max-opt-iter ${MAX_OPT_ITER} \
    --max-tolerance-iter ${MAX_TOLERANCE_ITER} --max-score ${MAX_SCORE} --skip-ratio ${SKIP_RATIO} ${DRAFT_TOKEN_ARG} ${OPTIMIZATION_KV_ARG} ${DYNAMIC_RETAIN_ARG} ${CACHE_ARG} \
    --optimization --bayes --draft-kv-compress --draft-kv-retain-ratio ${RETAIN_RATIO_ARG}

  if [ "${DO_QUALITY_EVAL}" = "1" ]; then
    run_quality_eval "${ANSWER_FILE}"
  fi
}

run_skip_layer_calibration() {
  if [ "${DYNAMIC_RETAIN_RATIO}" = "1" ]; then
    return 0
  fi

  if [ "${OPTIMIZE_WITH_COMPRESSED_DRAFT_KV}" = "0" ]; then
    echo "Calibrating shared skip layers for ${TASK_NAME} with uncompressed optimization KV..."
    run_swift_eval 1.0 "${SAVE_SKIP_LAYER_CACHE_ARG}" 0
  fi
}

run_swift_benchmark() {
  run_skip_layer_calibration
  if [ "${DYNAMIC_RETAIN_RATIO}" = "1" ]; then
    run_swift_eval ${RETAIN_RATIO} ""
    return 0
  fi

  for RETAIN_RATIO in 1.0 0.99 0.9 0.8 0.7 0.6; do
    run_swift_eval ${RETAIN_RATIO} "${LOAD_SKIP_LAYER_CACHE_ARG}"
  done
}

set_task_generation_params
echo "Running ${TASK_NAME} with MAX_NEW_TOKENS=${MAX_NEW_TOKENS}, DATA_NUM=${DATA_NUM}, CONTEXT_WINDOW=${CONTEXT_WINDOW}"
# run_baseline_eval
run_swift_benchmark
