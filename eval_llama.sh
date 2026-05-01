# MODEL_PATH=/data/models/Llama-2-13b-hf # Llama-2-13b-hf, CodeLlama-13b-hf, Llama-2-13b-chat-hf
# MODEL_NAME=llama-2-13b # llama-2-13b, codellama-13b, llama-2-13b-chat
MODEL_PATH=/home/tjlin/models/Meta-Llama-3-8B-Instruct
MODEL_NAME=Meta-Llama-3-8B-Instruct
TEMP=0.0 # 0.2 for general tasks and 0.6 for code generation
# TOP_P=0.85 # 0.85 for general tasks and 0.95 for code generation
SEED=2024
GPU_DEVICES=0
# MAX_NEW_TOKENS=512

# SWIFT Hyperparameters
OPT_INTERVAL=1
BAYES_INTERVAL=25
MAX_OPT_ITER=1000
MAX_TOLERANCE_ITER=300
MAX_SCORE=0.93
CONTEXT_WINDOW=50
SKIP_RATIO=0.45
DRAFT_TOKEN_NUM=3 # e.g. 8; leave empty to use stop_threshold
OPTIMIZE_WITH_COMPRESSED_DRAFT_KV=0 # 1: optimize layer skips with compressed KV, 0: use uncompressed KV

TASK_NAME="gsm8k" # cnndm, humaneval
DATA_NUM=100
MAX_NEW_TOKENS=256
TOP_P=1.0

torch_dtype="bfloat16" # ["float32", "float64", "float16", "bfloat16"]
DRAFT_TOKEN_ARG=""
if [ -n "${DRAFT_TOKEN_NUM}" ]; then
  DRAFT_TOKEN_ARG="--draft-token-num ${DRAFT_TOKEN_NUM}"
fi
OPTIMIZATION_KV_ARG="--optimize-with-compressed-draft-kv"
if [ "${OPTIMIZE_WITH_COMPRESSED_DRAFT_KV}" = "0" ]; then
  OPTIMIZATION_KV_ARG="--no-optimize-with-compressed-draft-kv"
fi

SKIP_LAYER_CACHE_FILE="outputs/skip_layer_cache.json"
LOAD_SKIP_LAYER_CACHE_ARG=""
SAVE_SKIP_LAYER_CACHE_ARG=""
if [ "${OPTIMIZE_WITH_COMPRESSED_DRAFT_KV}" = "0" ]; then
  LOAD_SKIP_LAYER_CACHE_ARG="--skip-layer-cache-file ${SKIP_LAYER_CACHE_FILE} --load-skip-layer-cache"
  SAVE_SKIP_LAYER_CACHE_ARG="--skip-layer-cache-file ${SKIP_LAYER_CACHE_FILE} --save-skip-layer-cache"
fi

run_swift_eval() {
  RETAIN_RATIO_ARG=$1
  CACHE_ARG=$2

  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_swift --model-path $MODEL_PATH --model-id ${MODEL_NAME} \
    --temperature $TEMP --top-p ${TOP_P} --dtype $torch_dtype --task-name ${TASK_NAME} --data-num ${DATA_NUM} --max-new-tokens ${MAX_NEW_TOKENS} \
    --seed $SEED --context-window ${CONTEXT_WINDOW} --opt-interval ${OPT_INTERVAL} --bayes-interval ${BAYES_INTERVAL} --max-opt-iter ${MAX_OPT_ITER} \
    --max-tolerance-iter ${MAX_TOLERANCE_ITER} --max-score ${MAX_SCORE} --skip-ratio ${SKIP_RATIO} ${DRAFT_TOKEN_ARG} ${OPTIMIZATION_KV_ARG} ${CACHE_ARG} \
    --optimization --bayes --draft-kv-compress --draft-kv-retain-ratio ${RETAIN_RATIO_ARG}
}

run_skip_layer_calibration() {
  if [ "${OPTIMIZE_WITH_COMPRESSED_DRAFT_KV}" = "0" ]; then
    echo "Calibrating shared skip layers for ${TASK_NAME} with uncompressed optimization KV..."
    run_swift_eval 1.0 "${SAVE_SKIP_LAYER_CACHE_ARG}"
  fi
}

run_swift_benchmark() {
  run_skip_layer_calibration
  for RETAIN_RATIO in 0.6 0.5; do
    run_swift_eval ${RETAIN_RATIO} "${LOAD_SKIP_LAYER_CACHE_ARG}"
  done
}

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_baseline --model-path $MODEL_PATH --model-id ${MODEL_NAME} --max-new-tokens ${MAX_NEW_TOKENS} --task-name ${TASK_NAME} --data-num ${DATA_NUM} --temperature $TEMP --top-p ${TOP_P} --seed ${SEED} --dtype $torch_dtype

run_swift_benchmark

TASK_NAME="mmlu"

run_swift_benchmark
