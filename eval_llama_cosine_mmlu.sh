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
MAX_NEW_TOKENS=${MAX_NEW_TOKENS-}
CONTEXT_WINDOW=${CONTEXT_WINDOW:-50}
OPT_INTERVAL=${OPT_INTERVAL:-1}
BAYES_INTERVAL=${BAYES_INTERVAL:-25}
MAX_OPT_ITER=${MAX_OPT_ITER:-1100}
MAX_TOLERANCE_ITER=${MAX_TOLERANCE_ITER:-300}
MAX_SCORE=${MAX_SCORE:-0.93}
SKIP_RATIO=${SKIP_RATIO:-0.45}
DRAFT_TOKEN_NUM=${DRAFT_TOKEN_NUM-}
RUN_BASELINE=${RUN_BASELINE-}
DEFAULT_FINAL_ADAPTIVE=0
DEFAULT_RUN_BASELINE=1

case "${TASK_NAME}" in
  mmlu)
    DEFAULT_RETAIN_RATIO=0.6
    DEFAULT_MAX_NEW_TOKENS=256
    DEFAULT_ADAPTIVE_WINDOW=8
    DEFAULT_ADAPTIVE_MIN_OBSERVATIONS=12
    ;;
  gsm8k)
    DEFAULT_RETAIN_RATIO=0.7
    DEFAULT_MAX_NEW_TOKENS=384
    DEFAULT_ADAPTIVE_WINDOW=16
    DEFAULT_ADAPTIVE_MIN_OBSERVATIONS=24
    ;;
  samsum)
    DEFAULT_RETAIN_RATIO=0.6
    DEFAULT_MAX_NEW_TOKENS=96
    DEFAULT_ADAPTIVE_WINDOW=8
    DEFAULT_ADAPTIVE_MIN_OBSERVATIONS=12
    if [ "${DATA_NUM}" -gt 819 ]; then
      DATA_NUM=819
    fi
    ;;
  mt_bench)
    DEFAULT_RETAIN_RATIO=0.8
    DEFAULT_MAX_NEW_TOKENS=1024
    DEFAULT_ADAPTIVE_WINDOW=16
    DEFAULT_ADAPTIVE_MIN_OBSERVATIONS=24
    DEFAULT_FINAL_ADAPTIVE=1
    DEFAULT_RUN_BASELINE=0
    if [ "${DATA_NUM}" -gt 80 ]; then
      DATA_NUM=80
    fi
    ;;
  *)
    echo "This cosine-prefill script currently supports TASK_NAME=mmlu, TASK_NAME=gsm8k, TASK_NAME=samsum, or TASK_NAME=mt_bench." >&2
    exit 1
    ;;
esac

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-${DEFAULT_MAX_NEW_TOKENS}}
RETAIN_RATIO=${RETAIN_RATIO:-${DEFAULT_RETAIN_RATIO}}
RUN_BASELINE=${RUN_BASELINE:-${DEFAULT_RUN_BASELINE}}
AGGRESSIVE_ADAPTIVE=${AGGRESSIVE_ADAPTIVE:-1}
FINAL_ADAPTIVE=${FINAL_ADAPTIVE:-${DEFAULT_FINAL_ADAPTIVE}}
FINAL2_ADAPTIVE=${FINAL2_ADAPTIVE:-0}
ADAPTIVE_MIN_RETAIN_RATIO=${ADAPTIVE_MIN_RETAIN_RATIO:-0.1}
ADAPTIVE_RATIO_STEP=${ADAPTIVE_RATIO_STEP:-0.1}
ADAPTIVE_AGGRESSIVE_TOLERANCE=${ADAPTIVE_AGGRESSIVE_TOLERANCE:-0.02}
ADAPTIVE_AGGRESSIVE_STD_K=${ADAPTIVE_AGGRESSIVE_STD_K:-0.5}
ADAPTIVE_AGGRESSIVE_PATIENCE=${ADAPTIVE_AGGRESSIVE_PATIENCE:-1}
ADAPTIVE_MAX_EXTRA_SKIP_LAYERS=${ADAPTIVE_MAX_EXTRA_SKIP_LAYERS-}
ADAPTIVE_MAX_SKIP_LAYERS=${ADAPTIVE_MAX_SKIP_LAYERS-}
LYAPUNOV_ADAPTIVE=${LYAPUNOV_ADAPTIVE:-0}
LYAPUNOV_ACCEPTANCE_TARGET=${LYAPUNOV_ACCEPTANCE_TARGET:-0.92}
LYAPUNOV_V=${LYAPUNOV_V:-0.1}
LYAPUNOV_SWITCH_COST=${LYAPUNOV_SWITCH_COST:-0.01}
LYAPUNOV_LAYER_SWITCH_COST=${LYAPUNOV_LAYER_SWITCH_COST:-${LYAPUNOV_SWITCH_COST}}
LYAPUNOV_LAYER_PENALTY_WEIGHT=${LYAPUNOV_LAYER_PENALTY_WEIGHT:-0.02}
LYAPUNOV_PREDICTION_BETA=${LYAPUNOV_PREDICTION_BETA:-0.5}
LYAPUNOV_RATIO_ACCEPTANCE_SLOPE=${LYAPUNOV_RATIO_ACCEPTANCE_SLOPE:-0.2}
LYAPUNOV_LAYER_ACCEPTANCE_SLOPE=${LYAPUNOV_LAYER_ACCEPTANCE_SLOPE:-0.015}
LYAPUNOV_COLD_START_PENALTY=${LYAPUNOV_COLD_START_PENALTY:-0.03}
FINAL_TARGET_MEAN_ACCEPTED=${FINAL_TARGET_MEAN_ACCEPTED:-3.0}
FINAL_BAD_MEAN_ACCEPTED=${FINAL_BAD_MEAN_ACCEPTED:-2.5}
FINAL_SEVERE_MEAN_ACCEPTED=${FINAL_SEVERE_MEAN_ACCEPTED:-2.1}
FINAL_TOKEN_ACCEPTANCE_FLOOR=${FINAL_TOKEN_ACCEPTANCE_FLOOR:-0.85}
FINAL_MORE_SKIP_TOKEN_ACCEPTANCE_FLOOR=${FINAL_MORE_SKIP_TOKEN_ACCEPTANCE_FLOOR:-0.90}
FINAL_DRAFT_LEN_FLOOR=${FINAL_DRAFT_LEN_FLOOR:-2.0}
FINAL_MORE_SKIP_DRAFT_LEN_FLOOR=${FINAL_MORE_SKIP_DRAFT_LEN_FLOOR:-2.2}
FINAL_STABLE_MEAN_MARGIN=${FINAL_STABLE_MEAN_MARGIN:-0.1}
FINAL_SOFT_MAX_SKIP_LAYERS=${FINAL_SOFT_MAX_SKIP_LAYERS:-17}
FINAL_HARD_MAX_SKIP_LAYERS=${FINAL_HARD_MAX_SKIP_LAYERS:-18}
FINAL_MIN_RATIO_FOR_MORE_SKIP=${FINAL_MIN_RATIO_FOR_MORE_SKIP:-0.4}
FINAL_LOW_RATIO_GUARD=${FINAL_LOW_RATIO_GUARD:-0.2}
FINAL_LOW_RATIO_GUARD_SKIP_LAYERS=${FINAL_LOW_RATIO_GUARD_SKIP_LAYERS:-17}
FINAL_HARD_LAYER_MEAN_FLOOR=${FINAL_HARD_LAYER_MEAN_FLOOR:-2.8}
FINAL_HARD_PROBE_MEAN_MARGIN=${FINAL_HARD_PROBE_MEAN_MARGIN:-0.5}
FINAL_HARD_PROBE_TOKEN_ACCEPTANCE_FLOOR=${FINAL_HARD_PROBE_TOKEN_ACCEPTANCE_FLOOR:-0.92}
FINAL_HARD_PROBE_DRAFT_LEN_MARGIN=${FINAL_HARD_PROBE_DRAFT_LEN_MARGIN:-0.3}
FINAL_RATIO_DOWN_GAIN_WEIGHT=${FINAL_RATIO_DOWN_GAIN_WEIGHT:-1.0}
FINAL_LAYER_SKIP_GAIN_WEIGHT=${FINAL_LAYER_SKIP_GAIN_WEIGHT:-3.0}
FINAL2_LOW_STD_K=${FINAL2_LOW_STD_K:-0.5}
FINAL2_HIGH_STD_K=${FINAL2_HIGH_STD_K:-0.5}
FINAL2_MEAN_STD_FLOOR=${FINAL2_MEAN_STD_FLOOR:-0.10}
FINAL2_MIN_CONFIG_OBSERVATIONS=${FINAL2_MIN_CONFIG_OBSERVATIONS:-16}
FINAL2_PREDICTION_BETA=${FINAL2_PREDICTION_BETA:-0.5}
FINAL2_RATIO_MEAN_SLOPE=${FINAL2_RATIO_MEAN_SLOPE:-1.0}
FINAL2_LAYER_MEAN_SLOPE=${FINAL2_LAYER_MEAN_SLOPE:-0.45}
FINAL2_COLD_START_PENALTY=${FINAL2_COLD_START_PENALTY:-0.15}
FINAL2_TOKEN_ACCEPTANCE_FLOOR=${FINAL2_TOKEN_ACCEPTANCE_FLOOR:-0.85}
FINAL2_MORE_AGGRESSIVE_TOKEN_ACCEPTANCE_FLOOR=${FINAL2_MORE_AGGRESSIVE_TOKEN_ACCEPTANCE_FLOOR:-0.90}
FINAL2_DRAFT_LEN_FLOOR=${FINAL2_DRAFT_LEN_FLOOR:-2.0}
FINAL2_MORE_AGGRESSIVE_DRAFT_LEN_FLOOR=${FINAL2_MORE_AGGRESSIVE_DRAFT_LEN_FLOOR:-2.2}
FINAL2_SWITCH_COST=${FINAL2_SWITCH_COST:-0.02}
FINAL2_LAYER_SWITCH_COST=${FINAL2_LAYER_SWITCH_COST:-${FINAL2_SWITCH_COST}}
FINAL2_RATIO_DOWN_GAIN_WEIGHT=${FINAL2_RATIO_DOWN_GAIN_WEIGHT:-1.0}
FINAL2_LAYER_SKIP_GAIN_WEIGHT=${FINAL2_LAYER_SKIP_GAIN_WEIGHT:-2.0}

# Dynamic-6 cold start is a one-time warmup before the benchmark stream.  The
# default dynamic mode uses a task-agnostic repeat-passage prompt while final2
# adapts, then merges weak table priors back into the real run.
ADAPTIVE_COLD_START=${ADAPTIVE_COLD_START:-0}
ADAPTIVE_COLD_START_MODE=${ADAPTIVE_COLD_START_MODE:-dynamic}
ADAPTIVE_COLD_START_PROMPT=${ADAPTIVE_COLD_START_PROMPT:-auto}
ADAPTIVE_COLD_START_MAX_NEW_TOKENS=${ADAPTIVE_COLD_START_MAX_NEW_TOKENS:-192}
ADAPTIVE_COLD_START_MAX_STEPS=${ADAPTIVE_COLD_START_MAX_STEPS:-192}
ADAPTIVE_COLD_START_EFFECTIVE_COUNT=${ADAPTIVE_COLD_START_EFFECTIVE_COUNT:-16}
ADAPTIVE_COLD_START_MAX_CONFIGS=${ADAPTIVE_COLD_START_MAX_CONFIGS:-12}
ADAPTIVE_COLD_START_RATIOS=${ADAPTIVE_COLD_START_RATIOS-}
ADAPTIVE_COLD_START_SKIP_COUNTS=${ADAPTIVE_COLD_START_SKIP_COUNTS-}
ADAPTIVE_COLD_START_SKIP_DELTA=${ADAPTIVE_COLD_START_SKIP_DELTA:-2}

case "${LYAPUNOV_ADAPTIVE,,}" in
  1|true|yes|on)
    LYAPUNOV_ADAPTIVE_STATUS="enabled"
    ;;
  0|false|no|off)
    LYAPUNOV_ADAPTIVE_STATUS="disabled"
    ;;
  *)
    echo "LYAPUNOV_ADAPTIVE must be one of 1/0, true/false, yes/no, on/off." >&2
    exit 1
    ;;
esac

case "${AGGRESSIVE_ADAPTIVE,,}" in
  1|true|yes|on)
    AGGRESSIVE_ADAPTIVE_STATUS="enabled"
    ;;
  0|false|no|off)
    AGGRESSIVE_ADAPTIVE_STATUS="disabled"
    ;;
  *)
    echo "AGGRESSIVE_ADAPTIVE must be one of 1/0, true/false, yes/no, on/off." >&2
    exit 1
    ;;
esac

case "${FINAL_ADAPTIVE,,}" in
  1|true|yes|on)
    FINAL_ADAPTIVE_STATUS="enabled"
    ;;
  0|false|no|off)
    FINAL_ADAPTIVE_STATUS="disabled"
    ;;
  *)
    echo "FINAL_ADAPTIVE must be one of 1/0, true/false, yes/no, on/off." >&2
    exit 1
    ;;
esac

case "${FINAL2_ADAPTIVE,,}" in
  1|true|yes|on)
    FINAL2_ADAPTIVE_STATUS="enabled"
    ;;
  0|false|no|off)
    FINAL2_ADAPTIVE_STATUS="disabled"
    ;;
  *)
    echo "FINAL2_ADAPTIVE must be one of 1/0, true/false, yes/no, on/off." >&2
    exit 1
    ;;
esac

case "${ADAPTIVE_COLD_START,,}" in
  1|true|yes|on)
    ADAPTIVE_COLD_START_STATUS="enabled"
    ;;
  0|false|no|off)
    ADAPTIVE_COLD_START_STATUS="disabled"
    ;;
  *)
    echo "ADAPTIVE_COLD_START must be one of 1/0, true/false, yes/no, on/off." >&2
    exit 1
    ;;
esac

case "${ADAPTIVE_COLD_START_MODE}" in
  dynamic|probe)
    ;;
  *)
    echo "ADAPTIVE_COLD_START_MODE must be one of dynamic or probe." >&2
    exit 1
    ;;
esac

if [ "${ADAPTIVE_COLD_START_STATUS}" = "enabled" ]; then
  FINAL2_ADAPTIVE_STATUS="enabled"
  AGGRESSIVE_ADAPTIVE_STATUS="disabled"
  FINAL_ADAPTIVE_STATUS="disabled"
  LYAPUNOV_ADAPTIVE_STATUS="disabled"
elif [ "${FINAL2_ADAPTIVE_STATUS}" = "enabled" ]; then
  AGGRESSIVE_ADAPTIVE_STATUS="disabled"
  FINAL_ADAPTIVE_STATUS="disabled"
  LYAPUNOV_ADAPTIVE_STATUS="disabled"
elif [ "${FINAL_ADAPTIVE_STATUS}" = "enabled" ]; then
  AGGRESSIVE_ADAPTIVE_STATUS="disabled"
  LYAPUNOV_ADAPTIVE_STATUS="disabled"
elif [ "${LYAPUNOV_ADAPTIVE_STATUS}" = "enabled" ] && [ "${AGGRESSIVE_ADAPTIVE_STATUS}" = "enabled" ]; then
  AGGRESSIVE_ADAPTIVE_STATUS="disabled"
fi

if [ -z "${ADAPTIVE_RATIO_LADDER+x}" ]; then
  if [ "${AGGRESSIVE_ADAPTIVE_STATUS}" = "enabled" ] || [ "${LYAPUNOV_ADAPTIVE_STATUS}" = "enabled" ] || [ "${FINAL_ADAPTIVE_STATUS}" = "enabled" ] || [ "${FINAL2_ADAPTIVE_STATUS}" = "enabled" ]; then
    ADAPTIVE_RATIO_LADDER="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
  else
    ADAPTIVE_RATIO_LADDER="0.6,0.7,0.8,0.9,1.0"
  fi
fi

ADAPTIVE_WINDOW=${ADAPTIVE_WINDOW:-${DEFAULT_ADAPTIVE_WINDOW}}
ADAPTIVE_MIN_OBSERVATIONS=${ADAPTIVE_MIN_OBSERVATIONS:-${DEFAULT_ADAPTIVE_MIN_OBSERVATIONS}}
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

# Draft KV compression defaults to reuse+mask.  Reuse keeps attention-heavy
# token selection without the extra observation forward pass; mask avoids
# physically copying KV, while still using the selected indices as visibility.
DRAFT_KV_SCORE_SOURCE=${DRAFT_KV_SCORE_SOURCE:-reuse}
DRAFT_KV_CACHE_MODE=${DRAFT_KV_CACHE_MODE:-mask}
DRAFT_KV_REUSE_EMA=${DRAFT_KV_REUSE_EMA:-0.7}
WRITE_LOG=${WRITE_LOG:-1}

case "${DRAFT_KV_CACHE_MODE}" in
  copy|mask)
    ;;
  *)
    echo "DRAFT_KV_CACHE_MODE must be one of copy or mask." >&2
    exit 1
    ;;
esac

case "${WRITE_LOG,,}" in
  1|true|yes|on)
    WRITE_LOG_STATUS="enabled"
    ;;
  0|false|no|off)
    WRITE_LOG_STATUS="disabled"
    ;;
  *)
    echo "WRITE_LOG must be one of 1/0, true/false, yes/no, on/off." >&2
    exit 1
    ;;
esac
LOCAL_ADAPTIVE_CONTROLLER=${LOCAL_ADAPTIVE_CONTROLLER:-1}
if [ "${ADAPTIVE_COLD_START_STATUS}" = "enabled" ]; then
  LOCAL_ADAPTIVE_CONTROLLER=1
fi
ADAPTIVE_LAYER_CONTROLLER=${ADAPTIVE_LAYER_CONTROLLER:-1}
ADAPTIVE_LAYER_FALLBACK_WINDOW=${ADAPTIVE_LAYER_FALLBACK_WINDOW:-16}
ADAPTIVE_LAYER_IMPROVEMENT_DELTA=${ADAPTIVE_LAYER_IMPROVEMENT_DELTA:-0.0}

ADAPTIVE_MAX_EXTRA_ARG=""
if [ -n "${ADAPTIVE_MAX_EXTRA_SKIP_LAYERS}" ]; then
  ADAPTIVE_MAX_EXTRA_ARG="--adaptive-max-extra-skip-layers ${ADAPTIVE_MAX_EXTRA_SKIP_LAYERS}"
fi
ADAPTIVE_MAX_SKIP_ARG=""
ADAPTIVE_MAX_SKIP_SUFFIX=""
if [ -n "${ADAPTIVE_MAX_SKIP_LAYERS}" ]; then
  ADAPTIVE_MAX_SKIP_ARG="--adaptive-max-skip-layers ${ADAPTIVE_MAX_SKIP_LAYERS}"
  ADAPTIVE_MAX_SKIP_SUFFIX="-skipcap${ADAPTIVE_MAX_SKIP_LAYERS}"
fi

case "${AGGRESSIVE_ADAPTIVE_STATUS}" in
  enabled)
    AGGRESSIVE_ADAPTIVE_ARG="--adaptive-aggressive-controller --adaptive-min-retain-ratio ${ADAPTIVE_MIN_RETAIN_RATIO} --adaptive-ratio-step ${ADAPTIVE_RATIO_STEP} --adaptive-aggressive-tolerance ${ADAPTIVE_AGGRESSIVE_TOLERANCE} --adaptive-aggressive-std-k ${ADAPTIVE_AGGRESSIVE_STD_K} --adaptive-aggressive-patience ${ADAPTIVE_AGGRESSIVE_PATIENCE} ${ADAPTIVE_MAX_EXTRA_ARG}"
    AGGRESSIVE_ADAPTIVE_SUFFIX="-aggressive-k${ADAPTIVE_AGGRESSIVE_STD_K}"
    ;;
  disabled)
    AGGRESSIVE_ADAPTIVE_ARG=""
    AGGRESSIVE_ADAPTIVE_SUFFIX=""
    ;;
esac

case "${LYAPUNOV_ADAPTIVE_STATUS}" in
  enabled)
    LYAPUNOV_ADAPTIVE_ARG="--lyapunov-adaptive-controller --lyapunov-acceptance-target ${LYAPUNOV_ACCEPTANCE_TARGET} --lyapunov-v ${LYAPUNOV_V} --lyapunov-switch-cost ${LYAPUNOV_SWITCH_COST} --lyapunov-layer-switch-cost ${LYAPUNOV_LAYER_SWITCH_COST} --lyapunov-layer-penalty-weight ${LYAPUNOV_LAYER_PENALTY_WEIGHT} --lyapunov-prediction-beta ${LYAPUNOV_PREDICTION_BETA} --lyapunov-ratio-acceptance-slope ${LYAPUNOV_RATIO_ACCEPTANCE_SLOPE} --lyapunov-layer-acceptance-slope ${LYAPUNOV_LAYER_ACCEPTANCE_SLOPE} --lyapunov-cold-start-penalty ${LYAPUNOV_COLD_START_PENALTY} --adaptive-min-retain-ratio ${ADAPTIVE_MIN_RETAIN_RATIO} --adaptive-ratio-step ${ADAPTIVE_RATIO_STEP} ${ADAPTIVE_MAX_EXTRA_ARG}"
    ;;
  disabled)
    LYAPUNOV_ADAPTIVE_ARG=""
    ;;
esac

case "${FINAL_ADAPTIVE_STATUS}" in
  enabled)
    FINAL_ADAPTIVE_ARG="--adaptive-final-controller --adaptive-min-retain-ratio ${ADAPTIVE_MIN_RETAIN_RATIO} --adaptive-ratio-step ${ADAPTIVE_RATIO_STEP} --final-target-mean-accepted ${FINAL_TARGET_MEAN_ACCEPTED} --final-bad-mean-accepted ${FINAL_BAD_MEAN_ACCEPTED} --final-severe-mean-accepted ${FINAL_SEVERE_MEAN_ACCEPTED} --final-token-acceptance-floor ${FINAL_TOKEN_ACCEPTANCE_FLOOR} --final-more-skip-token-acceptance-floor ${FINAL_MORE_SKIP_TOKEN_ACCEPTANCE_FLOOR} --final-draft-len-floor ${FINAL_DRAFT_LEN_FLOOR} --final-more-skip-draft-len-floor ${FINAL_MORE_SKIP_DRAFT_LEN_FLOOR} --final-stable-mean-margin ${FINAL_STABLE_MEAN_MARGIN} --final-soft-max-skip-layers ${FINAL_SOFT_MAX_SKIP_LAYERS} --final-hard-max-skip-layers ${FINAL_HARD_MAX_SKIP_LAYERS} --final-min-ratio-for-more-skip ${FINAL_MIN_RATIO_FOR_MORE_SKIP} --final-low-ratio-guard ${FINAL_LOW_RATIO_GUARD} --final-low-ratio-guard-skip-layers ${FINAL_LOW_RATIO_GUARD_SKIP_LAYERS} --final-hard-layer-mean-floor ${FINAL_HARD_LAYER_MEAN_FLOOR} --final-hard-probe-mean-margin ${FINAL_HARD_PROBE_MEAN_MARGIN} --final-hard-probe-token-acceptance-floor ${FINAL_HARD_PROBE_TOKEN_ACCEPTANCE_FLOOR} --final-hard-probe-draft-len-margin ${FINAL_HARD_PROBE_DRAFT_LEN_MARGIN} --final-ratio-down-gain-weight ${FINAL_RATIO_DOWN_GAIN_WEIGHT} --final-layer-skip-gain-weight ${FINAL_LAYER_SKIP_GAIN_WEIGHT}"
    ;;
  disabled)
    FINAL_ADAPTIVE_ARG=""
    ;;
esac

case "${FINAL2_ADAPTIVE_STATUS}" in
  enabled)
    FINAL2_ADAPTIVE_ARG="--adaptive-final2-controller --adaptive-min-retain-ratio ${ADAPTIVE_MIN_RETAIN_RATIO} --adaptive-ratio-step ${ADAPTIVE_RATIO_STEP} --final2-low-std-k ${FINAL2_LOW_STD_K} --final2-high-std-k ${FINAL2_HIGH_STD_K} --final2-mean-std-floor ${FINAL2_MEAN_STD_FLOOR} --final2-min-config-observations ${FINAL2_MIN_CONFIG_OBSERVATIONS} --final2-prediction-beta ${FINAL2_PREDICTION_BETA} --final2-ratio-mean-slope ${FINAL2_RATIO_MEAN_SLOPE} --final2-layer-mean-slope ${FINAL2_LAYER_MEAN_SLOPE} --final2-cold-start-penalty ${FINAL2_COLD_START_PENALTY} --final2-token-acceptance-floor ${FINAL2_TOKEN_ACCEPTANCE_FLOOR} --final2-more-aggressive-token-acceptance-floor ${FINAL2_MORE_AGGRESSIVE_TOKEN_ACCEPTANCE_FLOOR} --final2-draft-len-floor ${FINAL2_DRAFT_LEN_FLOOR} --final2-more-aggressive-draft-len-floor ${FINAL2_MORE_AGGRESSIVE_DRAFT_LEN_FLOOR} --final2-switch-cost ${FINAL2_SWITCH_COST} --final2-layer-switch-cost ${FINAL2_LAYER_SWITCH_COST} --final2-ratio-down-gain-weight ${FINAL2_RATIO_DOWN_GAIN_WEIGHT} --final2-layer-skip-gain-weight ${FINAL2_LAYER_SKIP_GAIN_WEIGHT}"
    ;;
  disabled)
    FINAL2_ADAPTIVE_ARG=""
    ;;
esac

ADAPTIVE_COLD_START_ARGS=()
case "${ADAPTIVE_COLD_START_STATUS}" in
  enabled)
    ADAPTIVE_COLD_START_ARGS=(
      --adaptive-cold-start
      --adaptive-cold-start-mode "${ADAPTIVE_COLD_START_MODE}"
      --adaptive-cold-start-prompt "${ADAPTIVE_COLD_START_PROMPT}"
      --adaptive-cold-start-max-new-tokens "${ADAPTIVE_COLD_START_MAX_NEW_TOKENS}"
      --adaptive-cold-start-max-steps "${ADAPTIVE_COLD_START_MAX_STEPS}"
      --adaptive-cold-start-effective-count "${ADAPTIVE_COLD_START_EFFECTIVE_COUNT}"
      --adaptive-cold-start-max-configs "${ADAPTIVE_COLD_START_MAX_CONFIGS}"
      --adaptive-cold-start-skip-delta "${ADAPTIVE_COLD_START_SKIP_DELTA}"
    )
    if [ -n "${ADAPTIVE_COLD_START_RATIOS}" ]; then
      ADAPTIVE_COLD_START_ARGS+=(--adaptive-cold-start-ratios "${ADAPTIVE_COLD_START_RATIOS}")
    fi
    if [ -n "${ADAPTIVE_COLD_START_SKIP_COUNTS}" ]; then
      ADAPTIVE_COLD_START_ARGS+=(--adaptive-cold-start-skip-counts "${ADAPTIVE_COLD_START_SKIP_COUNTS}")
    fi
    ;;
  disabled)
    ;;
esac

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
    LOCAL_ADAPTIVE_ARG="--local-adaptive-controller --adaptive-ratio-ladder ${ADAPTIVE_RATIO_LADDER} --adaptive-window ${ADAPTIVE_WINDOW} --adaptive-min-observations ${ADAPTIVE_MIN_OBSERVATIONS} --adaptive-std-k ${ADAPTIVE_STD_K} --adaptive-up-std-k ${ADAPTIVE_UP_STD_K} --adaptive-down-std-k ${ADAPTIVE_DOWN_STD_K} --adaptive-std-floor ${ADAPTIVE_STD_FLOOR} --adaptive-patience ${ADAPTIVE_PATIENCE} --adaptive-cooldown ${ADAPTIVE_COOLDOWN} ${ADAPTIVE_LAYER_ARG} ${ADAPTIVE_MAX_SKIP_ARG} ${AGGRESSIVE_ADAPTIVE_ARG} ${LYAPUNOV_ADAPTIVE_ARG} ${FINAL_ADAPTIVE_ARG} ${FINAL2_ADAPTIVE_ARG}"
    if [ "${LYAPUNOV_ADAPTIVE_STATUS}" = "enabled" ]; then
      ADAPTIVE_MODE_SUFFIX="-lyapunov-v${LYAPUNOV_V}${ADAPTIVE_LAYER_SUFFIX}${ADAPTIVE_MAX_SKIP_SUFFIX}"
    elif [ "${FINAL2_ADAPTIVE_STATUS}" = "enabled" ]; then
      ADAPTIVE_MODE_SUFFIX="-final2${ADAPTIVE_LAYER_SUFFIX}${ADAPTIVE_MAX_SKIP_SUFFIX}"
    elif [ "${FINAL_ADAPTIVE_STATUS}" = "enabled" ]; then
      ADAPTIVE_MODE_SUFFIX="-final${ADAPTIVE_LAYER_SUFFIX}${ADAPTIVE_MAX_SKIP_SUFFIX}"
    else
      ADAPTIVE_MODE_SUFFIX="${AGGRESSIVE_ADAPTIVE_SUFFIX}${ADAPTIVE_LAYER_SUFFIX}${ADAPTIVE_MAX_SKIP_SUFFIX}"
    fi
    LOCAL_ADAPTIVE_STATUS="enabled"
    ;;
  0|false|no|off)
    LOCAL_ADAPTIVE_ARG=""
    ADAPTIVE_MODE_SUFFIX="-cosine-only"
    LOCAL_ADAPTIVE_STATUS="disabled"
    ADAPTIVE_LAYER_STATUS="disabled"
    AGGRESSIVE_ADAPTIVE_STATUS="disabled"
    LYAPUNOV_ADAPTIVE_STATUS="disabled"
    FINAL_ADAPTIVE_STATUS="disabled"
    FINAL2_ADAPTIVE_STATUS="disabled"
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

if [ "${ADAPTIVE_COLD_START_STATUS}" = "enabled" ]; then
  RETAIN_RATIO_RUN_NAME="dynamic-6"
elif [ "${LYAPUNOV_ADAPTIVE_STATUS}" = "enabled" ]; then
  RETAIN_RATIO_RUN_NAME="dynamic-5${ADAPTIVE_MODE_SUFFIX}"
else
  RETAIN_RATIO_RUN_NAME="dynamic-4${ADAPTIVE_MODE_SUFFIX}"
fi
DRAFT_KV_SCORE_SUFFIX="-kvsrc-${DRAFT_KV_SCORE_SOURCE}"
DRAFT_KV_CACHE_SUFFIX=""
if [ "${DRAFT_KV_CACHE_MODE}" = "mask" ]; then
  DRAFT_KV_CACHE_SUFFIX="-kvmask"
fi
MODEL_RUN_NAME="${MODEL_NAME}-swift-${torch_dtype}-temp-${TEMP}-top-p-${TOP_P}-seed-${SEED}-max_new_tokens-${MAX_NEW_TOKENS}-opt_interval-${OPT_INTERVAL}-max_score-${MAX_SCORE}-context_window-${CONTEXT_WINDOW}-skip_ratio-${SKIP_RATIO}-draft_kv_retain_ratio-${RETAIN_RATIO_RUN_NAME}-opt_compressed_draft_kv-True${DRAFT_KV_CACHE_SUFFIX}${DRAFT_KV_SCORE_SUFFIX}${DRAFT_TOKEN_SUFFIX}"
ANSWER_FILE="outputs/${TASK_NAME}/${TASK_NAME}_${DATA_NUM}/without_layerskip_draft_model_answer/${MODEL_NAME}/${MODEL_RUN_NAME}.jsonl"
LOG_FILE="${ANSWER_FILE%.jsonl}.log"


baseline_answer_file() {
  local baseline_run_name
  baseline_run_name="${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP}-top-p-${TOP_P}-seed-${SEED}-max_new_tokens-${MAX_NEW_TOKENS}"
  echo "test/${TASK_NAME}/${TASK_NAME}_${DATA_NUM}/model_answer/${MODEL_NAME}/${baseline_run_name}.jsonl"
}

run_quality_eval() {
  local answer_file
  answer_file=$1
  if [ "${TASK_NAME}" = "samsum" ]; then
    python evaluation_llama/eval_rouge.py \
      --task-name "${TASK_NAME}" \
      --answer-file "${answer_file}" \
      --seed "${SEED}" \
      --data-num "${DATA_NUM}"
  fi
  return 0
}

run_baseline_eval() {
  local baseline_file
  baseline_file=$(baseline_answer_file)

  echo "Running vanilla baseline on ${TASK_NAME}, DATA_NUM=${DATA_NUM}"
  echo "Baseline output to ${baseline_file}"

  CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_baseline \
    --model-path "${MODEL_PATH}" \
    --model-id "${MODEL_NAME}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --task-name "${TASK_NAME}" \
    --data-num "${DATA_NUM}" \
    --temperature "${TEMP}" \
    --top-p "${TOP_P}" \
    --seed "${SEED}" \
    --dtype "${torch_dtype}"

  run_quality_eval "${baseline_file}"
}

mkdir -p "$(dirname "${ANSWER_FILE}")"

case "${RUN_BASELINE,,}" in
  1|true|yes|on)
    run_baseline_eval
    ;;
  0|false|no|off)
    echo "Skipping vanilla baseline because RUN_BASELINE=${RUN_BASELINE}"
    ;;
  *)
    echo "RUN_BASELINE must be one of 1/0, true/false, yes/no, on/off." >&2
    exit 1
    ;;
esac

echo "Running cosine-prefill SWIFT on ${TASK_NAME} with retain ratio ${RETAIN_RATIO}, DATA_NUM=${DATA_NUM}"
echo "Local adaptive controller: ${LOCAL_ADAPTIVE_STATUS}"
echo "Aggressive adaptive: ${AGGRESSIVE_ADAPTIVE_STATUS}"
echo "Final adaptive: ${FINAL_ADAPTIVE_STATUS}"
echo "Final2 adaptive: ${FINAL2_ADAPTIVE_STATUS}"
echo "Lyapunov adaptive: ${LYAPUNOV_ADAPTIVE_STATUS}"
echo "Adaptive cold start: ${ADAPTIVE_COLD_START_STATUS}"
echo "Adaptive layer fallback: ${ADAPTIVE_LAYER_STATUS}"
echo "Adaptive ratio ladder: ${ADAPTIVE_RATIO_LADDER}"
echo "Draft KV cache mode: ${DRAFT_KV_CACHE_MODE}"
echo "Draft KV score source: ${DRAFT_KV_SCORE_SOURCE}, reuse EMA: ${DRAFT_KV_REUSE_EMA}"
echo "Write log: ${WRITE_LOG_STATUS}"
if [ "${LYAPUNOV_ADAPTIVE_STATUS}" = "enabled" ]; then
  echo "Lyapunov target=${LYAPUNOV_ACCEPTANCE_TARGET}, V=${LYAPUNOV_V}, switch_cost=${LYAPUNOV_SWITCH_COST}, layer_penalty=${LYAPUNOV_LAYER_PENALTY_WEIGHT}"
fi
if [ "${FINAL_ADAPTIVE_STATUS}" = "enabled" ]; then
  echo "Final target_mean=${FINAL_TARGET_MEAN_ACCEPTED}, bad_mean=${FINAL_BAD_MEAN_ACCEPTED}, soft_skip=${FINAL_SOFT_MAX_SKIP_LAYERS}, hard_skip=${FINAL_HARD_MAX_SKIP_LAYERS}"
fi
if [ "${FINAL2_ADAPTIVE_STATUS}" = "enabled" ]; then
  echo "Final2 low_std_k=${FINAL2_LOW_STD_K}, high_std_k=${FINAL2_HIGH_STD_K}, min_config_obs=${FINAL2_MIN_CONFIG_OBSERVATIONS}"
fi
if [ "${ADAPTIVE_COLD_START_STATUS}" = "enabled" ]; then
  echo "Cold start mode=${ADAPTIVE_COLD_START_MODE}, configs=${ADAPTIVE_COLD_START_MAX_CONFIGS}, max_new_tokens=${ADAPTIVE_COLD_START_MAX_NEW_TOKENS}, effective_count=${ADAPTIVE_COLD_START_EFFECTIVE_COUNT}"
fi
echo "Output to ${ANSWER_FILE}"
if [ "${WRITE_LOG_STATUS}" = "enabled" ]; then
  echo "Log to ${LOG_FILE}"
else
  echo "Log disabled; set WRITE_LOG=1 to write ${LOG_FILE}"
fi

run_swift_eval() {
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_swift \
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
    --draft-kv-cache-mode "${DRAFT_KV_CACHE_MODE}" \
    --draft-kv-score-source "${DRAFT_KV_SCORE_SOURCE}" \
    --draft-kv-reuse-ema "${DRAFT_KV_REUSE_EMA}" \
    --optimize-with-compressed-draft-kv \
    --cosine-prefill-skip-layers \
    --cosine-skip-mode "${COSINE_SKIP_MODE}" \
    --cosine-attn-alpha "${COSINE_ATTN_ALPHA}" \
    ${COSINE_MAX_SKIP_ARG} \
    --cosine-keep-first-layers "${COSINE_KEEP_FIRST_LAYERS}" \
    --cosine-keep-last-layers "${COSINE_KEEP_LAST_LAYERS}" \
    --cosine-mlp-interval "${COSINE_MLP_INTERVAL}" \
    "${ADAPTIVE_COLD_START_ARGS[@]}" \
    ${LOCAL_ADAPTIVE_ARG}
}

if [ "${WRITE_LOG_STATUS}" = "enabled" ]; then
  run_swift_eval 2>&1 | tee "${LOG_FILE}"
else
  run_swift_eval
fi
