#!/usr/bin/env bash
set -uo pipefail

# Run a sequential, fixed-seed grid over the two paired-attention controls.
# Override either grid with space-separated values, for example:
#   STREAM_WEIGHTS="0.25 0.5" FUTURE_ETAS="0.0 0.2" ./scripts/grid_search_paired_attention.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"

read -r -a STREAM_WEIGHT_GRID <<< "${STREAM_WEIGHTS:-0.0 0.25 0.5 0.75 1.0}"
read -r -a FUTURE_ETA_GRID <<< "${FUTURE_ETAS:-0.0 0.2 0.5 0.75 1.0}"

CKPT_DIR="${CKPT_DIR:-/home/gchen/Wan2.2-I2V-A14B}"
IMG1="${IMG1:-/home/gchen/image_relay/Img1.png}"
IMG2="${IMG2:-/home/gchen/image_relay/Img2.png}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/gchen/image_relay/Wan2.2/grid_outputs}"
BASE_SEED="${BASE_SEED:-4827316}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

PROMPT="${PROMPT:-A continuous 180-degree orbit around two stationary warriors in a moonlit Japanese temple courtyard. The camera starts behind the red-armored samurai facing the black-clad ninja, then orbits smoothly around them and ends behind the black-clad ninja facing the red-armored samurai.}"

mkdir -p "${OUTPUT_DIR}"

total=$((${#STREAM_WEIGHT_GRID[@]} * ${#FUTURE_ETA_GRID[@]}))
run_index=0
failures=0

for stream_weight in "${STREAM_WEIGHT_GRID[@]}"; do
  for future_eta in "${FUTURE_ETA_GRID[@]}"; do
    run_index=$((run_index + 1))
    weight_tag="${stream_weight//./p}"
    eta_tag="${future_eta//./p}"
    run_name="paired_w${weight_tag}_eta${eta_tag}_seed${BASE_SEED}"
    output_file="${OUTPUT_DIR}/${run_name}.mp4"
    log_file="${OUTPUT_DIR}/${run_name}.log"

    if [[ "${SKIP_EXISTING}" == "1" && -s "${output_file}" ]]; then
      echo "[${run_index}/${total}] Skipping existing ${output_file}"
      continue
    fi

    echo "[${run_index}/${total}] stream_weight=${stream_weight} future_eta=${future_eta}"

    if CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      torchrun --standalone --nproc_per_node="${NPROC}" generate.py \
        --task i2v-A14B \
        --size '832*480' \
        --frame_num 81 \
        --ckpt_dir "${CKPT_DIR}" \
        --image "${IMG1}" \
        --image_latent_index 0 \
        --image "${IMG2}" \
        --image_latent_index 10 \
        --image_condition_paired_anchor_attention \
        --image_condition_paired_stream_weight "${stream_weight}" \
        --image_condition_paired_prefix_future_eta "${future_eta}" \
        --image_condition_paired_suffix_fade_start_step 25 \
        --image_condition_paired_suffix_fade_end_step 30 \
        --sample_shift 3 \
        --base_seed "${BASE_SEED}" \
        --prompt "${PROMPT}" \
        --ulysses_size "${NPROC}" \
        --dit_fsdp \
        --t5_cpu \
        --offload_model True \
        --convert_model_dtype \
        --save_file "${output_file}" 2>&1 | tee "${log_file}"; then
      echo "Completed ${output_file}"
    else
      failures=$((failures + 1))
      echo "FAILED: ${run_name}; see ${log_file}" >&2
    fi
  done
done

if ((failures > 0)); then
  echo "Grid finished with ${failures} failed run(s)." >&2
  exit 1
fi

echo "Grid finished successfully: ${total} configuration(s)."
