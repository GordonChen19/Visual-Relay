#!/usr/bin/env bash
# Run one of the supplied image-pair experiments with grouped reference attention.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
EXPERIMENT=${1:?Usage: run_reference_budget.sh warriors|falling}
CKPT_DIR=${CKPT_DIR:?Set CKPT_DIR to the Wan2.2-I2V-A14B checkpoint directory.}
IMAGE_DIR=${IMAGE_DIR:-$REPO_DIR}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/outputs}
RHO=${RHO:-0.3}
CURVE=${CURVE:-linear}
GPU_COUNT=${GPU_COUNT:-1}
PYTHON=${PYTHON:-python}

case "$EXPERIMENT" in
  warriors) LEFT=Img1.png; RIGHT=Img2.png; ANCHOR=12; SUFFIX_WEIGHT=0.5 ;;
  falling) LEFT=fnr1.png; RIGHT=fnr2.png; ANCHOR=10; SUFFIX_WEIGHT=1 ;;
  *) printf 'Unknown experiment: %s\n' "$EXPERIMENT" >&2; exit 2 ;;
esac

case "$GPU_COUNT" in
  1)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    LAUNCH=("$PYTHON" generate.py)
    DISTRIBUTED=()
    ;;
  2)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
    LAUNCH=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 generate.py)
    DISTRIBUTED=(--ulysses_size 2 --dit_fsdp)
    ;;
  *) printf 'GPU_COUNT must be 1 or 2.\n' >&2; exit 2 ;;
esac

# Resolve user-supplied relative paths before changing into the model directory.
CKPT_DIR=$(cd -- "$CKPT_DIR" && pwd)
IMAGE_DIR=$(cd -- "$IMAGE_DIR" && pwd)
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
cd -- "$REPO_DIR/Wan2.2"

exec "${LAUNCH[@]}" \
  --task i2v-A14B \
  --size '832*480' \
  --frame_num 81 \
  --ckpt_dir "$CKPT_DIR" \
  --image "$IMAGE_DIR/$LEFT" --image_latent_index 0 \
  --image "$IMAGE_DIR/$RIGHT" --image_latent_index "$ANCHOR" \
  --image_condition_paired_anchor_attention \
  --image_condition_paired_stream_weight "$SUFFIX_WEIGHT" \
  --image_condition_paired_suffix_fade_start_step 25 \
  --image_condition_paired_suffix_fade_end_step 30 \
  --image_condition_paired_prefix_reference_budget "$RHO" \
  --image_condition_paired_prefix_reference_curve "$CURVE" \
  --sample_steps 40 \
  --sample_shift 3 \
  --base_seed 4827316 \
  --prompt "$(cat "$SCRIPT_DIR/prompts/$EXPERIMENT.txt")" \
  "${DISTRIBUTED[@]}" \
  --t5_cpu \
  --offload_model True \
  --convert_model_dtype \
  --save_file "$OUTPUT_DIR/$EXPERIMENT-budget$RHO-$CURVE-seed4827316-$STAMP.mp4"
