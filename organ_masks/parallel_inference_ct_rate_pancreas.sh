#!/usr/bin/env bash
#
# Usage:
#   bash parallel_inference.sh --pth <path/to/dataset> \
#     --outdir <path/to/output> \
#     --checkpoint <path/to/model> \
#     --gpus <gpu0,gpu1,…> \
#     [--parts <p0,p1,…> --num_parts <N>]
#
# Examples:
#   # default: one part per GPU
#   bash parallel_inference.sh \
#     --pth /data/dataset \
#     --outdir /data/output \
#     --checkpoint /models/model.ckpt \
#     --gpus 0,1,2,3
#
#   # explicit parts 0,2,5 of 10
#   bash parallel_inference.sh \
#     --pth /data/dataset \
#     --outdir /data/output \
#     --checkpoint /models/model.ckpt \
#     --gpus 0,1 \
#     --parts 0,2,5 \
#     --num_parts 10
#

set -euo pipefail

#——— Parse arguments —————————————————————————————————————————————
declare -a gpus=()
declare -a parts=()
num_parts_user=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pth)        pth="$2";           shift 2 ;;
    --outdir)     outdir="$2";        shift 2 ;;
    --checkpoint) checkpoint="$2";    shift 2 ;;
    --gpus)       IFS=',' read -r -a gpus <<< "$2"; shift 2 ;;
    --parts)      IFS=',' read -r -a parts <<< "$2"; shift 2 ;;
    --num_parts)  num_parts_user="$2"; shift 2 ;;
    *)
      echo "Usage: $0 --pth <path> --outdir <outdir> --checkpoint <ckpt> --gpus gpu0,gpu1,… [--parts p0,p1,… --num_parts N]" >&2
      exit 1
      ;;
  esac
done

#——— Validate required —————————————————————————————————————————————
if [[ -z "${pth:-}" || -z "${outdir:-}" || -z "${checkpoint:-}" || "${#gpus[@]}" -eq 0 ]]; then
  echo "Missing required argument." >&2
  exit 1
fi

#——— Determine num_parts ——————————————————————————————————————————
if [[ -n "$num_parts_user" ]]; then
  num_parts="$num_parts_user"
else
  num_parts=${#gpus[@]}
fi

#——— Decide explicit vs default mode —————————————————————————————————
explicit=false
if [[ "${#parts[@]}" -gt 0 && -n "$num_parts_user" ]]; then
  explicit=true
fi

#——— Prepare logs directory ——————————————————————————————————————
mkdir -p "logs"

#——— Launch processes ——————————————————————————————————————————
if [[ "$explicit" = true ]]; then
  echo "Running explicit parts: (${parts[*]}) with num_parts=$num_parts"
  for idx in "${!parts[@]}"; do
    part_id="${parts[$idx]}"
    gpu="${gpus[$(( idx % ${#gpus[@]} ))]}"
    logf="logs/part${part_id}_gpu${gpu}.log"
    echo "Part $part_id of $num_parts on GPU $gpu (log: $logf)"
    python PredictSubOrgansnUnet.py \
      --num_parts  "$num_parts" \
      --part_id    "$part_id" \
      --gpu        "$gpu" \
      --pth        "$pth" \
      --outdir     "$outdir" \
      --checkpoint "$checkpoint" \
      --ids /home/psalvad2/data/pancreas_and_some_normals_ct_rate.csv \
      2>&1 | tee -a "$logf" &
  done
else
  echo "Running default mode: one part per GPU (num_parts=$num_parts)"
  for part_id in "${!gpus[@]}"; do
    gpu="${gpus[$part_id]}"
    logf="logs/part${part_id}_gpu${gpu}.log"
    echo "Part $part_id of $num_parts on GPU $gpu (log: $logf)"
    python PredictSubOrgansnUnet.py \
      --num_parts  "$num_parts" \
      --part_id    "$part_id" \
      --gpu        "$gpu" \
      --pth        "$pth" \
      --outdir     "$outdir" \
      --checkpoint "$checkpoint" \
      --ids /home/psalvad2/data/pancreas_and_some_normals_ct_rate.csv \
      2>&1 | tee -a "$logf" &
  done
fi

wait
echo "All jobs completed."run_ct_0