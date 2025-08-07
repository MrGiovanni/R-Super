#!/usr/bin/env bash
#
# Usage:
#   bash parallel_inference.sh \
#     --pth <path/to/dataset> \
#     --outdir <path/to/output> \
#     --checkpoint <path/to/model> \
#     --gpus <gpu0,gpu1,...> \
#     [--parts <part0,part1,...>] \
#     [--num_parts <total_parts>] \
#     [--ids <ids.csv>] [--BDMAP_format]
#
set -euo pipefail

# ---------- Parse CLI ---------------------------------------------------------
declare -a gpus=() parts=()
ids=""
bdmap_flag=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pth)         pth="$2";           shift 2 ;;
    --outdir)      outdir="$2";        shift 2 ;;
    --checkpoint)  checkpoint="$2";    shift 2 ;;
    --gpus)        IFS=',' read -r -a gpus <<< "$2"; shift 2 ;;
    --parts)       IFS=',' read -r -a parts <<< "$2"; shift 2 ;;
    --num_parts)   num_parts="$2";     shift 2 ;;
    --ids)         ids="$2";           shift 2 ;;
    --BDMAP_format) bdmap_flag=true;   shift 1 ;;  # flag
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# ---------- Validate required args ------------------------------------------
if [[ -z "${pth:-}" || -z "${outdir:-}" || -z "${checkpoint:-}" || "${#gpus[@]}" -eq 0 ]]; then
  cat >&2 <<EOF
Usage: bash $0 \
  --pth <dataset> --outdir <output> --checkpoint <ckpt> \
  --gpus gpu0,gpu1,... \
  [--parts part0,part1,...] [--num_parts N] \
  [--ids <ids.csv>] [--BDMAP_format]
EOF
  exit 1
fi

mkdir -p logs

# ---------- Default parts & num_parts if missing ---------------------------
# default total parts = number of GPUs
if [[ -z "${num_parts:-}" ]]; then
  num_parts=${#gpus[@]}
fi

# default parts = 0,1,2,...,num_parts-1
if [[ "${#parts[@]}" -eq 0 ]]; then
  parts=()
  for ((i=0; i<num_parts; i++)); do
    parts+=("$i")
  done
fi

# ---------- Launch per-part workers ------------------------------------------
gpu_count=${#gpus[@]}

for idx in "${!parts[@]}"; do
  part_id="${parts[$idx]}"
  # round-robin selection of GPU
  gpu="${gpus[$(( idx % gpu_count ))]}"
  logf="logs/part${part_id}_gpu${gpu}.log"
  echo "Launching part $part_id / $num_parts on GPU $gpu → $logf"

  cmd=(python PredictSubOrgansnUnet.py
          --num_parts  "$num_parts"
          --part_id    "$part_id"
          --gpu        "$gpu"
          --pth        "$pth"
          --outdir     "$outdir"
          --checkpoint "$checkpoint")

  [[ -n "$ids"           ]] && cmd+=( --ids "$ids" )
  [[ "$bdmap_flag" == true ]] && cmd+=( --BDMAP_format )

  "${cmd[@]}" 2>&1 | tee -a "$logf" &
done

wait
echo "All jobs completed."