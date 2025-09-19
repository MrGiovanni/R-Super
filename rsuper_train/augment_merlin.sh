#!/bin/bash
# run_parallel.sh

# Trap Ctrl+C (SIGINT) and kill all background processes
trap "echo 'Caught Ctrl+C. Exiting...'; kill 0" SIGINT


CMD1="python AugmentEternal.py --dataset atlas_ufo --model medformer --dimension 3d --batch_size 2 \
--crop_on_tumor --workers_overwrite 4 \
--save_destination /projects/bodymaps/Pedro/data/AbdomenAtlas3_pancreas_Merlin_MedformerNpzAugmentedBalancedCropper/ \
--dataset_path /projects/bodymaps/Data/AbdomenAtlas3.0MedformerNpz/ \
--UFO_root /projects/bodymaps/Data/Merlin/merlin_processed_rsuper/merlin_medformer_pancreas_npz/ \
--ucsf_ids /projects/bodymaps/Data/Merlin/merlin_processed_rsuper/pancreas_train.csv \
--reports /projects/bodymaps/Data/Merlin/Merlin_per_tumor_metadata_with_slices_skip_missing_mask.csv \
--tumor_classes pancreas"


# Function to repeatedly run a command:
run_forever () {
  local cmd="$1"
  while true; do
    echo "Starting command: $cmd"
    eval "$cmd"
    echo "Command [$cmd] terminated. Restarting in 10 seconds..."
    sleep 10
  done
}

# Run both commands in background:
run_forever "$CMD1" &
PID1=$!

# Wait for both processes (this script will never exit on its own)
wait $PID1