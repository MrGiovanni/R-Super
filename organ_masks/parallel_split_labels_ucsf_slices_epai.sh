#!/usr/bin/env bash
#SBATCH -p cpu                # Partition (queue)
#SBATCH --nodes=1             # Number of nodes for each array task
#SBATCH --ntasks=1            # Number of tasks per array task
#SBATCH --cpus-per-task=30    # Number of CPU cores per array task
#SBATCH --mem=300G            # Memory per node
#SBATCH --gres=gpu:0          # No GPUs needed
#SBATCH -t 100:00:00           # Max time for each array job (HH:MM:SS)
#SBATCH --account=bodymaps
#SBATCH --mail-type=ALL
#SBATCH --array=0-19           # Create an array of 10 jobs, with indices 0..9

# --------------------------------------------------------------------
# Each of the 10 array jobs will run on a separate node (subject to availability).
#   * PART_ID = the array index (0..9)
#   * We pass it to the Python script's --part argument.
#   * Each job uses --workers 30, leveraging the 30 CPU cores on that node.
# --------------------------------------------------------------------

# Initialize Conda (adjust the path if needed)
source /projects/bodymaps/Pedro/anaconda3/etc/profile.d/conda.sh

# Activate your environment
module unload python39
conda activate former

# Navigate to your code directory
cd /projects/bodymaps/Pedro/R-Super_public/organ_masks/
mkdir -p  slurm_logs

# Capture the Slurm array index for this job
PART_ID=$SLURM_ARRAY_TASK_ID

# Run the Python script, directing real-time stdout and stderr to separate log files

python split_labels_epai.py \
--input_nnunet_gt_path /projects/bodymaps/Data/UCSF_with_tumor_slices/epai_masks_UCSF_slices_batch_5/ \
--output_bdmap_gt_path /projects/bodymaps/Data/UCSF_with_tumor_slices/epai_masks_UCSF_slices_batch_5_split/ \
--parts 20 --part ${PART_ID} > slurm_logs/epai_split_labels_part_${PART_ID}.out \
    2> slurm_logs/epai_split_labels_part_${PART_ID}.err

