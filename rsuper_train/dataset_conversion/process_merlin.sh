
# Run the Python script, directing real-time stdout and stderr to separate log files
python abdomenatlas_3d.py \
    --src_path /projects/bodymaps/Data/Merlin/merlin_processed_rsuper/ct_symlinks/ \
    --label_path /projects/bodymaps/Data/Merlin/merlin_processed_rsuper/mask_symlinks/ \
    --tgt_path /projects/bodymaps/Data/Merlin/merlin_processed_rsuper/merlin_medformer_public/ --workers 10 \
    --ids /projects/bodymaps/Data/Merlin/merlin_processed_rsuper/pancreas_all_cases.csv \
    --label_yaml label_names.yaml