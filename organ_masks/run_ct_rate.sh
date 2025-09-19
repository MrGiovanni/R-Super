bash parallel_inference.sh --pth /home/psalvad2/data/ct_rate_symlinks/ \
--outdir /mnt/ccvl15/psalvad2/ct_rate_nnunet_labels/ \
--checkpoint nnUNetTrainer__nnUNetPlannerResEncL_torchres_isotropic__3d_fullres/ --parts 0,1,2,3,4 --gpus 0,1,2,3,4 --num_parts 5