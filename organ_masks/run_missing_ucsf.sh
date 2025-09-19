bash parallel_inference.sh --pth /mnt/bodymaps/image_only/AbdomenAtlasPro/AbdomenAtlasPro/ \
--outdir /mnt/ccvl15/psalvad2/missing_ucsf_nnunet/ \
--checkpoint nnUNetTrainer__nnUNetPlannerResEncL_torchres_isotropic__3d_fullres/ --gpus 0,1,2,3,4 \
--parts 0,1,2,3,4 --num_parts 23 --BDMAP_format \
--ids /home/psalvad2/data/UCSF_missing_nnunet_masks_ccvl.csv