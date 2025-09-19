bash parallel_inference.sh --pth /mnt/bodymaps/image_only/AbdomenAtlasPro/AbdomenAtlasPro/ \
--outdir /mnt/ccvl15/psalvad2/missing_ucsf_nnunet_4/ \
--checkpoint nnUNetTrainer__nnUNetPlannerResEncL_torchres_isotropic__3d_fullres/ --gpus 0,1,2,3,4 \
--parts 0,1,2,3,4 --num_parts 13 --BDMAP_format \
--ids /home/psalvad2/data/missing_in_discovery_nnunet.csv