bash parallel_inference.sh --pth /mnt/bodymaps/image_only/AbdomenAtlasPro/AbdomenAtlasPro/ \
--outdir /mnt/ccvl15/psalvad2/epai_ct_rate/ \
--checkpoint /home/psalvad2/models/qchen76_2025_0421/nnUNetTrainer__nnUNetPlans__3d_fullres/ --gpus 0,1,2,3 \
--parts 4,5,6,7 --num_parts 8 --BDMAP_format \
--ids /home/psalvad2/data/mapping_ct_rate.csv