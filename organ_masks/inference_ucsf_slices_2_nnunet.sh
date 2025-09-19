bash parallel_inference.sh --pth /projects/bodymaps/Data/UCSF_with_tumor_slices/UCSF_extra_slice_2/ \
--outdir /projects/bodymaps/Data/UCSF_with_tumor_slices/UCSF_extra_slice_2_nnunet/ \
--checkpoint nnUNetTrainer__nnUNetPlannerResEncL_torchres_isotropic__3d_fullres/ --gpus 0,1 --BDMAP_format