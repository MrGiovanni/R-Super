bash parallel_inference.sh --pth /projects/bodymaps/Data/UCSF_with_tumor_slices/UCSF_extra_slice_2/ \
--outdir /projects/bodymaps/Data/UCSF_with_tumor_slices/UCSF_extra_slice_2_epai/ \
--checkpoint /projects/bodymaps/Pedro/R-Super_public/organ_masks/qchen76_2025_0421/nnUNetTrainer__nnUNetPlans__3d_fullres/ --gpus 0,1,2,3 --BDMAP_format