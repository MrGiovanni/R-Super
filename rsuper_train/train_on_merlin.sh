#!/bin/bash

# Trap Ctrl+C and exit immediately
trap "echo 'Ctrl-C detected. Exiting...'; exit 1" SIGINT

while true; do
    python train_ddp.py --dataset abdomenatlas_ufo --model medformer --dimension 3d --batch_size 4 --unique_name merlin_pancreas \
    --crop_on_tumor --gpu '0,1' --workers 2 --classes_number 42 --load_augmented \
    --pretrain --pretrained /projects/bodymaps/Pedro/foundational/MedFormer/exp/abdomenatlas/atlas3_medformer/fold_0_latest.pth \
    --loss ball_dice_last --dist_url tcp://127.0.0.1:9697 --report_volume_loss_basic 0.1 \
    --save_destination /projects/bodymaps/Pedro/data/AbdomenAtlas3_pancreas_Merlin_MedformerNpzAugmentedBalancedCropper/ \
    --data_root /projects/bodymaps/Data/AbdomenAtlas3.0MedformerNpz/ \
    --UFO_root /projects/bodymaps/Data/Merlin/merlin_processed_rsuper/merlin_medformer_pancreas_npz/ \
    --ucsf_ids /projects/bodymaps/Data/Merlin/merlin_processed_rsuper/pancreas_train.csv \
    --reports /projects/bodymaps/Data/Merlin/Merlin_per_tumor_metadata_with_slices_skip_missing_mask.csv \
    --tumor_classes pancreas \
    --epochs 100 --lr 0.0001 \
    --resume --load /projects/bodymaps/Pedro/foundational/MedFormer/exp/abdomenatlas_ufo_multi_tumor/merlin_pancreas/fold_0_latest.pth

    #here there is not report loss, we just fine-tune

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        break
    else
        echo "Error encountered (exit code $exit_code). Restarting in 20 seconds..."
        sleep 20
    fi
done

