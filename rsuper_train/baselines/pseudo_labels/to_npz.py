import numpy as np
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
import yaml

label_name_list = []


def process_file(file_info):
    """
    This function moves the CT scan from the organs path to the target path and creates a npz file 
    with the organ labels and the lesion labels. Inside this npz, we get the lesion labels from the pseudo labels,
    and the organ labels from the organ labels.
    """
    id, args= file_info
    
    
    lesions_from_pseudo = 0
    
    if id in args.radiologist_ids:
        #just subset the fully annotated label
        pth = os.path.join(args.radiologist_path, id+'_gt.npz')
        labels = np.load(pth, allow_pickle=True)['arr_0']
        out_volumes = {}
        for label in sorted(args.label_name_list): #those are all labels
            if label not in args.label_name_radiologist:
                raise ValueError(f'Label {label} not found in radiologist labels')
            l = labels[args.label_name_radiologist.index(label)]
            out_volumes[label] = l
        assert len(out_volumes) == len(args.label_name_list), f'Radiologist annotated: Number of labels {len(out_volumes)} does not match number of labels in list {len(args.label_name_list)}'
    else:
        #get the organs label
        pth = os.path.join(args.organs_path, id+'_gt.npz')
        organs = np.load(pth, allow_pickle=True)['arr_0']
        out_volumes = {}
        for label in sorted(args.label_name_list): #those are all labels
            if label in args.label_name_organs:
                #organ, get label from the organ labels
                organ = organs[args.label_name_organs.index(label)]
                out_volumes[label] = organ
            else:
                assert 'lesion' in label, f'Label {label} not found in organ labels'
                #lesion, get label from pseudo labels
                pth = os.path.join(args.pseudo_path, id, label+'.npz')
                if os.path.exists(pth):
                    lesion = np.load(pth, allow_pickle=True)['mask']
                    out_volumes[label] = lesion   
                    lesions_from_pseudo += 1
                else:
                    #create zero mask
                    lesion = np.zeros_like(organs[0])
                    out_volumes[label] = lesion
                    
        assert len(out_volumes) == len(args.label_name_list), f'Number of labels {len(out_volumes)} does not match number of labels in list {len(args.label_name_list)}'
            
    #create a npz file with the out_volumes
    npz_path = os.path.join(args.tgt_path, id+'_gt.npz')
    #make a numpy tensor
    out_volumes = np.stack([out_volumes[label] for label in args.label_name_list])
    np.savez_compressed(npz_path, out_volumes.astype(np.uint8))

    #copy ct from the organs path
    if id in args.radiologist_ids:
        orig_pth = args.radiologist_path
    else:
        orig_pth = args.organs_path
    
    
    ct_path = os.path.join(orig_pth, id+'.npz')
    if not os.path.exists(ct_path):
        raise ValueError(f'No ct file found for {id} in {orig_pth}')
    if lesions_from_pseudo > 0:
        print(f'Found {lesions_from_pseudo} lesions from pseudo labels for {id}',flush=True)
    shutil.copy(ct_path, os.path.join(args.tgt_path, id+'.npz'))
    
    
    
    #copy ct from 
    return id  # Return processed file name for logging

# Main function
def main():
    parser = argparse.ArgumentParser(description="Convert Nifti files to NPZ format for MedFormer.")
    parser.add_argument("--organs_path", type=str, default="/projects/bodymaps/Data/UFO_27k_medformerNpz/",
                        help="Source path for the Nifti files.")
    parser.add_argument("--radiologist_path", type=str, default="/projects/bodymaps/Pedro/data/Radiologist_annotated_300_UCSF_medformer_npz/",
                        help="Source path for the Nifti files.")
    parser.add_argument("--pseudo_path", type=str, default="/projects/bodymaps/Pedro/data/UCSF_Tumor_Pseudo_Labels/probabilities/PRETRAIN2_single_ch_w0_many_cancers_100_epch_report_refined_npz/",
                        help="Source path for the Nifti files.")
    parser.add_argument("--tgt_path", type=str, default="/projects/bodymaps/Pedro/data/UCSF_Tumor_Pseudo_Labels/probabilities/PRETRAIN2_single_ch_w0_many_cancers_100_epch_report_refined_medformer_npz/",
                        help="Target path for the NPZ files.")
    parser.add_argument("--label_list", type=str, default="/projects/bodymaps/Pedro/foundational/MedFormer/baselines/pseudo_labels/label_names.yaml",
                        help="list of labels for target.")
    parser.add_argument("--parts", type=int, default=1,
                        help="Number of parts to split the dataset into (default 1, meaning no split).")
    parser.add_argument("--current_part", type=int, default=0,
                        help="The index (0-based) of the current part to process.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite already processed cases.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of worker threads for parallel processing.")
    args = parser.parse_args()
    
    target_path = args.tgt_path
    
    
    os.makedirs(os.path.join(target_path), exist_ok=True)
    
    
    #First: get BDMAP IDs in each folder
    organ_ids = [f.replace('.npz', '') for f in os.listdir(args.organs_path) if (f.endswith('.npz') and '_gt' not in f)]
    if len(organ_ids) == 0:
        raise ValueError(f'no npz files found in {args.organs_path}')
    radiologist_ids = [f.replace('.npz', '') for f in os.listdir(args.radiologist_path) if (f.endswith('.npz') and '_gt' not in f)]
    if len(radiologist_ids) == 0:
        raise ValueError(f'no npz files found in {args.radiologist_path}')
    pseudo_ids = [f for f in os.listdir(args.pseudo_path) if 'BDMAP' in f]
    #remove radiologist_ids from pseudo_ids and organs_ids
    pseudo_ids = [f for f in pseudo_ids if f not in radiologist_ids]
    organ_ids = [f for f in organ_ids if f not in radiologist_ids]
    
    args.pseudo_ids = pseudo_ids
    args.radiologist_ids = radiologist_ids
    
    #get label list
    with open(os.path.join(args.label_list), 'r', encoding='utf-8') as f:
        label_name_list = yaml.safe_load(f)
    #sort
    label_name_list = sorted(label_name_list)
        
    #get the organ labels
    with open(os.path.join(args.organs_path, 'list', 'label_names.yaml'), 'r', encoding='utf-8') as f:
        label_name_organs = yaml.safe_load(f)
    #sort
    label_name_organs = sorted(label_name_organs)
    
    #get the labels for radiologist
    with open(os.path.join(args.radiologist_path, 'list', 'label_names.yaml'), 'r', encoding='utf-8') as f:
        label_name_radiologist = yaml.safe_load(f)
    #sort
    label_name_radiologist = sorted(label_name_radiologist)
        
        
    os.makedirs(os.path.join(args.tgt_path, 'list'), exist_ok=True)
    #save the labels
    with open(os.path.join(args.tgt_path, 'list', 'label_names.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(label_name_list, f)
    #save the dataset
    ids = radiologist_ids+pseudo_ids
    print(f'Number of ids: {len(ids)}')

    
    parts = args.parts
    current_part = args.current_part
    dataset_list = [
        ('abdomenatlas', 'ct'),
    ]

    args.label_name_list = label_name_list
    args.label_name_organs = label_name_organs
    args.label_name_radiologist = label_name_radiologist

    

    for dataset, modality in dataset_list:
        # Filter out files that have already been processed if overwrite is not enabled
        if not args.overwrite:
            ids = [
                id for id in ids
                if not (
                    os.path.exists(os.path.join(target_path, f"{id}.npz")) and
                    os.path.exists(os.path.join(target_path, f"{id}_gt.npz"))
                )
            ]
            print(f"After filtering, {len(ids)} cases remain.")
        
        # Split the dataset if requested
        if parts > 1:
            splits = np.array_split(ids, parts)
            if current_part < 0 or current_part >= len(splits):
                raise ValueError(f"current_part must be between 0 and {len(splits)-1}")
            ids = splits[current_part].tolist()
            print(f"Processing part {current_part+1}/{parts} with {len(ids)} cases.")
        else:
            print(f"Processing {len(ids)} cases.")

        file_info_list = [(id, args) for id in ids]

        # Process in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as executor:  # Adjust `max_workers` as per hardware
            for result in tqdm(executor.map(process_file, file_info_list), total=len(file_info_list), desc=f"Processing {dataset}"):
                pass
        
        saved_ids = [f.replace('_gt.npz', '') for f in os.listdir(target_path) if (f.endswith('_gt.npz'))]
        with open(os.path.join(target_path, 'list', 'dataset.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(saved_ids, f)

if __name__ == "__main__":
    main()