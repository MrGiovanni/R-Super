import SimpleITK as sitk
import numpy as np
import os
import shutil
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
import yaml

lab_name_list = []

def pad(img, lab):
    z, y, x = img.shape
    # pad if the image size is smaller than training size
    if z < 128:
        diff = int(math.ceil((128. - z) / 2)) 
        img = np.pad(img, ((diff, diff), (0, 0), (0, 0)))
        lab = np.pad(lab, ((0, 0), (diff, diff), (0, 0), (0, 0)))
    if y < 128:
        diff = int(math.ceil((128. - y) / 2)) 
        img = np.pad(img, ((0, 0), (diff, diff), (0, 0)))
        lab = np.pad(lab, ((0, 0), (0, 0), (diff, diff), (0, 0)))
    if x < 128:
        diff = int(math.ceil((128. - x) / 2)) 
        img = np.pad(img, ((0, 0), (0, 0), (diff, diff)))
        lab = np.pad(lab, ((0, 0), (0, 0), (0, 0), (diff, diff)))

    return img, lab


def process_file(file_info):
    try:
        global lab_name_list
        name, source_path, target_path, modality = file_info
        img = sitk.ReadImage(os.path.join(source_path, name))
        img = sitk.GetArrayFromImage(img).astype(np.float32)

        lab = []
        create_bkg = False
        for i,file in enumerate(list(sorted(lab_name_list)),0):
            if file == 'background':
                #check if file exists
                if not os.path.exists(os.path.join(source_path, name.replace('.nii.gz', ''), file+'.nii.gz')):
                    create_bkg = True
                    bkg_index = i
                    continue
            pth = os.path.join(source_path, name.replace('.nii.gz', ''), file+'.nii.gz')
            item = sitk.ReadImage(pth)
            item = sitk.GetArrayFromImage(item).astype(np.int8)
            lab.append(item)
        try:
            lab = np.stack(lab, axis=0)  # Makes label multi-channel
        except:
            print(f"Error processing {name}")
            return None
        
        if create_bkg:
            background =  1 - np.sum(lab, axis=0)
            lab = np.insert(lab, bkg_index, background, axis=0)
                        
        # Clip intensities
        if modality == 'ct':
            img = np.clip(img, -991, 500)
        else:
            percentile_2 = np.percentile(img, 2, axis=None)
            percentile_98 = np.percentile(img, 98, axis=None)
            img = np.clip(img, percentile_2, percentile_98)

        # Normalize image
        mean = np.mean(img)
        std = np.std(img)
        img -= mean
        img /= std

        # Pad image and label
        img, lab = pad(img, lab)

        # Save as .npy files
        img, lab = img.astype(np.float32), lab.astype(np.int8)
        np.savez_compressed(os.path.join(target_path, f"{name.replace('.nii.gz', '')}.npz"), img)
        np.savez_compressed(os.path.join(target_path, f"{name.replace('.nii.gz', '')}_gt.npz"), lab)
        return name  # Return processed file name for logging
    except Exception as e:
        print(f"[error] {name}: {e}")
        return None

# Main function
def main():
    parser = argparse.ArgumentParser(description="Convert Nifti files to NPZ format for MedFormer.")
    parser.add_argument("--src_path", type=str, default="/projects/bodymaps/Data/UFO_27k_medformer/",
                        help="Source path for the Nifti files.")
    parser.add_argument("--tgt_path", type=str, default="/projects/bodymaps/Data/UFO_27k_medformerNpz/",
                        help="Target path for the NPZ files.")
    parser.add_argument("--parts", type=int, default=1,
                        help="Number of parts to split the dataset into (default 1, meaning no split).")
    parser.add_argument("--current_part", type=int, default=0,
                        help="The index (0-based) of the current part to process.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite already processed cases.")
    args = parser.parse_args()

    source_path = args.src_path
    target_path = args.tgt_path
    parts = args.parts
    current_part = args.current_part
    dataset_list = [
        ('abdomenatlas', 'ct'),
    ]

    global lab_name_list
    with open(os.path.join(source_path, 'list', 'label_names.yaml'), 'r', encoding='utf-8') as f:
        lab_name_list = yaml.safe_load(f)

    os.makedirs(os.path.join(target_path), exist_ok=True)

    for dataset, modality in dataset_list:
        names = sorted([name for name in os.listdir(os.path.join(source_path)) if '.nii.gz' in name])
        
        # Filter out files that have already been processed if overwrite is not enabled
        if not args.overwrite:
            names = [
                name for name in names
                if not (
                    os.path.exists(os.path.join(target_path, f"{name.replace('.nii.gz', '')}.npz")) and
                    os.path.exists(os.path.join(target_path, f"{name.replace('.nii.gz', '')}_gt.npz"))
                )
            ]
            print(f"After filtering, {len(names)} cases remain.")
        
        # Split the dataset if requested
        if parts > 1:
            splits = np.array_split(names, parts)
            if current_part < 0 or current_part >= len(splits):
                raise ValueError(f"current_part must be between 0 and {len(splits)-1}")
            names = splits[current_part].tolist()
            print(f"Processing part {current_part+1}/{parts} with {len(names)} cases.")
        else:
            print(f"Processing {len(names)} cases.")

        # After splitting, filter out files that have already been processed
        names = [
            name for name in names
            if not (
                os.path.exists(os.path.join(target_path, f"{name.replace('.nii.gz', '')}.npz")) and
                os.path.exists(os.path.join(target_path, f"{name.replace('.nii.gz', '')}_gt.npz"))
            )
        ]
        print(f"After filtering, {len(names)} cases remain.")

        file_info_list = [(name, source_path, target_path, modality) for name in names]

        # Process in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=8) as executor:  # Adjust `max_workers` as per hardware
            for result in tqdm(executor.map(process_file, file_info_list), total=len(file_info_list), desc=f"Processing {dataset}"):
                pass
    
    os.makedirs(os.path.join(target_path, 'list'), exist_ok=True)
    shutil.copy(os.path.join(source_path, 'list', 'dataset.yaml'), os.path.join(target_path, 'list', 'dataset.yaml'))
    shutil.copy(os.path.join(source_path, 'list', 'label_names.yaml'), os.path.join(target_path, 'list', 'label_names.yaml'))


if __name__ == "__main__":
    main()