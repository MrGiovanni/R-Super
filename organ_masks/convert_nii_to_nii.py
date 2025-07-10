# -*- coding: utf-8 -*-
import os
import argparse
import shutil
import pandas as pd
import tqdm
import numpy as np
import nibabel as nib
import csv
import itk

# --- parallelism ---
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import tempfile, time, json
from pathlib import Path

errors_csv='problems.csv'

def clear_temp_dir(temp_dir):
    for file in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            
def clip_image(image, min_val=-1000, max_val=1000):
    image[image > max_val] = max_val
    image[image < min_val] = min_val
    normalized_image = image
    
    return normalized_image

def clip_nifti(img,img_data):
    normalized_data = clip_image(img_data)

    normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)

    normalized_img.set_data_dtype(np.int16)
    normalized_img.get_data_dtype(finalize=True)
    return normalized_img, normalized_data


def fix_cosines_and_reorient_image(input_path, output_path, img):
    try:
        image = itk.imread(input_path, itk.F)
    except Exception as e:
        print(f'An error occurred: {e}')
        print(f'Attempting to fix cosines problem for {input_path}...')
        qform = img.get_qform()
        img.set_qform(qform)
        sform = img.get_sform()
        img.set_sform(sform)
        nib.save(img, input_path)
        image = itk.imread(input_path, itk.F)
        print(f'Cosines problem has been fixed for {input_path}.')

    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float64) # RPS
    filter.SetDesiredCoordinateDirection(itk.GetMatrixFromArray(matrix))
    filter.Update()
    reoriented = filter.GetOutput()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    itk.imwrite(reoriented, output_path)
    

def convert_image_dtype(file_path, target_dtype):
    img = nib.load(file_path)
    data = np.array(img.dataobj)
    
    img = nib.Nifti1Image(data, img.affine, img.header)
    img.set_data_dtype(target_dtype)
    img.get_data_dtype(finalize=True)
    return img


def convert_id(row, base_output_dir):
    nifti_name = row['path_in']
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    
    #Conversion complete. Now we load and post-process the nifti file
    # 1- skip if no HU values below -100
    nii_loaded = nib.load(nifti_name)
    nii_data = nii_loaded.get_fdata()
    min_hu = np.min(nii_data)
    if min_hu > -100:
        print(f"Skipping {nifti_name} as it has no HU values below -100.")
        #add to error list(csv), avoid race condition with other processes
        with open(errors_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([nifti_name, f'Skipping as it has no HU values below -100: {min_hu}'])
        return  # nothing else to do
    
    #2- clip HU values to [-1000, 1000]
    nii_loaded, nii_data = clip_nifti(nii_loaded, nii_data)
    #save
    nib.save(nii_loaded, nifti_name)
    
    #3- fix itk error and reorient
    fix_cosines_and_reorient_image(nifti_name, nifti_name, nii_loaded)
    
    #4- convert to int16
    nii_loaded = convert_image_dtype(nifti_name, np.int16)
    
    #5- save to output directory
    output_path = row['path_out']
    nib.save(nii_loaded, output_path)
    
    #except Exception as e:
    #    print(f"Error processing folder '{root}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM files to NIfTI format.")
    parser.add_argument("--base_output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--base_input_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--parts", type=int, default=1, help="Number of parts in which to split the ids")
    parser.add_argument("--part", type=int, default=0, help="which part to process, starting from 0")
    parser.add_argument("--restart", action="store_true", help="starts generating from scratch.")
    parser.add_argument("--debug", action="store_true", help="Debug mode, will process only 10 cases.")
    parser.add_argument("--workers", type=int,
                    default=max(1, os.cpu_count() // 4),
                    help="Number of parallel workers "
                         "(default ≈ one-fourth of CPU cores)")

    args = parser.parse_args()
    
    os.makedirs(args.base_output_dir, exist_ok=True)
    
    #get the ids and paths
    ids = [f for f in os.listdir(args.base_input_dir) if f.endswith('.nii.gz')]
    ids = (pd.DataFrame({"fname": ids})
          .assign(path_in = lambda d: d.fname.apply(lambda x: Path(args.base_input_dir)/x),
                  path_out = lambda d: d.fname.apply(lambda x: Path(args.base_output_dir)/x)))
    print(f'Number of ids (total): {len(ids)}',flush=True)
    
    #remove cases already processed
    if not args.restart:
        saved_files = {p.name for p in Path(args.base_output_dir).glob("*.nii.gz")}
        ids = ids[~ids.fname.isin(saved_files)]
        print(f'Number of ids (after removing saved): {len(ids)}',flush=True)
    else:
        print('Restarting from scratch, not removing saved files.',flush=True)
    
    
    #split the ids into parts
    if args.parts > 1:
        chunks = np.array_split(ids, args.parts)
        ids = chunks[args.part]
        print(f'Number of ids (after splitting): {len(ids)}',flush=True)
        
        
    if ids.empty:
        raise ValueError("Nothing to process.  Exiting.")
        
        
    #conversion
    # ------------- parallel fan-out ------------- #
    worker_fn = partial(convert_id,
                        base_output_dir=args.base_output_dir)

    n_workers = min(args.workers, mp.cpu_count())
    print(f"Processing {len(ids)} studies with {n_workers} worker(s)…")

    failed = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        fut2row = {pool.submit(worker_fn, row): row
                   for _, row in ids.iterrows()}

        for fut in tqdm.tqdm(as_completed(fut2row),
                             total=len(fut2row), desc="Total"):
            try:
                fut.result()
            except Exception as e:
                r = fut2row[fut]
                failed.append((r["path_out"], str(e)))

    # ------------- log failures ------------- #
    if failed:
        with open(errors_csv, "a", newline="") as fh:
            csv.writer(fh).writerows(failed)
        print(f"\n⚠  {len(failed)} study(ies) failed; see '{errors_csv}'")

    print("✓ All done.")
