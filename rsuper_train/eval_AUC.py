import os
import csv
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import fcntl
from filelock import FileLock  # Added for cross-process locking

def write_csv_row(row, csv_columns, csv_file_path):
    """
    Writes a single row to the CSV file (in append mode) under an exclusive lock.
    The columns in the CSV are ordered alphabetically.

    Parameters:
        row (dict): The row data to write.
        csv_columns (list): The list of keys (columns).
        csv_file_path (str): The target CSV file path.
    """
    lock_file = csv_file_path + ".lock"
    with FileLock(lock_file, timeout=10):
        sorted_columns = sorted(csv_columns)  # Order keys alphabetically
        write_header = not os.path.exists(csv_file_path)
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=sorted_columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

def resample_image(image, original_spacing, target_spacing=(1, 1, 1), order=1):
    """
    Resample the image to the target spacing.

    Parameters:
        image (nibabel.Nifti1Image or np.ndarray): Input image to resample.
        original_spacing (tuple or list): The current spacing in (x, y, z).
        target_spacing (tuple): Target spacing in x, y, z directions.
        order (int): The interpolation order. 0=nearest neighbor, etc.

    Returns:
        (np.ndarray, np.ndarray): Resampled image data, and the resize factor.
    """
    resize_factor = np.array(original_spacing) / np.array(target_spacing)

    # Convert nibabel image to NumPy if needed
    try:
        image = image.get_fdata()
    except AttributeError:
        pass

    resampled_image = ndimage.zoom(image, resize_factor, order=order)
    return resampled_image, resize_factor

def detection(tumor_mask, organ_mask, spacing, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], erode=True):
    """
    Returns the total volume (in voxels) of the tumor mask after morphological operations.

    Parameters:
        tumor_mask (str): File path for the tumor segmentation mask.
        organ_mask (str or None): File path for the organ segmentation mask, or None.
        spacing (tuple): Original spacing in x, y, z directions.
        th (float): Threshold for binarizing tumor_mask, default is 0.5.
        erode (bool): Whether to perform morphological erosion + dilation to denoise.

    Returns:
        float: Sum of voxels (volume) after morphological filtering.
    """
    # Load and threshold the tumor mask
    array = nib.load(tumor_mask).get_fdata()

    # If an organ mask is provided, multiply
    if organ_mask is not None:
        raise ValueError('Organ mask was already applied when saving the predictions, no?')
        organ_array = nib.load(organ_mask).get_fdata()
        array = array * organ_array

    # Resample to 1×1×1 mm
    array, _ = resample_image(array, original_spacing=spacing,
                              target_spacing=(1, 1, 1), order=1)#order 1, linear interpolation, masks not binary

    # Optionally erode + dilate
    if erode:
        m_prob = np.max(array)
        #raise ValueError('Applying erosion to many confidence threshold is too expensive. Easier to just resempla once and apply th after. Erosion needs binary before dilation.')
        volumes={}
        for i in range(len(thresholds)):
            arr = array.copy()
            # Threshold the array
            arr = (arr > thresholds[i])
            original = arr.copy()
            # Erode
            arr = ndimage.binary_erosion(arr, structure=np.ones((3, 3, 3)), iterations=1)
            # Dilate
            arr = ndimage.binary_dilation(arr, structure=np.ones((3, 3, 3)), iterations=2)
            # Multiply
            arr *= original
            v = arr.sum()
            volumes[thresholds[i]] = v
    else:
        volumes={}
        for th in thresholds:
            # Threshold the array
            v = (array > th).astype(np.float32).sum()
            volumes[th] = v
            
        #maximum probability
        m_prob = np.max(array)

    # Return the count of 'True' voxels as volume
    return volumes, m_prob

def get_spacing(ct_scan_path):
    """
    Get the spacing from a CT scan file.

    Parameters:
        ct_scan_path (str): Path to the CT scan file.

    Returns:
        tuple: The spacing in x, y, z directions.
    """
    ct_scan = nib.load(ct_scan_path)
    spacing = ct_scan.header.get_zooms()
    return spacing

def _normalize_id(file_name):
    """
    Convert the file name into the ID stored in the CSV ("BDMAP_ID").
    Matches your logic of replacing '_0000.' with '.' and removing '.nii.gz'.
    """
    return file_name.replace('_0000.', '.').replace('.nii.gz', '')

def _process_single_file(file, outputs_folder, ct_folder):
    """
    Helper function that processes one directory (file) and returns a dictionary for CSV.
    Returns None if it is not a valid directory.

    Parameters:
        file (str): Directory name under 'outputs_folder'.
        outputs_folder (str): Path to the outputs folder.
        ct_folder (str): Path to the original CT scans folder (NIfTI).
        th (float): Threshold for binarizing the tumor mask.

    Returns:
        dict or None: A row of results for CSV, or None if skipped.
    """
    file_path = os.path.join(outputs_folder, file)
    if not os.path.isdir(file_path):
        return None

    # Prepare row. We will have one per threshold, and one excel sheet per threshold.
    row={}
    for i in list(range(1,10)):
        i=i/10
        row[i] = {"BDMAP_ID": _normalize_id(file)}

    # Load the CT to get spacing
    ct_path = os.path.join(ct_folder, file)
    if '.nii.gz' not in ct_path:
        ct_path = ct_path+'.nii.gz'
    if not os.path.exists(ct_path):
        ct_path = ct_path.replace('.nii.gz','/ct.nii.gz')
    spacing = get_spacing(ct_path)

    #get organs with tumors
    folder = os.path.join(outputs_folder, file, 'predictions_raw')
    if not os.path.isdir(folder):
        folder=os.path.join(outputs_folder, file)
    
    # Define the suffixes to check for
    suffixes = ['_lesion.nii.gz', '_pdac.nii.gz', '_pnet.nii.gz', '_cyst.nii.gz']

    # Loop over each file in the folder and check if it matches one of the suffixes
    for f in os.listdir(folder):
        for s in suffixes:
            if f.endswith(s):
                # Extract the organ name by removing the suffix
                organ = f[:-len(s)]
                if not args.any_lesion:
                    tumor_mask_path = os.path.join(folder, f)
                else:
                    tumor_mask_path = os.path.join(folder, 'any_lesion.nii.gz')
                # organ mask already applied.
                organ_mask_path = None
                volumes,max_prob = detection(tumor_mask_path, organ_mask_path, spacing)
                
                # Determine the column name based on the suffix
                if s == '_lesion.nii.gz':
                    col_name = f"{organ} tumor volume predicted"
                elif s == '_pdac.nii.gz':
                    col_name = f"{organ} pdac volume predicted"
                elif s == '_pnet.nii.gz':
                    col_name = f"{organ} pnet volume predicted"
                elif s == '_cyst.nii.gz':
                    col_name = f"{organ} cyst volume predicted"
                col_name_max_prob = col_name.replace('volume predicted', 'maximum probability')
                
                for i in list(range(1,10)):
                    i=i/10
                    row[i][col_name] = volumes[i]
                    row[i][col_name_max_prob] = max_prob
                break  # Found a matching suffix, no need to check further

    return row

def split_into_parts(file_list, n_parts, part_index):
    """
    Splits file_list into n_parts nearly equal parts and returns the sublist corresponding to part_index.
    For the final partition, it returns file_list[start:] to capture all remaining files.
    
    Parameters:
        file_list (list): List of files to split.
        n_parts (int): Number of parts to split into.
        part_index (int): Which part (0-indexed) to return.
        
    Returns:
        list: The sublist for the requested part.
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be a positive integer.")
    if part_index < 0 or part_index >= n_parts:
        raise ValueError("part_index must be between 0 and n_parts-1.")
    
    total_files = len(file_list)
    step = total_files // n_parts
    remainder = total_files % n_parts
    
    # Calculate the start index (each partition before part_index gets an extra one if remainder > part_index)
    start = part_index * step + min(part_index, remainder)
    
    # If it's the last part, return all remaining files
    if part_index == n_parts - 1:
        return file_list[start:]
    else:
        # Otherwise, calculate the end index normally.
        end = start + step + (1 if part_index < remainder else 0)
        return file_list[start:end]

def process_outputs(args, outputs_folder, ct_folder, workers=10, continue_processing=False, cases=None):
    """
    Processes each item in 'outputs_folder' in parallel, appending results to a CSV
    and printing them in real time, but only for those not already in the CSV.
    Gracefully stops on Ctrl+C.

    Parameters:
        outputs_folder (str): Path to the outputs folder (segmentations).
        ct_folder (str): Path to the original CT scans folder (NIfTI).
        th (float): Threshold for binarizing the tumor mask.
        workers (int): Number of parallel processes to use (default=10).
    """
    if not args.any_lesion:
        csv_file_path = os.path.join(outputs_folder, "tumor_detection_results.csv")
    else:
        csv_file_path = os.path.join(outputs_folder, "tumor_detection_results_global_lesion.csv")

    #csv_columns = [
    #    "BDMAP_ID",
    #    "liver tumor volume predicted",
    #    "pancreatic tumor volume predicted",
    #    "kidney tumor volume predicted"
    #]

    # Build a list of all items to process
    all_files = [
        f for f in os.listdir(outputs_folder)
        if ('.nii.gz' in f) or os.path.isdir(os.path.join(outputs_folder, f))
    ]
    
    # Determine which files still need to be processed
    need_header = False
    if os.path.exists(csv_file_path.replace('.csv', f'_th0.1.csv')) and continue_processing:
        # CSV already exists; read existing BDMAP_IDs
        existing_df = pd.read_csv(csv_file_path.replace('.csv', f'_th0.1.csv'))
        existing_ids = set(existing_df["BDMAP_ID"].tolist())

        # Filter out those IDs from our to-process list
        files_to_process = []
        for f in all_files:
            if _normalize_id(f) not in existing_ids:
                files_to_process.append(f)

        # We'll open the CSV in append mode, no header
        open_mode = "a"
        need_header = False
        print(f"Resuming: skipping {len(existing_ids)} already processed items.")
    else:
        # remove csv 
        if os.path.exists(csv_file_path.replace('.csv', f'_th0.1.csv')) and (args.parts==1 or args.part==0):
            for i in range(1,10):
                i=i/10
                os.remove(csv_file_path.replace('.csv', f'_th{str(i)}.csv'))
            print(f"Existing CSV '{csv_file_path}' removed. Starting fresh.")
        files_to_process = all_files
        open_mode = "w"
        need_header = True
        print(f"No existing CSV found. Processing all {len(all_files)} items.")

    if not files_to_process:
        print("No new files to process. Exiting.")
        return

    #remove cases not in cases
    if cases is not None:
        cases=pd.read_csv(cases)
        ids=cases['BDMAP_ID'].to_list()
        print(ids[:10])
        print([f.replace('.nii.gz','').replace('_0000','') for f in files_to_process[:10]])
        
        files_to_process = [f for f in files_to_process if f.replace('.nii.gz','').replace('_0000','') in ids]
        print('Number of cases to process after reading csv:', len(files_to_process))

    if args.parts>1:
        files_to_process = split_into_parts(sorted(files_to_process),args.parts,args.part)

    first_row = _process_single_file(files_to_process[0], outputs_folder, ct_folder)
    csv_columns = sorted(list(first_row[0.1].keys()))

    # Process the rest in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_process_single_file, f, outputs_folder, ct_folder): f
            for f in files_to_process
        }

        try:
            for future in as_completed(future_map):
                file_name = future_map[future]
                try:
                    row = future.result()
                    if row is not None:
                        for i in list(range(1,10)):
                            i=i/10
                            write_csv_row(row[i], csv_columns, csv_file_path.replace('.csv', f'_th{str(i)}.csv'))
                            print(f'th: {i}, ',row)
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            print("Received Ctrl+C! Terminating all workers...")
            raise

        print("CSV file saved at:", csv_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect tumors from segmentation masks, append volumes to CSV, and print in real time."
    )
    parser.add_argument("--outputs_folder", type=str, required=True,
                        help="Path to the outputs folder (segmentations)")
    parser.add_argument("--ct_folder", type=str, required=True,
                        help="Path to the original CT scans folder (NIfTI)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel processes to use (default=10)")
    parser.add_argument("--cases", default=None,
                        help="Path to csv with the cases to evaluate")
    parser.add_argument("--continuing", action='store_true',
                        help="continues processing") 
    parser.add_argument("--parts", type=int, default=1,
                        help="number of parts to split the dataset")      
    parser.add_argument("--part", type=int, default=0,
                        help="part to process")  
    parser.add_argument("--any_lesion", action='store_true',
                        help="use the global aby_lesion class for detection")     
    args = parser.parse_args()

    process_outputs(
        args,
        outputs_folder=args.outputs_folder,
        ct_folder=args.ct_folder,
        workers=args.workers,
        continue_processing = args.continuing,
        cases=args.cases,
    )