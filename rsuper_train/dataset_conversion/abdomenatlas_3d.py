import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, reorient_image
import os
import random
import yaml
import copy
import numpy as np
import pdb
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
from functools import partial

import uuid
import tempfile
import nibabel as nib
import numpy as np

sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(16)  # Set the number of threads (adjust to your hardware)

def _fix_nifti_affine(in_path: str, out_path: str):
    """Rewrite a NIfTI with an orthonormalized affine so ITK/SimpleITK will accept it."""
    img = nib.load(in_path)
    data = img.get_fdata(dtype=np.float32)  # safe dtype
    aff = img.affine

    # Closest orthonormal rotation via SVD
    U, _, Vt = np.linalg.svd(aff[:3, :3])
    R = U @ Vt
    fixed = aff.copy()
    fixed[:3, :3] = R

    nib.Nifti1Image(data, fixed, img.header).to_filename(out_path)

def read_sitk_with_nib_fallback(path: str, tmp_dir='tmp'):
    """Try sitk.ReadImage; if it fails due to non-orthonormal direction cosines,
    rewrite the file via NiBabel and read the fixed version instead."""
    try:
        return sitk.ReadImage(path)
    except RuntimeError as e:
        msg = str(e).lower()
        if "orthonormal direction cosines" in msg or "orthonormal" in msg:
            # Make a unique temp file
            os.makedirs(tmp_dir, exist_ok=True)
            fixed_path = os.path.join(tmp_dir, f"fixed_{uuid.uuid4().hex}.nii.gz")
            _fix_nifti_affine(path, fixed_path)
            img = sitk.ReadImage(fixed_path)
            # Clean up (best-effort)
            try:
                os.remove(fixed_path)
            except Exception:
                pass
            return img
        raise  # some other error: propagate


def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

    imImage = reorient_image(imImage, 'RAI')
    for key in imLabel.keys():
        imLabel[key] = reorient_image(imLabel[key], 'RAI')
        assert imLabel[key].GetSize() == imImage.GetSize(), f'size mismatch for {key}'

    spacing = imImage.GetSpacing()

    mx = []
    for key in imLabel.keys():
        mx.append(sitk.GetArrayFromImage(imLabel[key]).astype(np.uint8).max())
    mx = np.max(mx)

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))


    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    im_size = re_img_xy.GetSize()
    im_spacing = re_img_xy.GetSpacing()
    re_lab_xy = {}
    for key in imLabel.keys():
        re_lab_xy[key]=ResampleLabelToRef(imLabel[key], re_img_xy, interp=sitk.sitkNearestNeighbor)
        assert re_lab_xy[key].GetSize() == im_size
        assert re_lab_xy[key].GetSpacing() == im_spacing
        
    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = {}
    for key in imLabel.keys():
        re_lab_xyz[key]=ResampleLabelToRef(re_lab_xy[key], re_img_xyz, interp=sitk.sitkNearestNeighbor)
    
    #this crop below makes it super difficult to reshape the predictions back to the original shape. Don't.
    #if np.random.uniform() < 0.25:
    #    pass
    #else:
    #    if mx == 0:
    #        pass
    #    else:
    #        re_img_xyz, re_lab_xyz = CropForeground(re_img_xyz, re_lab_xyz, context_size=[20, 30, 30])

    sitk.WriteImage(re_img_xyz, '%s/%s.nii.gz'%(save_path, name))
    for key in re_lab_xyz.keys():
        os.makedirs('%s/%s'%(save_path, name), exist_ok=True)
        sitk.WriteImage(re_lab_xyz[key], '%s/%s/%s.nii.gz'%(save_path, name, key))

# Define the processing function
def process_case_bdmap_format(name,overwrite=False):
    try:
        # Define paths for the output files
        output_ct_path = os.path.join(tgt_path, f"{name}.nii.gz")
        output_label_dir = os.path.join(tgt_path, name)

        # Check if the output CT and all labels already exist
        if os.path.exists(output_ct_path) and all(
            os.path.exists(os.path.join(output_label_dir, f"{lab_name}.nii.gz")) for lab_name in lab_name_list
        ) and (not overwrite):
            print(f"Skipping {name}: All outputs already exist.")
            return

        # Load the CT image
        img_name = os.path.join(src_path, name, 'ct.nii.gz')
        itk_img = read_sitk_with_nib_fallback(img_name)

        # Prepare the label dictionary
        lab_dict = {}
        for lab_name in lab_name_list:
            pth = os.path.join(label_path, name, 'segmentations', f"{lab_name}.nii.gz")
            if not os.path.exists(pth):
                pth = os.path.join(label_path, name, 'predictions', f"{lab_name}.nii.gz")
            if not os.path.exists(pth):
                print(f"File {pth} does not exist")
                # Create a zero label
                l = sitk.Image(itk_img.GetSize(), sitk.sitkUInt8)
                l.SetSpacing(itk_img.GetSpacing())  # Match spacing
                l.SetOrigin(itk_img.GetOrigin())    # Match origin
                l.SetDirection(itk_img.GetDirection())  # Match orientation
            else:
                l = read_sitk_with_nib_fallback(pth)
            lab_dict[lab_name] = l

        # Resample the image and labels
        ResampleImage(itk_img, lab_dict, tgt_path, name, (1.0, 1.0, 1.0))
        print(f"{name} processed successfully.")

    except Exception as e:
        print(f"Error processing {name}: {e}")


def process_case_nnunet_format(name,overwrite=False):
    lab_name_list=sorted(list(labels_nnunet.keys()))
    name = name.replace("_0000.nii.gz","").replace(".nii.gz","")
    
    try:
        # Define paths for the output files
        output_ct_path = os.path.join(tgt_path, f"{name}.nii.gz")
        output_label_dir = os.path.join(tgt_path, name)

        # Check if the output CT and all labels already exist
        if os.path.exists(output_ct_path) and all(
            os.path.exists(os.path.join(output_label_dir, f"{lab_name}.nii.gz")) for lab_name in lab_name_list
        ) and (not overwrite):
            print(f"Skipping {name}: All outputs already exist.")
            return

        # Load the CT image
        img_name = os.path.join(src_path, name+'_0000.nii.gz')
        if not os.path.exists(img_name):
            img_name = os.path.join(src_path, name+'.nii.gz')
        if not os.path.exists(img_name):
            img_name = os.path.join(src_path, name, 'ct.nii.gz')
        if not os.path.exists(img_name):
            raise ValueError(f"File {img_name} does not exist")
        itk_img = read_sitk_with_nib_fallback(img_name)

        # load the nnunet labels
        pth = os.path.join(label_path, name+'_0000.nii.gz')
        if not os.path.exists(pth):
            pth = os.path.join(label_path, name+'_0000.nii.gz.nii.gz')
        if not os.path.exists(pth):
            pth = os.path.join(label_path, name+'.nii.gz')
        if not os.path.exists(pth):
            raise ValueError(f"File {pth} does not exist")
        
        itk_lab = read_sitk_with_nib_fallback(pth)
        # Prepare the label dictionary
        lab_dict = {}

        for lab_name, value in labels_nnunet.items():
            itk_arr = sitk.GetArrayFromImage(itk_lab)  # Extract array

            #if the label is a list, then we need to create a mask for each element in the list
            if isinstance(value, list):
                m = np.zeros_like(itk_arr)
                for v in value:
                    m += (itk_arr == v).astype(np.uint8)
                itk_arr = (m>0).astype(np.uint8)
            else:
                # Efficient masking using single operation
                itk_arr = (itk_arr == value).astype(np.uint8)

            # Convert back to SimpleITK Image
            l = sitk.GetImageFromArray(itk_arr)

            # Preserve metadata
            l.SetOrigin(itk_lab.GetOrigin())
            l.SetSpacing(itk_lab.GetSpacing())
            l.SetDirection(itk_lab.GetDirection())

            # Store in dictionary
            lab_dict[lab_name] = l
        # Resample the image and labels
        ResampleImage(itk_img, lab_dict, tgt_path, name, (1.0, 1.0, 1.0))
        print(f"{name} processed successfully.")

    except Exception as e:
        print(f"Error processing {name}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process AbdomenAtlas cases for MedFormer.")
    parser.add_argument("--src_path", type=str, default="/projects/bodymaps/Data/AbdomenAtlasPro/",
                        help="Source path for the CT images.")
    parser.add_argument("--label_path", type=str, default="/projects/bodymaps/Data/UFO_OrganSubSegments_nnUNet/subSegments_output/",
                        help="Label path for the segmentation masks.")
    parser.add_argument("--tgt_path", type=str, default="/projects/bodymaps/Data/UFO_27k_medformer/",
                        help="Target path for the processed outputs.")
    parser.add_argument("--parts", type=int, default=1,
                        help="Number of parts to split the dataset into (default 1, meaning no split).")
    parser.add_argument("--current_part", type=int, default=0,
                        help="The index (0-based) of the current part to process.")
    parser.add_argument("--nnunet_labels_used", action="store_true",
                        help="Flag to indicate whether to use nnUNet label formatting.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of workers")
    parser.add_argument("--overwrite", action="store_true",
                        help="Flag to indicate whether to overwrite existing files.")
    parser.add_argument("--add_lesions", type=str, default=None,
                        help="Path to a yaml file with the lesions to add to the classes list here.")
    parser.add_argument("--label_yaml", type=str, default=None,
                        help="Path to a yaml file with the label names.")
    parser.add_argument("--ids", default=None,
                        help="IDS in the dataset. path to a csv with a column BDMAP ID")    
    

    args = parser.parse_args()

    src_path = args.src_path
    label_path = args.label_path
    tgt_path = args.tgt_path
    #cases=pd.read_csv('/projects/bodymaps/Data/UCSF_metadata_filled.csv')['BDMAP ID'].to_list()
    name_list = [file for file in os.listdir(label_path)]
    if args.ids is not None:
        ids=pd.read_csv(args.ids)
        name_list = [f for f in name_list if f.replace('.nii.gz','').replace('.npz','') in ids['BDMAP ID'].to_list()]
    nnunet_labels_used = args.nnunet_labels_used
    print('Number of cases:', len(name_list))
    # If splitting is requested, divide the name_list accordingly.
    if args.parts > 1:
        splits = np.array_split(name_list, args.parts)
        # Ensure current_part is a valid index.
        if args.current_part < 0 or args.current_part >= len(splits):
            raise ValueError(f"current_part must be between 0 and {len(splits)-1}")
        name_list = splits[args.current_part].tolist()
        print(f"Processing part {args.current_part+1}/{args.parts} with {len(name_list)} cases.")
        
    #print([file for file in os.listdir(src_path) if file.endswith('.nii.gz') and not file.startswith('BDMAP_0000')])
    #remove the cases already predicted and saved in the tgt_path
    workers = args.workers
    if args.label_yaml is not None:
        with open(args.label_yaml, 'r') as f:
            lab_name_list = yaml.safe_load(f)
    else:
        #get labels from source
        case_id = name_list[0]
        seg_dir  = os.path.join(label_path, case_id, "segmentations")

        lab_name_list = [
            fn.replace('.nii.gz','')               # strip ".nii.gz"
            for fn in os.listdir(seg_dir)
            if fn.endswith(".nii.gz") and 'background' not in fn
        ]

        if not lab_name_list:
            raise ValueError(f"No .nii.gz label found in {seg_dir}")

        lab_name_list.sort()
        #raise ValueError(f'Labels are: {lab_name_list}')


    if args.add_lesions is not None:
        with open(args.add_lesions, 'r') as f:
            lesions = yaml.load(f, Loader=yaml.SafeLoader)
            lab_name_list.extend(lesions)
        #remove duplicates
        lab_name_list = list(set(lab_name_list))
        #sort
        lab_name_list = sorted(lab_name_list)

    nnunet_labels_saved = {'background': 0,
	 'aorta': 1,
	 'gall_bladder': 2,
	 'kidney_left': 3,
	 'kidney_right': 4,
	 'postcava': 5,
	 'spleen': 6,
	 'stomach': 7,
	 'adrenal_gland_left': 8,
	 'adrenal_gland_right': 9,
	 'bladder': 10,
	 'celiac_trunk': 11,
	 'colon': 12,
	 'duodenum': 13,
	 'esophagus': 14,
	 'femur_left': 15,
	 'femur_right': 16,
	 'hepatic_vessel': 17,
	 'intestine': 18,
	 'lung_left': 19,
	 'lung_right': 20,
	 'portal_vein_and_splenic_vein': 21,
	 'prostate': 22,
	 'rectum': 23,
     'liver_segment_1': 24,
     'liver_segment_2': 25,
     'liver_segment_3': 26,
     'liver_segment_4': 27,
     'liver_segment_5': 28,
     'liver_segment_6': 29,
     'liver_segment_7': 30,
     'liver_segment_8': 31,
     'pancreas_head': 32,
     'pancreas_body': 33,
     'pancreas_tail': 34,
     }
    
    joint_labels_nnunet = {'liver': [f'liver_segment_{i}' for i in range(1, 9)]+['hepatic_vessel'],
                           'pancreas': ['pancreas_head', 'pancreas_body', 'pancreas_tail']}
    
    names_labels_nnunet = set(list(nnunet_labels_saved.keys())+list(joint_labels_nnunet.keys()))-{'background','hepatic_vessel'}

    tmp = {}
    for key in names_labels_nnunet:
        if key in nnunet_labels_saved:
            tmp[key] = nnunet_labels_saved[key]
        elif key in joint_labels_nnunet:
            group_labels = [nnunet_labels_saved[name] for name in joint_labels_nnunet[key]]
            tmp[key] = group_labels
    labels_nnunet = tmp

    #name_list = os.listdir(src_path)

    os.makedirs(tgt_path+"/list/", exist_ok=True)
    with open(tgt_path+"/list/dataset.yaml", "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)
    
    if nnunet_labels_used:
        lab_name_list = sorted(list(labels_nnunet.keys()))
    with open(tgt_path+"/list/label_names.yaml", "w",encoding="utf-8") as f:
        yaml.dump(lab_name_list, f)

    os.chdir(src_path)
    
    if nnunet_labels_used:
        process_case = process_case_nnunet_format
    else:
        process_case = process_case_bdmap_format
        

    process_case_with_overwrite = partial(process_case, overwrite=args.overwrite)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in tqdm(executor.map(process_case_with_overwrite, name_list),
                    total=len(name_list), desc="Processing Cases"):
            pass