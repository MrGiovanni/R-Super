import builtins
import logging
import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from inference.utils import get_inference
from dataset_conversion.utils import ResampleXYZAxis, ResampleLabelToRef, reorient_image
from torch.utils import data
import scipy.ndimage as ndi

import SimpleITK as sitk
import yaml
import argparse
import time
import math
import sys
import pdb
import warnings
import pandas
from pathlib import Path
import re
import unicodedata

import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

import matplotlib.pyplot as plt

from utils import (
    configure_logger,
    save_configure,
)
warnings.filterwarnings("ignore", category=UserWarning)


def clean_ufo(reports,annotated_tumors=['bladder', 'duodenum','esophagus', 'gallbladder','prostate','spleen','stomach','uterus']):
    """
    This function gets a list of reports and removes cases of no interest:
    - We get the healthy patients
    - We get, for each tumor we have annotated organs, all reports that have known tumor size
    - We remove, for organs that have rignr and left (adrenal glands, kidneys), the reports that have unknown sub-segment (not right or left)
    Then, we print the number of useful cases per tumor
    """
    
    
    interest = {}
    
    for organ in annotated_tumors:
        interest[organ] = reports[reports['Standardized Organ'] == organ]
        interest[organ] = interest[organ][interest[organ]['Tumor Size (mm)'] != 'u']
        interest[organ] = interest[organ][interest[organ]['Tumor Size (mm)'] != 'multiple']
        interest[organ] = interest[organ][interest[organ]['Unknow Tumor Size'] == 'no']
        if organ in ['kidney','adrenal_gland','lung','breast','femur']:
            interest[organ] = interest[organ][interest[organ]['Standardized Location'].str.contains('right') | interest[organ]['Standardized Location'].str.contains('left')]
        print('Number of useful cases for %s: %s'%(organ, interest[organ]['BDMAP_ID'].nunique()))

    #interest['healthy'] = reports[reports['no lesion'] == True]
    #print('Number of healthy cases:', interest['healthy']['BDMAP_ID'].nunique())
    #concat
    tumors_per_type = {}
    for k,v in interest.items():
        tumors_per_type[k]=v['BDMAP_ID'].unique().tolist()
    interest = pd.concat(interest.values())
    interest = interest.drop_duplicates()
    print('Total number of useful cases:', interest['BDMAP_ID'].nunique())
    ids_of_interest = interest['BDMAP_ID'].unique().tolist()
    return interest, ids_of_interest,tumors_per_type

ALIASES = {
    "gall bladder": "gallbladder",
    "gall_bladder": "gallbladder",
    "gall‑bladder": "gallbladder",
}


def canon(name: str) -> str:
    name = unicodedata.normalize("NFKC", name).strip().lower()
    return ALIASES.get(name.replace(" ", "_"), name.replace(" ", "_"))

def restrictive_filtering(
    meta,
    class_list=['adrenal gland', 'bladder', 'colon', 'duodenum',
    'esophagus', 'gallbladder','prostate','spleen','stomach','uterus'],
    single_tumor=False,
    id_col=None,):
    """
    Keep only IDs whose reports show lesions exclusively in organs from
    `class_list`.  Optionally require lesions in exactly one organ.

    Prints how many kept IDs belong to each organ in `class_list`.

    Returns
    -------
    list[str] : BDMAP IDs that satisfy the constraints.
    """
    """
    Keep only IDs whose reports show lesions exclusively in organs from
    `class_list`.  Optionally require lesions in exactly one organ.

    Prints:
      • counts of kept IDs per organ in class_list
      • (up to) 10 kept IDs and which tumour organs they have

    Returns
    -------
    list[str] : BDMAP IDs that satisfy the constraints.
    """
    # 0. Ensure an index column
    if id_col is None:
        if "BDMAP ID" in meta.columns:
            id_col = "BDMAP ID"
        elif "BDMAP_ID" in meta.columns:
            id_col = "BDMAP_ID"
        else:
            raise ValueError(
                "Cannot detect ID column; pass id_col='...' explicitly."
            )
    meta = meta.set_index(id_col, drop=False)

    # 1. Canonicalise allowed organs
    allowed = {canon(o) for o in class_list}

    # 2. Map every '*lesion instances' column -> organ
    col_to_organ: dict[str, str] = {}
    rgx = re.compile(r"number of (.+?) lesions? instances?", re.I)
    for col in meta.columns:
        if "lesion instances" not in col.lower():
            continue
        m = rgx.search(col.lower())
        if m:
            col_to_organ[col] = canon(m.group(1))

    if not col_to_organ:
        raise ValueError("No columns containing 'lesion instances' found.")

    # 3. Row‑wise filtering + per‑organ counter
    kept, id_to_orgs = [], {}
    per_organ = {canon(o): 0 for o in class_list}

    for bid, row in meta.iterrows():
        lesion_orgs = {
            col_to_organ[c]
            for c in col_to_organ
            if row.get(c, 0) > 0
        }

        if not lesion_orgs:                 # no reported tumour
            continue
        if not lesion_orgs.issubset(allowed):
            continue
        if single_tumor and len(lesion_orgs) != 1:
            continue

        kept.append(str(bid))
        id_to_orgs[str(bid)] = lesion_orgs
        for org in lesion_orgs:
            if org in per_organ:
                per_organ[org] += 1

    # 4. Print summary
    print("\n--- restrictive_filtering summary ---")
    for org in class_list:
        print(f"{org}: {per_organ.get(canon(org), 0)} IDs")
    print(f"Total kept IDs: {len(kept)}")

    # 5. Show up to 10 example IDs and their tumour organs
    print("\nSample of kept IDs (≤10):")
    for bid in kept[:10]:
        print(f"  {bid}: {sorted(id_to_orgs[bid])}")
    print("------------------------------------\n")

    return kept

def prediction(model_list, tensor_img, args, tgt_organ=None):
    
    inference = get_inference(args)

    with torch.no_grad():
        D, H, W = tensor_img.shape
        print(f'Shape of the input image: {tensor_img.shape}')
        
        tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
        
        z_len = 800
        if D > z_len:
            num_z_chunks = math.ceil(D / z_len)
            z_chunk_len = math.ceil(D / num_z_chunks)

            
            label_pred_list = []
            raw_pred_list = []
            for i in range(num_z_chunks):
                image_chunk = tensor_img[:, :, i*z_chunk_len: (i+1)*z_chunk_len, :, :]
                _, _, D1, H1, W1 = image_chunk.shape

                print(f'Shape of the chunk path: {image_chunk.shape}')

                tensor_pred = torch.zeros([args.classes, D1, H1, W1])
                for model in model_list:
                    pred = inference(model, image_chunk, args, pancreas=tgt_organ)
                    print('Pred shape:',pred.shape)
                    tensor_pred = tensor_pred.type_as(pred) 
                    pred = pred.squeeze(0) 
                    tensor_pred += pred    
               
                label_pred = tensor_pred>0.5
                label_pred = label_pred.to(torch.uint8)
                raw_pred = tensor_pred.clone().cpu()
                del tensor_pred
                torch.cuda.empty_cache()

                label_pred_list.append(label_pred)
                raw_pred_list.append(raw_pred)
            
            label_pred = torch.cat(label_pred_list, dim=1)
            raw_pred =  torch.cat(raw_pred_list, dim=1)

        else:
            tensor_pred = torch.zeros([args.classes, D, H, W])

            for model in model_list:
                pred = inference(model, tensor_img, args, pancreas=tgt_organ)
                tensor_pred = tensor_pred.type_as(pred) 
                
                if args.dimension == '2d':
                    pred = pred.permute(1, 0, 2, 3)
                else:
                    pred = pred.squeeze(0)
                
                tensor_pred += pred       
           
            #_, label_pred = torch.max(tensor_pred, dim=0)
            label_pred = tensor_pred>0.5
            label_pred = label_pred.to(torch.uint8)
            raw_pred = tensor_pred.clone().cpu()
            del pred
            torch.cuda.empty_cache()


    return label_pred, raw_pred


def pad_to_training_size(tensor_img, args):

    z, y, x = tensor_img.shape
   
    if args.dimension == '3d':
        if z < args.training_size[0]:
            diff = (args.training_size[0]+2 - z) // 2
            tensor_img = F.pad(tensor_img, (diff, diff, 0,0, 0,0))
            z_start = diff
            z_end = diff + z
        else:
            z_start = 0
            z_end = z

        if y < args.training_size[1]:
            diff = (args.training_size[1]+2 - y) // 2
            tensor_img = F.pad(tensor_img, (0,0, diff, diff, 0,0))
            y_start = diff
            y_end = diff + y
        else:
            y_start = 0
            y_end = y

        if x < args.training_size[2]:
            diff = (args.training_size[2]+2 -x) // 2
            tensor_img = F.pad(tensor_img, (0,0, 0,0, diff, diff))
            x_start = diff
            x_end = diff + x
        else:
            x_start = 0
            x_end = x

        return tensor_img, [z_start, z_end, y_start, y_end, x_start, x_end]

    elif args.dimension == '2d':
        
        if y < args.training_size[0]:
            diff = (args.training_size[0]+2 - y) // 2
            tensor_img = F.pad(tensor_img, (0,0, diff, diff, 0,0))
            y_start = diff
            y_end = diff + y
        else:
            y_start = 0
            y_end = y

        if x < args.training_size[1]:
            diff = (args.training_size[1]+2 -x) // 2
            tensor_img = F.pad(tensor_img, (0,0, 0,0, diff, diff))
            x_start = diff
            x_end = diff + x
        else:
            x_start = 0
            x_end = x

        return tensor_img, [y_start, y_end, x_start, x_end]

    else:
        raise ValueError




def unpad_img(tensor_pred, original_idx, args):
    if args.dimension == '3d':
        z_start, z_end, y_start, y_end, x_start, x_end = original_idx
    
        return tensor_pred[z_start:z_end, y_start:y_end, x_start:x_end]
    elif args.dimension == '2d':
        y_start, y_end, x_start, x_end = original_idx

        return tensor_pred[:, y_start:y_end, x_start:x_end]
        
    else:
        raise ValueError


def preprocess(itk_img, target_spacing, args):
    '''
    This function performs preprocessing to make images to be consistent with training, e.g. spacing resample, redirection and etc.
    Args:
        itk_img: the simpleITK image to be predicted
    Return: the preprocessed image tensor
    '''
    
    origin_orientation = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(itk_img.GetDirection())

    imImage = reorient_image(itk_img, 'RAI')
    spacing = list(imImage.GetSpacing())
    if not torch.equal(torch.tensor(spacing), torch.tensor(target_spacing)):
        #raise ValueError(f'Debugging: you should not need to resize if loading from _medformer preprocessed samples, spacing is {itk_img.GetSpacing()}, target spacing is {target_spacing}')
        re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
        re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    else:
        re_img_xyz=imImage
    np_img = sitk.GetArrayFromImage(re_img_xyz).astype(np.float32)
    tensor_img = torch.from_numpy(np_img).cuda().float()
    del np_img

    tensor_img = torch.clip(tensor_img, -991, 500)
    mean = torch.mean(tensor_img)
    std = torch.std(tensor_img)

    tensor_img -= mean
    tensor_img /= std

    tensor_img, original_idx = pad_to_training_size(tensor_img, args)

    return tensor_img, original_idx, origin_orientation, re_img_xyz

def postprocess_non_binary(pred, reoriented_itk_img, original_idx, origin_orientation, target_spacing, classes, args):
    # Remove any squeezing if needed.
    pred = pred.squeeze(0)
    pred_dict = {}
    
    for i in range(pred.shape[0]):
        tensor_pred = pred[i]
        # Unpad to original region.
        tensor_pred = unpad_img(tensor_pred, original_idx, args)

        # If resampling is required, resample using 'trilinear' (continuous)
        if tuple(target_spacing) != reoriented_itk_img.GetSpacing():
            tensor_pred = resample_image_with_gpu(
                            tensor_pred.float(), 
                            target_spacing, 
                            tensor_pred.shape[::-1], 
                            reoriented_itk_img.GetSpacing(), 
                            reoriented_itk_img.GetSize(), 
                            interp='trilinear'
                        )

        # Create a SimpleITK image, preserving the float values.
        itk_pred = sitk.GetImageFromArray(tensor_pred.cpu().numpy().astype(np.float32))
        itk_pred.CopyInformation(reoriented_itk_img)
        itk_pred = reorient_image(itk_pred, origin_orientation)

        pred_dict[classes[i]] = itk_pred

    return pred_dict


def postprocess(pred, reoriented_itk_img, original_idx, origin_orientation, target_spacing, classes, args):
    print(f'Shape of the prediction for postprocessing: {pred.shape}')
    pred = pred.squeeze(0)
    pred_dict = {}
    
    for i in range(pred.shape[0]):
        #skip lesions:
        #do organ first
        if 'lesion' in classes[i]:
            continue
        tensor_pred = pred[i]

        #if 'pancrea' in classes[i]:
        #    print("Sum of organ mask binary:", pred[i].sum(),classes[i])

        tensor_pred = unpad_img(tensor_pred, original_idx, args)

        resized=[]
        if tuple(target_spacing) != reoriented_itk_img.GetSpacing():
            raise ValueError(f'Spacing should be the same, it is {target_spacing} and {reoriented_itk_img.GetSpacing()}')
            tensor_pred=resample_image_with_gpu(tensor_pred.float(), 
            target_spacing, tensor_pred.shape[::-1], reoriented_itk_img.GetSpacing(), 
            reoriented_itk_img.GetSize(), interp='nearest').long()
        
        itk_pred = sitk.GetImageFromArray(tensor_pred.cpu().numpy().astype(np.uint8))
        itk_pred.CopyInformation(reoriented_itk_img)

        itk_pred = reorient_image(itk_pred, origin_orientation)

        np_pred = sitk.GetArrayFromImage(itk_pred)

        lab_arr = np_pred
        lab_arr = lab_arr.astype(np.uint8)
        itk_lab = sitk.GetImageFromArray(lab_arr)
        itk_lab.CopyInformation(itk_pred)
        pred_dict[classes[i]] = itk_lab

    #now do lesions
    for i in range(pred.shape[0]):
        if 'lesion' not in classes[i]:
            continue
        tensor_pred = pred[i]
        tensor_pred = unpad_img(tensor_pred, original_idx, args)

        resized=[]
        if tuple(target_spacing) != reoriented_itk_img.GetSpacing():
            tensor_pred=resample_image_with_gpu(tensor_pred.float(), 
            target_spacing, tensor_pred.shape[::-1], reoriented_itk_img.GetSpacing(), 
            reoriented_itk_img.GetSize(), interp='nearest').long()
        
        itk_pred = sitk.GetImageFromArray(tensor_pred.cpu().numpy().astype(np.uint8))
        itk_pred.CopyInformation(reoriented_itk_img)

        itk_pred = reorient_image(itk_pred, origin_orientation)

        np_pred = sitk.GetArrayFromImage(itk_pred)

        lab_arr = np_pred
        lab_arr = lab_arr.astype(np.uint8)
        itk_lab = sitk.GetImageFromArray(lab_arr)
        itk_lab.CopyInformation(itk_pred)

        if args.organ_mask_on_lesion:
            # remove anything outside of the organ
            organ_name = classes[i].split('_')[0].replace('pancreatic', 'pancreas')
            if organ_name == 'kidney':
                # Combine kidney_right and kidney_left using SimpleITK's Add function.
                organ = sitk.Add(pred_dict['kidney_right'], pred_dict['kidney_left'])
            elif organ_name == 'adrenal':
                organ = sitk.Add(pred_dict['adrenal_gland_right'], pred_dict['adrenal_gland_left'])
            elif organ_name == 'lung':
                organ = sitk.Add(pred_dict['lung_right'], pred_dict['lung_left'])
            elif organ_name == 'uterus':
                organ = pred_dict['prostate']
            elif organ_name == 'gallbladder':
                organ = pred_dict['gall_bladder']
            elif organ_name in ['bone','breast']:
                #we do not have organ masks for these, make a mask of ones
                size = pred_dict['prostate'].GetSize()
                organ = sitk.Image(size, sitk.sitkUInt8)
                organ = organ + 1  # This adds 1 to every voxel, making an image of ones.
            else:
                organ = pred_dict[organ_name]
                
            # Convert organ to NumPy, threshold to binary, and convert back to SimpleITK.
            organ_np = sitk.GetArrayFromImage(organ)
            organ_np = (organ_np > 0).astype(np.uint8)
            organ = sitk.GetImageFromArray(organ_np)
            # Copy spatial information (assuming pred_dict[organ_name] or kidney_right has correct info)
            if organ_name == 'kidney':
                organ.CopyInformation(pred_dict['kidney_right'])
            elif organ_name == 'adrenal':
                organ.CopyInformation(pred_dict['adrenal_gland_right'])
            elif organ_name == 'lung':
                organ.CopyInformation(pred_dict['lung_right'])
            elif organ_name == 'uterus':
                organ.CopyInformation(pred_dict['prostate'])
            elif organ_name == 'gallbladder':
                organ.CopyInformation(pred_dict['gall_bladder'])
            elif organ_name in ['bone','breast']:
                #we do not have organ masks for these, make a mask of ones
                organ.CopyInformation(pred_dict['prostate'])
            else:
                organ.CopyInformation(pred_dict[organ_name])
            
            # Dilate the organ mask using a radius of 3 voxels.
            organ = sitk.BinaryDilate(organ, (3, 3, 3))

            organ = sitk.Cast(organ, sitk.sitkUInt8)
            itk_lab = sitk.Cast(itk_lab, sitk.sitkUInt8)        
            
            # Multiply the lesion label (itk_lab) by the organ mask using SimpleITK.Multiply.
            itk_lab = sitk.Multiply(organ, itk_lab)

        if args.connected_components:
            itk_lab = keep_largest_component(itk_lab)
            #get only largest connected component
        
        pred_dict[classes[i]] = itk_lab

    return pred_dict


def postprocess_non_binary_lesion(pred, reoriented_itk_img, original_idx, origin_orientation, target_spacing, classes, args):
    print(f'Shape of the prediction for postprocessing: {pred.shape}')
    pred = pred.squeeze(0)
    pred_dict = {}
    
    for i in range(pred.shape[0]):
        #skip lesions:
        #do organ first
        if 'lesion' in classes[i]:
            continue

        #if 'pancrea' in classes[i]:
        #    print("Sum of organ mask raw:", pred[i].sum(),classes[i])
        tensor_pred = unpad_img(pred[i], original_idx, args)
        if tuple(target_spacing) != reoriented_itk_img.GetSpacing():
            tensor_pred = resample_image_with_gpu(
                tensor_pred.float(), target_spacing, tensor_pred.shape[::-1],
                reoriented_itk_img.GetSpacing(), reoriented_itk_img.GetSize(),
                interp="nearest")

        itk_pred = sitk.GetImageFromArray(tensor_pred.cpu().numpy())   # scores
        itk_pred.CopyInformation(reoriented_itk_img)
        itk_pred = reorient_image(itk_pred, origin_orientation)

        # hard mask – keeps spacing / origin / direction intact
        itk_lab  = sitk.Cast( itk_pred > 0.5 , sitk.sitkFloat32 )
        #if 'pancrea' in classes[i]:
        #    print("Sum of organ mask:", float(sitk.GetArrayViewFromImage(itk_lab).sum()),classes[i])
        pred_dict[classes[i]] = itk_lab

    #now do lesions
    for i in range(pred.shape[0]):
        if 'lesion' not in classes[i]:
            continue
        tensor_pred = pred[i]
        tensor_pred = unpad_img(tensor_pred, original_idx, args)

        resized=[]
        if tuple(target_spacing) != reoriented_itk_img.GetSpacing():
            tensor_pred=resample_image_with_gpu(tensor_pred.float(), target_spacing, 
            tensor_pred.shape[::-1], reoriented_itk_img.GetSpacing(), 
            reoriented_itk_img.GetSize(), interp='trilinear')
        
        itk_pred = sitk.GetImageFromArray(tensor_pred.cpu().numpy().astype(np.float32))
        itk_pred.CopyInformation(reoriented_itk_img)

        itk_pred = reorient_image(itk_pred, origin_orientation)

        np_pred = sitk.GetArrayFromImage(itk_pred)

        lab_arr = np_pred
        lab_arr = lab_arr.astype(np.float32)
        itk_lab = sitk.GetImageFromArray(lab_arr)
        itk_lab.CopyInformation(itk_pred)
        #print('Sum of lesion mask:',lab_arr.sum(), classes[i])

        if args.organ_mask_on_lesion:
            # remove anything outside of the organ
            organ_name = classes[i].split('_')[0].replace('pancreatic', 'pancreas')
            if organ_name == 'kidney':
                # Combine kidney_right and kidney_left using SimpleITK's Add function.
                organ = sitk.Add(pred_dict['kidney_right'], pred_dict['kidney_left'])
            elif organ_name == 'adrenal':
                organ = sitk.Add(pred_dict['adrenal_gland_right'], pred_dict['adrenal_gland_left'])
            elif organ_name == 'lung':
                organ = sitk.Add(pred_dict['lung_right'], pred_dict['lung_left'])
            elif organ_name == 'uterus':
                organ = pred_dict['prostate']
            elif organ_name == 'gallbladder':
                organ = pred_dict['gall_bladder']
            elif organ_name in ['bone','breast']:
                #we do not have organ masks for these, make a mask of ones
                size = pred_dict['prostate'].GetSize()
                organ = sitk.Image(size, sitk.sitkFloat32)
                organ = organ + 1  # This adds 1 to every voxel, making an image of ones.
            else:
                organ = pred_dict[organ_name]

            #print('Sum of organ mask after getting it:', sitk.GetArrayFromImage(organ).sum(),organ_name)
                
            # Convert organ to NumPy, threshold to binary, and convert back to SimpleITK.
            organ_np = sitk.GetArrayFromImage(organ)
            organ_np = (organ_np > 0).astype(np.float32)
            organ = sitk.GetImageFromArray(organ_np)
            # Copy spatial information (assuming pred_dict[organ_name] or kidney_right has correct info)
            if organ_name == 'kidney':
                organ.CopyInformation(pred_dict['kidney_right'])
            elif organ_name == 'adrenal':
                organ.CopyInformation(pred_dict['adrenal_gland_right'])
            elif organ_name == 'lung':
                organ.CopyInformation(pred_dict['lung_right'])
            elif organ_name == 'uterus':
                organ.CopyInformation(pred_dict['prostate'])
            elif organ_name == 'gallbladder':
                organ.CopyInformation(pred_dict['gall_bladder'])
            elif organ_name in ['bone','breast']:
                #we do not have organ masks for these, make a mask of ones
                organ.CopyInformation(pred_dict['prostate'])
            else:
                organ.CopyInformation(pred_dict[organ_name])
            
            # Dilate the organ mask using a radius of 3 voxels.
            organ  = sitk.Cast( organ > 0.5 , sitk.sitkUInt8 )   # binarise in‑place
            organ  = sitk.BinaryDilate(organ, (3, 3, 3))

            #print('Sum of organ and lesion mask before gating:',sitk.GetArrayFromImage(itk_lab).sum(),sitk.GetArrayFromImage(organ).sum())     

            organ = sitk.Cast(organ, sitk.sitkFloat32)
            itk_lab = sitk.Cast(itk_lab, sitk.sitkFloat32)   

            
            #print('Sum of organ and lesion mask before gating f32:',sitk.GetArrayFromImage(itk_lab).sum(),sitk.GetArrayFromImage(organ).sum())     
            
            # Multiply the lesion label (itk_lab) by the organ mask using SimpleITK.Multiply.
            itk_lab = sitk.Multiply(organ, itk_lab)

        
        print('Sum of gated lesion mask:',sitk.GetArrayFromImage(itk_lab).sum())

        itk_lab = sitk.Cast(itk_lab, sitk.sitkFloat32)   
        
        pred_dict[classes[i]] = itk_lab

    return pred_dict

def postprocess_npz(pred, classes, args):
    print(f'Shape of the prediction for postprocessing: {pred.shape}')
    pred = pred.squeeze(0)
    pred_dict = {}
    
    for i in range(pred.shape[0]):
        #skip lesions:
        #do organ first
        if 'lesion' in classes[i]:
            continue
        tensor_pred = pred[i]
        np_pred = tensor_pred.cpu().numpy()
        pred_dict[classes[i]] = np_pred

    #now do lesions
    for i in range(pred.shape[0]):
        if 'lesion' not in classes[i]:
            continue
        np_pred = pred[i].cpu().numpy()

        if args.organ_mask_on_lesion:
            # remove anything outside of the organ
            organ_name = classes[i].split('_')[0].replace('pancreatic', 'pancreas')
            if organ_name == 'kidney':
                # Combine kidney_right and kidney_left using SimpleITK's Add function.
                organ = pred_dict['kidney_right']+pred_dict['kidney_left']
            elif organ_name == 'adrenal':
                organ = pred_dict['adrenal_gland_right']+pred_dict['adrenal_gland_left']
            elif organ_name == 'lung':
                organ = pred_dict['lung_right']+pred_dict['lung_left']
            elif organ_name == 'uterus':
                organ = pred_dict['prostate']
            elif organ_name == 'gallbladder':
                organ = pred_dict['gall_bladder']
            elif organ_name in ['bone','breast']:
                #we do not have organ masks for these, make a mask of ones
                organ = np.ones_like(pred_dict['prostate'], dtype=np.uint8)
            else:
                organ = pred_dict[organ_name]
                
            #threshold to binary
            organ = (organ > 0.5).astype(np.uint8)
            
            # Dilate the organ mask using a radius of 3 voxels in numpy
            organ = ndi.binary_dilation(organ, structure=np.ones((3, 3, 3)))
            
            #type as np_pred
            organ = organ.astype(np_pred.dtype)
            
            np_pred = organ * np_pred
        
        pred_dict[classes[i]] = np_pred

    return pred_dict

def keep_largest_component(label_map):
    # Convert the label map to a binary image
    binary_image = label_map > 0
    
    # Run connected component analysis
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_image = cc_filter.Execute(binary_image)
    
    # Get the number of connected components
    num_components = cc_filter.GetObjectCount()
    
    # Find the largest connected component
    largest_component_label = 0
    largest_component_size = 0
    for i in range(1, num_components + 1):
        component_mask = sitk.Equal(cc_image, i)
        component_size = sitk.GetArrayFromImage(component_mask).sum()
        if component_size > largest_component_size:
            largest_component_label = i
            largest_component_size = component_size
    
    # Create a new label map with only the largest component
    largest_component_map = sitk.Equal(cc_image, largest_component_label)
    
    return largest_component_map

def resample_image_with_gpu(tensor_img, old_spacing=(2., 2., 2.), old_size=(512, 512, 512), new_spacing=(1., 1., 1.), new_size=None, interp='trilinear'):
    # space order: x, y, z. numpy/pytorch tensor order z, y, x
    
    new_spacing = np.array(new_spacing)[::-1] # -> z, y, x

    tensor_img = tensor_img.unsqueeze(0).unsqueeze(0) # -> b, c, z, y, x

    old_spacing = np.array(old_spacing)[::-1]
    old_size = np.array(old_size, dtype=np.float32)[::-1] # -> z, y, x
    
    if new_size == None:
        new_size = old_size * (old_spacing / new_spacing)
        new_size = new_size.round().astype(int).tolist()
    else:
        new_size = np.array(new_size)[::-1].tolist() # -> z, y, x
    
    if interp in  ['linear', 'bilinear', 'bicubic', 'trilinear']:
        resampled_tensor_img = F.interpolate(tensor_img, size=new_size, mode=interp, align_corners=True)
    else:
        resampled_tensor_img = F.interpolate(tensor_img, size=new_size, mode=interp)
    resampled_tensor_img = resampled_tensor_img.squeeze(0).squeeze(0)
    
    torch.cuda.empty_cache()

    return resampled_tensor_img

def init_model(args,classes,old_classes=None):
    #checkpoint = torch.load(args.load)
    #net.load_state_dict(checkpoint['model_state_dict'])
    #args.start_epoch = checkpoint['epoch']
    print(f"Number of classes: {len(classes)}")
    if old_classes is not None:
        print(f"Number of old classes: {len(old_classes)}")
    if args.update_output_layer:
        c = old_classes # we must load the checkpoint with the old classes
    else:
        c = classes

    model_list = []
    for ckp_path in args.load:
        print('Number of classes for model loading: ', len(c))
        model = get_model(args,classes=c)
        if not args.EMA:
            pth = torch.load(ckp_path, map_location=torch.device('cpu'))['model_state_dict']
        else:
            pth = torch.load(ckp_path, map_location=torch.device('cpu'))['ema_model_state_dict']
        try:
            pth = pth.state_dict()
        except:
            pass
        model.load_state_dict(pth,strict=False)
        model.cuda()
        model.eval()
        model_list.append(model)
        print(f"Model loaded from {ckp_path}")

    if args.update_output_layer:
        from model.dim3.medformer import update_output_layer_onk
        model=update_output_layer_onk(model, original_classes=old_classes, new_classes=classes)
        #cuda
        model.cuda()
        model.eval()
        #print shape of last conv layer
        

    return model_list

def _nii_stem(fn: str) -> str:
    # convert 'adrenal_gland_left.nii.gz' -> 'adrenal_gland_left'
    return fn[:-7] if fn.endswith('.nii.gz') else os.path.splitext(fn)[0]

def nib_load(path_to_nii):
    """
    Attempt to load a NIfTI file with nibabel in a way that more closely
    matches SimpleITK's default LPS orientation from sitk.ReadImage().
    
    Returns:
        tmp_itk_img (SimpleITK.Image)
    """
    # We import nibabel here to avoid "UnboundLocalError"
    import nibabel as nb
    from nibabel.orientations import (
        io_orientation,
        axcodes2ornt,
        ornt_transform,
        apply_orientation
    )

    # 1) Load with nibabel
    nib_img = nb.load(path_to_nii)
    affine = nib_img.affine
    np_array = nib_img.get_fdata(dtype=np.float32)  # shape: (x, y, z) typically

    # 2) Determine the orientation of the current image
    orig_ornt = io_orientation(affine)          # e.g. RAS, LAS, etc.
    lps_ornt  = axcodes2ornt(("L", "P", "S"))    # LPS convention

    # 3) Compute the transform that takes us from the original orientation to LPS
    trans_ornt = ornt_transform(orig_ornt, lps_ornt)

    # 4) Reorient the data array to LPS axis ordering
    np_array_lps = apply_orientation(np_array, trans_ornt)
    # apply_orientation() only reorders the array (axes / flips); 
    # it does NOT return a new affine in nibabel’s current versions.

    # 5) By default, SimpleITK’s GetArrayFromImage() returns data in (z, y, x).
    #    Our LPS array here is in (L, P, S) which typically aligns with (x, y, z).
    #    We can transpose to get (z, y, x) if your code expects that ordering:
    np_array_zyx = np.transpose(np_array_lps, (2, 1, 0))  # now shape: (z, y, x)

    # 6) Make a SimpleITK image from that array
    tmp_itk_img = sitk.GetImageFromArray(np_array_zyx)

    return tmp_itk_img

def get_parser():

    def parse_spacing_list(string):
        return tuple([float(spacing) for spacing in string.split(',')])
    def parse_model_list(string):
        return string.split(',')
    parser = argparse.ArgumentParser(description='CBIM Medical Image Segmentation')
    parser.add_argument('--parts', type=int, default=1, help='For running multiple instances of this script. Parts divides the dataset, and this script runs on current_part')
    parser.add_argument('--current_part', type=int, default=0, help='For running multiple instances of this script. Parts divides the dataset, and this script runs on current_part')
    parser.add_argument('--dataset', type=str, default='abdomenatlas', help='dataset name')
    parser.add_argument('--model', type=str, default='medformer', help='model name')
    parser.add_argument('--dimension', type=str, default='3d', help='2d model or 3d model')

    parser.add_argument('--load', type=parse_model_list, default='./exp/abdomenatlas/former_batch4_pth128/fold_0_latest.pth', help='the path of trained model checkpoint. Use \',\' as the separator if load multiple checkpoints for ensemble')
    parser.add_argument('--img_path', type=str, default='/projects/bodymaps/Data/Dataset244_smallAtlasUCSF/imagesTr/', help='the path of the directory of images to be predicted')
    parser.add_argument('--save_path', type=str, default='./result/UFO/', help='the path to save predicted label')
    parser.add_argument('--learnable_loss_weights', action='store_true', help='Allows learnable loss weigths (https://arxiv.org/pdf/1705.07115).')  

    parser.add_argument('--ids', type=str, default=None, help='ids of testing samples')
    
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--class_list', type=str, default='/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml')
    parser.add_argument('--connected_components', action='store_true', help='whether to keep the largest connected component')
    parser.add_argument('--organ_mask_on_lesion', action='store_true', help='whether to keep the largest connected component')
    parser.add_argument('--classification_branch', action='store_true', help='whether to use the classification branch')
    parser.add_argument('--cls_gate', action='store_true', help='multiplies the segmentation sigmoid output by the classification sigmoid output--gate')
    parser.add_argument('--cls_gate_norm', action='store_true', help='before applying the the cls gate, the segmentation output is normalized, making its maximum value above 0.5 become 1')
    parser.add_argument('--update_output_layer', action='store_true', help='update the output layer to have the same number of classes as the number of classes in the class_list')
    parser.add_argument('--old_classes', type=str, default=None, help='old classes, we will keep weights/kernels of the old classes. This parameter should be a location of a yaml file with the old classes, we will sort them!')
    parser.add_argument('--mtl', type=str, default=None, help='multi-task learning method. If None, no MTL. Uses method from https://github.com/SamsungLabs/MTL/')
    parser.add_argument('--save_probabilities', action='store_true', help='saves probabilities')
    parser.add_argument('--not_save_binary', action='store_true', help='does not save binary')
    parser.add_argument('--save_probabilities_lesions', action='store_true', help='saves probabilities only for lesion classes')
    parser.add_argument('--save_probabilities_report_tumors_only', action='store_true', help='saves probabilities')
    parser.add_argument('--overwrite', action='store_true', help='overwrites last saved results')
    parser.add_argument('--save_pancreas_lesion_only', action='store_true', help='overwrites last saved results')
    parser.add_argument('--predict_pancreas_only', action='store_true', help='overwrites last saved results')
    parser.add_argument('--epai_stage_2', action='store_true', help='only for testing epai stage 2')
    #meta
    parser.add_argument('--meta', type=str, default='/home/psalvad2/data/UCSF_metadata_filled.csv', help='meta from reports')
    parser.add_argument('--reports', type=str, default='/home/psalvad2/data/UCSFLLMOutputLarge27k.csv', help='meta from reports')
    parser.add_argument('--filter_cases_ufo', action='store_true', help='predict only cases of interest (no missing size)')
    parser.add_argument('--restrictive_filter', action='store_true', help='only consider cases that have no tumor outside of a given class list')
    parser.add_argument('--restrictive_filter_one_organ', action='store_true', help='only consider cases that tumors in a single organ')

    parser.add_argument('--aggregator_mode', type=str, default='concat', help='mode for the aggregator')
    parser.add_argument('--cls_on_output', action='store_true', help='if true, the classification branch is on the output of the model, otherwise it is on the bottleneck')

    #extra classifiers on top of the segmentation output
    parser.add_argument('--attenuation_classifier', type=str, default='none')
    parser.add_argument('--train_att_MLP_on_mask_only', action='store_true', help='if true, the attenuation classifier MLP is trained only on the mask (segmentation) output. Otherwise, it is trained on mask and model outputs.')
    parser.add_argument('--att_weight', type=float, default=0.01, help='weight for the tumor attenuation loss')
    parser.add_argument('--tumor_classifier', action='store_true', help='if true, adds a tumor classifier on top of the segmentation output. The classifier classifies tumor number and diameters.')
    parser.add_argument('--cls_weight', type=float, default=0.01, help='weight for the tumor classifier loss')
    parser.add_argument('--organs_with_tumor', type=str, nargs='+', default=['bladder','gall_bladder','esophagus','duodenum','stomach','adrenal_gland_right','adrenal_gland_left','prostate','spleen'], help='organs that may have tumors, used in the two pass inference')
    parser.add_argument('--disable_inference_2_stages', action='store_true', help='if not set, we run one normal inference pass, then a second pass cropping on all organs that may have tumors')
    parser.add_argument('--EMA', action='store_true', help='If set, we test the EMA model')
    
    #parser.add_argument('--class_list', type=str, default='/projects/bodymaps/Pedro/data/atlas_300_medformer_multi_ch_tumor_npy/list/label_names.yaml')

    

    args = parser.parse_args()


    args.clip_loss = False
    args.load_clip = False

    #latest folder of load
    if isinstance(args.load, list):
        model_name = os.path.basename(os.path.dirname(args.load[0]))  # Gets the last folder name
    else:
        model_name = os.path.basename(os.path.dirname(args.load))  # Handles single path case

    args.save_path = os.path.join(args.save_path, args.dataset, model_name)
    os.makedirs(args.save_path, exist_ok=True)
    print('Save path: %s'%args.save_path)

    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    args.inference_2_stages=True
    if args.disable_inference_2_stages:
        args.inference_2_stages=False
        print('Inference 2 stages disabled! (Why?)')
    

    return args
def filter_already_predicted(ids, save_path, class_list, overwrite):
    print('Filtering predicted')
    if overwrite:
        return ids

    def _nii_stem(fn: str) -> str:
        # convert 'adrenal_gland_left.nii.gz' -> 'adrenal_gland_left'
        return fn[:-7] if fn.endswith('.nii.gz') else os.path.splitext(fn)[0]

    kept = []
    for img_name in ids:
        case_id = img_name.replace('/ct.nii.gz','').replace('.nii.gz','').replace('.npz','')
        pred_dir = os.path.join(save_path, case_id, 'predictions')

        if not os.path.isdir(pred_dir):
            kept.append(img_name)
            continue

        # Collect all .nii.gz stems
        stems = set()
        with os.scandir(pred_dir) as it:
            for entry in it:
                if entry.name.endswith('.nii.gz'):
                    stems.add(_nii_stem(entry.name))

        # Require every class to be present as a .nii.gz
        if set(class_list).issubset(stems):
            print(f"Skipping {img_name}: found all {len(class_list)} classes (.nii.gz only)")
        else:
            kept.append(img_name)

    print(f"Ids after filtering: {len(kept)} / {len(ids)}")
    return kept

if __name__ == '__main__':
    
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.sliding_window = True
    args.window_size = args.training_size

    if args.ids is None:
        ids = [i for i in os.listdir(args.img_path) if 'gt.' not in i]
    else:
        if 'BDMAP_ID' in pandas.read_csv(args.ids).columns:
            col = 'BDMAP_ID'
        else:
            col = 'BDMAP ID'
        ex = pandas.read_csv(args.ids)[col].tolist()[0]
        if os.path.exists(args.img_path+'/'+ex+'.npz'):
            ids = [f+'.npz' for f in pandas.read_csv(args.ids)[col].tolist()]
        elif os.path.exists(args.img_path+'/'+ex+'/ct.nii.gz'):
            ids = [f+'/ct.nii.gz' for f in pandas.read_csv(args.ids)[col].tolist()]
        elif os.path.exists(args.img_path+'/'+ex+'.nii.gz'):
            ids = [f+'.nii.gz' for f in pandas.read_csv(args.ids)[col].tolist()]
        else:
            raise ValueError('Path not found')

    #remove cases not in data folder:
    valid_ids = []
    removed = []
    for img_name in ids:
        full_path = os.path.join(args.img_path, img_name)
        if os.path.exists(full_path):
            valid_ids.append(img_name)
        else:
            logging.warning(f"Skipping '{img_name}': not found at {full_path}")
            removed.append(img_name)
    ids = valid_ids
    print(f'After removing missing cases, {len(ids)} ids remain. Number of removed files: {len(removed)}')

    if args.filter_cases_ufo:
        reports = pd.read_csv(args.reports)
        _,ids_of_interest,_ = clean_ufo(reports)
        print('Number of IDs before filtering IDs of interest:', len(ids))
        print(f'Examples of IDs of interest: {ids_of_interest[:10]}')
        print(f'Examples of our IDs: {ids[:10]}')
        ids = [i for i in ids if i[:len('BDMAP_00032584')] in ids_of_interest]
        print('Number of IDs after filtering IDs of interest:', len(ids))
        
    if args.restrictive_filter:
        meta = pd.read_csv(args.meta)
        ids_of_interest = restrictive_filtering(meta, single_tumor=args.restrictive_filter_one_organ)
        print('restrictive: number of IDs of interest:', len(ids_of_interest))
        print('restrictive: Number of IDs before filtering IDs of interest:', len(ids))
        print(f'restrictive: Examples of IDs of interest : {ids_of_interest[:10]}')
        print(f'restrictive: Examples of our IDs: {ids[:10]}')
        ids = [i for i in ids if i[:len('BDMAP_00032584')] in ids_of_interest]
        print('restrictive: Number of IDs after filtering IDs of interest:', len(ids))
        
    with open(args.class_list, 'r') as f:
        class_list = yaml.load(f, Loader=yaml.SafeLoader)
        #sort--we sorted when saving in nii2npy.py
    args.class_list = sorted(class_list)
    class_list=args.class_list
    print('Class list:', class_list)

    args.classes = len(class_list)

    if args.old_classes is not None:
        with open(args.old_classes, 'r') as f:
            old_classes = yaml.load(f, Loader=yaml.SafeLoader)
            #sort--we sorted when saving in nii2npy.py
        args.old_classes = sorted(old_classes)
        old_classes=args.old_classes
    else:
        old_classes = None

    #filter ids here
    if not args.overwrite:
        print('Ids before filtering:', len(ids))
        # ─── filter out already‐predicted cases ─────────────────────────────────────────────
        ids = filter_already_predicted(ids,
                                   save_path=args.save_path,
                                   class_list=class_list,
                                   overwrite=args.overwrite)
        print('Ids after filtering:', len(ids))
        #raise ValueError('This is just a test, please remove this line to run the code.')
        # ────────────────────────────────────────────────────────────────────────────────────

    if args.parts>1:
        window = len(ids)//args.parts
        start = args.current_part*window
        end = args.current_part*window + window
        if (end+window)>len(ids):
            #last part
            ids = ids[start:]
        else:
            ids = ids[start:end]
        print('Ids to predict after splitting:',len(ids))
        
    if args.save_probabilities_report_tumors_only:
        meta = pd.read_csv(args.meta)
        meta = meta.set_index('BDMAP ID')



    model_list = init_model(args,classes=class_list,old_classes=old_classes)

    random.shuffle(ids)

    import tqdm
    for img_name in tqdm.tqdm(ids):
        case_id = img_name.replace('/ct.nii.gz','').replace('.nii.gz','').replace('.npz','')
        pancreas = None
        tic = time.time()
        if not args.overwrite:
            pred_dir = os.path.join(args.save_path, case_id, 'predictions')
            if os.path.isdir(pred_dir):
                present = {_nii_stem(f) for f in os.listdir(pred_dir) if f.endswith('.nii.gz')}
                if set(class_list).issubset(present):
                    print('Already predicted:', img_name)
                    continue
    	
        print(f"Start processing {img_name}")
        if 'nii.gz' in img_name:
            if args.predict_pancreas_only:
                raise ValueError('You cannot predict only pancreas with nii.gz files, please use npz files.')
            
            try:
                itk_img = os.path.join(args.img_path, img_name)
                if '.nii.gz' not in img_name:
                    itk_img = os.path.join(args.img_path, img_name, 'ct.nii.gz')
                itk_img=sitk.ReadImage(itk_img)
                tmp_itk_img = sitk.GetImageFromArray(sitk.GetArrayFromImage(itk_img))
                tmp_itk_img.CopyInformation(itk_img)
            except:
                #for itk errors, load with nib.
                itk_img = os.path.join(args.img_path, img_name)
                if '.nii.gz' not in img_name:
                    itk_img = os.path.join(args.img_path, img_name, 'ct.nii.gz')
                tmp_itk_img=nib_load(itk_img)
                

            

            tensor_img, original_idx, origin_orientation, reoriented_itk_img = preprocess(tmp_itk_img, [1.0,1.0,1.0], args)
        else:
            #for npz files
            tensor_img = np.load(os.path.join(args.img_path, img_name))['arr_0']
            tensor_img = torch.from_numpy(tensor_img).cuda().float()
            
            if args.predict_pancreas_only:
                #load pancreas mask
                labels = np.load(os.path.join(args.img_path, img_name.replace('.npz', '_gt.npz')))['arr_0']
                if labels.shape[0] != args.classes:
                    labels = np.unpackbits(labels, axis=0)
                    assert labels.shape[0] < (args.classes+10)
                    assert labels.shape[0] >= (args.classes)
                    labels = labels[:args.classes]
                pancreas = labels[sorted(class_list).index('pancreas')]
                pancreas = torch.from_numpy(pancreas).cuda().float()
            
        pred_label, pred_raw = prediction(model_list, tensor_img, args, tgt_organ=pancreas)    
        #try:
        #    pred_label, pred_raw = prediction(model_list, tensor_img, args, tgt_organ=pancreas)
        #    print('Predicted case:', img_name)
        #except:
        #    print('FAILED')
        #    #raise ValueError('Failed to predict case:', img_name)
        #    with open('prediction_errors.txt', "a") as f:
        #        f.write(str(os.path.join(args.img_path, img_name, 'ct.nii.gz')) + "\n")
        #    continue
        


        try:
            if 'nii.gz' in img_name:
                pred_dict = postprocess(pred_label, reoriented_itk_img, original_idx, 
                origin_orientation, [1.0,1.0,1.0], class_list, args)
            else:
                pred_dict = postprocess_npz(pred_label, class_list, args)
        except:
            print('FAILED postprocess')
            #raise ValueError('Failed to predict case:', img_name)
            with open('prediction_errors.txt', "a") as f:
                f.write(str(os.path.join(args.img_path, img_name, 'ct.nii.gz')) + "\n")
            continue

        toc = time.time()
        
        if not os.path.exists(os.path.join(args.save_path, case_id, 'predictions')):
            os.makedirs(os.path.join(args.save_path, case_id, 'predictions'))
            
            
        if args.save_pancreas_lesion_only:
            tmp = {}
            for key in pred_dict.keys():
                if 'pancrea' in key:
                    tmp[key] = pred_dict[key]
            pred_dict = tmp
            
        if not args.not_save_binary:
            for key in pred_dict.keys():
                if 'nii.gz' in img_name:
                    sitk.WriteImage(pred_dict[key], os.path.join(args.save_path, case_id, 'predictions', f"{key}.nii.gz"))
                else:
                    #np.savez(os.path.join(args.save_path, case_id, 'predictions', f"{key}.npz"), pred_dict[key])
                    #use nib to save a nii.gz version too
                    nib.save(nib.Nifti1Image(pred_dict[key], np.eye(4)), os.path.join(args.save_path, case_id, 'predictions', f"{key}.nii.gz"))

        if args.save_probabilities:
            if 'nii.gz' in img_name:
                pred_raw_dict = postprocess_non_binary(pred_raw, reoriented_itk_img, original_idx, 
                                            origin_orientation, [1.0,1.0,1.0], class_list, args)
            else:
                pred_raw_dict = postprocess_npz(pred_raw, class_list, args)
            if args.save_pancreas_lesion_only:
                tmp = {}
                for key in pred_raw_dict.keys():
                    if 'pancrea' in key:
                        tmp[key] = pred_raw_dict[key]
                pred_raw_dict = tmp
            if not os.path.exists(os.path.join(args.save_path, case_id, 'predictions_raw')):
                os.makedirs(os.path.join(args.save_path, case_id, 'predictions_raw'))
            for key in pred_raw_dict.keys():
                if 'nii.gz' in img_name:
                    sitk.WriteImage(pred_raw_dict[key], os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.nii.gz"))
                else:
                    #np.savez(os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.npz"), pred_raw_dict[key])
                    #use nib to save a nii.gz version too
                    nib.save(nib.Nifti1Image(pred_raw_dict[key], np.eye(4)), os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.nii.gz"))
        
        if args.save_probabilities_lesions or args.save_probabilities_report_tumors_only:
            if 'nii.gz' in img_name:
                pred_raw_dict = postprocess_non_binary_lesion(pred_raw, reoriented_itk_img, original_idx, 
                                            origin_orientation, [1.0,1.0,1.0], class_list, args)
            else:
                pred_raw_dict = postprocess_npz(pred_raw, class_list, args)
            if args.save_pancreas_lesion_only:
                tmp = {}
                for key in pred_raw_dict.keys():
                    if 'pancrea' in key:
                        tmp[key] = pred_raw_dict[key]
                pred_raw_dict = tmp

            if not os.path.exists(os.path.join(args.save_path, case_id, 'predictions_raw')):
                os.makedirs(os.path.join(args.save_path, case_id, 'predictions_raw'))
            for key in pred_raw_dict.keys():
                if ('lesion' not in key) and ('pdac' not in key) and ('pnet' not in key) and ('cyst' not in key):
                    continue
                if args.save_probabilities_report_tumors_only:
                    column = f'number of {key.replace("_", " ").replace("adrenal","adrenal gland")} instances'
                    lesions= meta.loc[img_name[:len('BDMAP_00052990')], column]
                    # If more than one row matched, keep the first value (clean your data!!)
                    if isinstance(lesions, pd.Series):
                        lesions = lesions.iloc[0]
                    if lesions == 0:
                        continue
                
                if 'nii.gz' in img_name:
                    sitk.WriteImage(pred_raw_dict[key], os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.nii.gz"))
                else:
                    #np.savez(os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.npz"), pred_raw_dict[key])
                    #use nib to save a nii.gz version too
                    nib.save(nib.Nifti1Image(pred_raw_dict[key], np.eye(4)), os.path.join(args.save_path, case_id, 'predictions_raw', f"{key}.nii.gz"))
        
        print(f"{img_name} finished. Process time: {toc-tic}s")


