import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
import pdb
from training import augmentation
import os
import yaml
import time
import sys
import pandas as pd
import json

import  importlib
from pathlib import Path

#python dataset_abdomenatlas.py --dataset abdomenatlas --model medformer --dimension 3d --batch_size 2 --crop_on_tumor --save_destination /fastwork/psalvador/JHU/data/atlas_300_medformer_augmented_npy_augmented_multich_crop_on_tumor/ --crop_on_tumor --multi_ch_tumor --workers_overwrite 10


def clean_ufo(reports,annotated_tumors):
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

    interest['healthy'] = reports[reports['no lesion'] == True]
    print('Number of healthy cases:', interest['healthy']['BDMAP_ID'].nunique())
    #concat
    tumors_per_type = {}
    for k,v in interest.items():
        tumors_per_type[k]=v['BDMAP_ID'].unique().tolist()
    interest = pd.concat(interest.values())
    interest = interest.drop_duplicates()
    print('Total number of useful cases:', interest['BDMAP_ID'].nunique())
    ids_of_interest = interest['BDMAP_ID'].unique().tolist()
    return interest, ids_of_interest,tumors_per_type

class AbdomenAtlasDataset(Dataset):
    def __init__(self, args, mode='train', seed=0, all_train=False,
                crop_on_tumor=True,
                 save_destination=None,  
                 load_augmented=False,
                 gigantic_length=True,
                 save_augmented=False,
                 tumor_classes=['liver','kidney','pancreas'],
                 balance_supervision=True,
                 UFO_only=False,
                 Atlas_only=False):    
        super(AbdomenAtlasDataset, self).__init__()
        print('Tumor classes:', tumor_classes, flush=True, file=sys.stderr)
        
        self.mode = mode
        self.args = args
        self.load_augmented = load_augmented   
        self.save_counter = 0 
        self.save_destination = save_destination
        self.gigantic_length=gigantic_length
        self.save_augmented = save_augmented
        self.tumor_class_names = tumor_classes
        self.reports = pd.read_csv(args.reports)
        print('Reports loaded from:', args.reports, flush=True, file=sys.stderr)
        print('Number of reports:', len(self.reports), flush=True, file=sys.stderr)
        self.tumor_classes=tumor_classes
        self.zero_masks={}
        assert mode in ['train', 'test']
        self.counter=0

        with open(os.path.join(args.data_root, 'list', 'dataset.yaml'), 'r') as f:
            atlas_name_list = yaml.load(f, Loader=yaml.SafeLoader)
           #print('Number of Atlas Images:', len(atlas_name_list), flush=True, file=sys.stderr)

        with open(os.path.join(args.UFO_root, 'list', 'dataset.yaml'), 'r') as f:
            img_name_list_UFO = yaml.load(f, Loader=yaml.SafeLoader)
           #print('UFO root:', args.UFO_root, flush=True, file=sys.stderr)
           #print('Number of UFO Images:', len(img_name_list_UFO), flush=True, file=sys.stderr)

        #from reports, get only those in the dataset
        ids = [case.replace('_0000.nii.gz','').replace('.nii.gz','') for case in img_name_list_UFO]

        if args.ucsf_ids is not None:
            cases = pd.read_csv(args.ucsf_ids)
            cases = cases['BDMAP_ID'].tolist()
            ids = [case for case in ids if case in cases]
            #filter out img_name_list_UFO
            img_name_list_UFO = [case for case in img_name_list_UFO if case in cases]
            
        print()
        print('NUMBER OF SELECTED UFO IDs:', len(ids), flush=True, file=sys.stderr)
        print('NUMBER OF SELECTED UFO IMAGES:', len(img_name_list_UFO), flush=True, file=sys.stderr)
        print()

        self.reports = self.reports[self.reports['BDMAP_ID'].isin(ids)]
        self.reports, ids, tumors_per_type = clean_ufo(self.reports,tumor_classes)
        print('Number of reports after filtering:', len(self.reports), flush=True, file=sys.stderr)
        #use ids to filter img_name_list and img_name_list_UFO
        img_name_list_UFO = [case for case in img_name_list_UFO \
            if case.replace('_0000.nii.gz','').replace('.nii.gz','') in ids]
        

        if mode == 'train' and balance_supervision is True:
            if len(atlas_name_list)>len(img_name_list_UFO):
                diff = len(atlas_name_list) - len(img_name_list_UFO)
                #randomly select some from ufo
                sampled_items = random.choices(img_name_list_UFO, k=diff)
                img_name_list_UFO = img_name_list_UFO + sampled_items
            elif len(img_name_list_UFO)>len(atlas_name_list):
                #randomly select some from atlas
                diff = len(img_name_list_UFO) - len(atlas_name_list)
                sampled_items = random.choices(atlas_name_list, k=diff)
                atlas_name_list = atlas_name_list + sampled_items


        #concatenate the two lists 
        if UFO_only and Atlas_only:
            raise ValueError('You cannot use both UFO_only and Atlas_only at the same time. Please choose one or the other.')
        if UFO_only:
            img_name_list = img_name_list_UFO
            atlas_name_list = []
            print('Using only UFO images:', flush=True, file=sys.stderr)
        elif Atlas_only:
            img_name_list = atlas_name_list
            img_name_list_UFO = []
            print('Using only Atlas images:', flush=True, file=sys.stderr)
        else:
            img_name_list = atlas_name_list + img_name_list_UFO
        random.Random(seed).shuffle(img_name_list)

        self.tumor_annotated_seg = {}



        if not all_train:
            length = len(img_name_list)
            test_name_list = img_name_list[:min(200, length//10)]
            train_name_list = list(set(img_name_list) - set(test_name_list))
        else:
            train_name_list = img_name_list
            test_name_list = None
        
        if mode == 'train':
            img_name_list = train_name_list
        else:
            img_name_list = test_name_list

        #print(img_name_list)
        #print('Start loading %s data'%self.mode)

        self.img_list = []
        self.lab_list = []
        self.spacing_list = []

        self.UFO_paths=[]
        self.Atlas_paths=[]

        for name in img_name_list:
                
            img_name = name + '.npy'
            lab_name = name + '_gt.npy'

            if name in atlas_name_list:
                img_path = os.path.join(args.data_root, img_name)
                lab_path = os.path.join(args.data_root, lab_name)

                #npy or npz?
                if not os.path.exists(img_path):
                    img_path = img_path.replace('.npy','.npz')
                    lab_path = lab_path.replace('.npy','.npz')
                if not os.path.exists(img_path):
                    raise ValueError('Image %s not found in npy nor npz'%img_path)

                self.Atlas_paths.append(img_path)
                self.tumor_annotated_seg[img_path] = True
            elif name in img_name_list_UFO:
                img_path = os.path.join(args.UFO_root, img_name)
                lab_path = os.path.join(args.UFO_root, lab_name)

                #npy or npz?
                if not os.path.exists(img_path):
                    img_path = img_path.replace('.npy','.npz')
                    lab_path = lab_path.replace('.npy','.npz')
                if not os.path.exists(img_path):
                    raise ValueError('Image %s not found in npy nor npz'%img_path)

                self.UFO_paths.append(img_path)
                self.tumor_annotated_seg[img_path] = False
            else:
                raise ValueError('Image %s not in any of the two lists'%name)

            spacing = np.array((1.0, 1.0, 1.0)).tolist()
            self.spacing_list.append(spacing[::-1])  # itk axis order is inverse of numpy axis order

            self.img_list.append(img_path)
            self.lab_list.append(lab_path)
            
        self.crop_on_tumor = crop_on_tumor
        
        with open(os.path.join(args.data_root, 'list', 'label_names.yaml'), 'r') as f:
            classes = yaml.load(f, Loader=yaml.SafeLoader)
            #sort--we sorted when saving in nii2npy.py
            classes = sorted(classes)
            print('Classes list loaded from %s'%f, flush=True, file=sys.stderr)
            print('Classes:', classes, flush=True, file=sys.stderr)
            print('Number of Classes:', len(classes), flush=True, file=sys.stderr)

        with open(os.path.join(args.UFO_root, 'list', 'label_names.yaml'), 'r') as f:
            classes_UFO = yaml.load(f, Loader=yaml.SafeLoader)
            #sort--we sorted when saving in nii2npy.py
            classes_UFO = sorted(classes_UFO)

        self.classes = classes
        self.classes_UFO = classes_UFO
        self.num_classes = len(classes)
        self.num_classes_UFO = len(classes_UFO)

        #print('Classes:')
        #for i, c in enumerate(classes):
            #print(i, c)
        #print('Classes UFO:')
        #for i, c in enumerate(classes_UFO):
            #print(i, c)
            #raise ValueError('Classes UFO are:', classes_UFO)

        if self.crop_on_tumor:
            lesion_classes = []
            for i, c in enumerate(classes):
                if 'lesion' in c.lower():
                    if c.lower().replace('_lesion','').replace('pancreatic','pancreas') in tumor_classes:
                        lesion_classes.append(i)
            self.lesion_classes = lesion_classes
            print('Lesion classes:', lesion_classes)

        self.saved_count = 0  # Reset the saved count on instantiation
        #print('Load done, length of dataset:', len(self.img_list))

        #check if all ids are in the reports
        # Convert IDs from the DataFrame to a set for faster lookup
        report_ids = set(self.reports['BDMAP_ID'].values)
        # Find IDs not present in the DataFrame
        missing_ids = [id for id in ids if id not in report_ids]
        # Raise an error if there are missing IDs
        if missing_ids:
            raise ValueError(f"IDs not in reports: {missing_ids}. Length of reports: {len(self.reports)}, number of missing ids: {len(missing_ids)}")



        if args.model_genesis_pretrain:
            # 1) Find MedFormer/ (it’s four levels up from this file):
            medformer_root = Path(__file__).resolve().parents[4]
            if not medformer_root.joinpath("baselines").is_dir():
                raise ImportError(f"Cannot find baselines/ under {medformer_root}")
            # 2) Ensure Python will search there
            if str(medformer_root) not in sys.path:
                sys.path.insert(0, str(medformer_root))
            # 3) Import the utils module and bind your method
            mg = importlib.import_module("baselines.model_genesis.utils")
            self.generate_pair = mg.generate_one_pair
        else:
            self.generate_pair = None
            
            
        if UFO_only:
            #UFO list may include atlas itens IF atlas is the seed dataset
            tmp=[]
            atlas_ids = [x for x in os.listdir(args.data_root) if 'BDMAP' in x]
            for filename in self.img_list:
                skip = False
                for atlas in atlas_ids:
                    id_=atlas[atlas.rfind('BDMAP'):atlas.rfind('BDMAP')+len('BDMAP_12345678')]
                    #assert id_ not in filename, f"Found forbidden ID '{id_}' in filename '{filename}'"
                    if id_ in filename:
                        skip = True
                        break    # stop checking more IDs
                if not skip:
                    tmp.append(filename)
            print('Removed cases:', len(self.img_list)-len(tmp),flush=True)
            print('Remaining cases:', len(tmp),flush=True)
            self.img_list = tmp
            
        if Atlas_only:
            tmp=[]
            ufo_ids = [x for x in os.listdir(args.UFO_root) if 'BDMAP' in x]
            for filename in self.img_list:  
                skip = False
                for ufo in ufo_ids:
                    id_=ufo[ufo.rfind('BDMAP'):ufo.rfind('BDMAP')+len('BDMAP_12345678')]
                    #assert id_ not in filename, f"Found forbidden ID '{id_}' in filename '{filename}'"
                    if id_ in filename:
                        skip = True
                        break
                if not skip:
                    tmp.append(filename)
            print('Removed cases:', len(self.img_list)-len(tmp),flush=True)
            print('Remaining cases:', len(tmp),flush=True)
            self.img_list = tmp
            
        print(f'Number of images in {self.mode} set:', len(self.img_list), flush=True, file=sys.stderr)
        
    def read_report(self, idx):
        id = self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
        if id not in self.reports['BDMAP_ID'].values:
            #print('ID is: ',id)
            raise ValueError('ID is not in the reports:', id, 'Length of reports:', len(self.reports))
            return None #no tumor
        else:
            tumors=self.reports[self.reports['BDMAP_ID']==id]
            #tumors=tumors.to_dict(orient='records')
            return tumors

    def __len__(self):
        if self.mode == 'train':
            if self.gigantic_length:
                return len(self.img_list) * 100000
            else:
                return len(self.img_list)
        else:
            return len(self.img_list)

    def __getitem__(self, idx):
        start = time.time()
        
        idx = idx % len(self.img_list)
        
        #print('Loading:', self.img_list[idx], self.lab_list[idx])
        self.current_sample = self.img_list[idx]
        if self.load_augmented:
            #return self.load_augmented_data(idx)
            try:
                return self.load_augmented_data(idx)#loads and returns data already augmented and pre-saved
            except:
                #change index to another one at random
                idx = np.random.randint(len(self.img_list))
                try:
                    return self.load_augmented_data(idx)
                except:
                    print('FAILED TO LOAD AUGMENTED DATA:', self.img_list[idx], self.lab_list[idx])
            #    #print('FAILED TO LOAD AUGMENTED DATA:', self.img_list[idx], self.lab_list[idx])
            #    pass

        try:
            np_img = np.load(self.img_list[idx], mmap_mode='r', allow_pickle=False)
            if '.npz' in self.img_list[idx]:
                np_img = np_img['arr_0']
        except:
            print('Error loading:', self.img_list[idx])
            try:
                np_img = np.load(self.img_list[idx])
                if '.npz' in self.img_list[idx]:
                    np_img = np_img['arr_0']
            except:
                raise ValueError('Error loading:', self.img_list[idx])
        try:
            np_lab = np.load(self.lab_list[idx], mmap_mode='r', allow_pickle=False)
            if '.npz' in self.lab_list[idx]:
                np_lab = np_lab['arr_0']
        except:
            print('Error loading:', self.lab_list[idx])
            try:
                np_lab = np.load(self.lab_list[idx])
                if '.npz' in self.lab_list[idx]:
                    np_lab = np_lab['arr_0']
            except:
                raise ValueError('Error loading:', self.lab_list[idx])

        if self.img_list[idx] in self.UFO_paths:
            classes = self.classes_UFO
        else:
            classes = self.classes

        if np_lab.shape[0] != len(classes):
            ##print('Shape before unpack:', np_lab.shape, flush=True, file=sys.stderr)
            start_unpack = time.time()
            # 4. Unpack the bits along the same axis.
            np_lab = np.unpackbits(np_lab, axis=0)
            assert np_lab.shape[0] < len(classes) +10
            np_lab = np_lab[:len(classes)]
            ##print('Label unpacked:', np_lab.shape, flush=True, file=sys.stderr)
            ##print('Time to unpack:', time.time() - start_unpack, flush=True, file=sys.stderr)


        if self.mode == 'train':
            d, h, w = self.args.training_size
            #np_img, np_lab = augmentation.np_crop_3d(np_img, np_lab, [d+20, h+40, w+40], mode='random')

            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)
            tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)
            ##print('Time to load data:', time.time() - start, flush=True, file=sys.stderr)
            aug_start = time.time()

            del np_img, np_lab
            tensor_img, tensor_lab = tensor_img.contiguous(), tensor_lab.contiguous()
            #pad with zeros if the image is smaller than the training patch size + a little margin
            tensor_img, tensor_lab = augmentation.pad_volume_pair(tensor_img, tensor_lab, d+20, h+40, w+40)
            
            tensor_img, tensor_lab, tumor_dict, selected_tumor = self.crop(tensor_img, tensor_lab, idx, d, h, w)
            
            

            if not self.save_augmented:
                #this augmentation is online.
                if np.random.random() < 0.3:
                    tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
                if np.random.random() < 0.3:
                    tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.5])
                if np.random.random() < 0.3:
                    std = np.random.random() * 0.2 
                    tensor_img = augmentation.gaussian_noise(tensor_img, std=std)
        
        else:
            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)#.float()
            tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)#.to(torch.uint8)
            #assert type is int8
            assert tensor_lab.dtype == torch.int8
            assert tensor_img.dtype == torch.float32
            del np_img, np_lab

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        assert tensor_img.shape[1:] == tensor_lab.shape[1:]
        
        #if the item is from the UFO dataset, we convert the labels to the atlas format--negative classes are set to 0, unknown classes are SET TO NAN.
        if self.img_list[idx] in self.UFO_paths:
            #convert to atlas format
            tensor_lab, unk_channels, unk_channels_tensor = self.assign_labels(tensor_lab,idx)
            tumor_volumes_in_crop,tumor_diameters=self.estimate_tumor_volume(idx,tumor_segment_crop=selected_tumor)
            chosen_segment_mask=self.get_chosen_segment_mask(tensor_lab, selected_tumor)
        else:
            unk_channels_tensor = torch.zeros(tensor_lab.shape).type_as(tensor_lab)
            unk_channels = {}
            tumor_volumes_in_crop=[0,0,0,0,0,0,0,0,0,0]
            tumor_diameters=torch.zeros((10,3)).float()
            chosen_segment_mask = torch.zeros(tensor_lab.shape).type_as(tensor_lab)#it is important to define this as 0--or it will cause loss problems!

        dta={'tumor_in_crop':selected_tumor,
             'unknown_per_voxel':unk_channels}
        

        if self.save_augmented:
            self.save(tensor_img, tensor_lab, idx, tumor_dict, dta, unk_channels_tensor=unk_channels_tensor, tumor_volumes_in_crop=tumor_volumes_in_crop,chosen_segment_mask=chosen_segment_mask,tumor_diameters=tumor_diameters)
        ##print('Time to augment data:', time.time() - aug_start, flush=True, file=sys.stderr)


        if self.mode == 'train':
            ##print('Shapes:', tensor_img.shape, tensor_lab.shape)
            self.SanityAssertOutput(tensor_lab, unk_channels_tensor,torch.tensor(tumor_volumes_in_crop).float(),chosen_segment_mask.float())
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            
            retur = {"image":           tensor_img.clone(),
                    "label":           tensor_lab.clone(),
                    "unk_channels":    unk_channels_tensor.clone(),
                    "volumes":         torch.tensor(tumor_volumes_in_crop).float().clone(),
                    "mask":            chosen_segment_mask.float().clone(),
                    "diameters":       tumor_diameters.type_as(tensor_img).clone()
                    }
            return retur
            
            #return tensor_img, tensor_lab, unk_channels_tensor,torch.tensor(tumor_volumes_in_crop).float(),chosen_segment_mask.float(),tumor_diameters.type_as(tensor_img)
        else:
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            return tensor_img, tensor_lab, np.array(self.spacing_list[idx])
        
    def random_crop(self, tensor_img, tensor_lab, d, h, w):
        
        tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, [d+20, h+40, w+40], mode='random')
        if self.args.aug_device == 'gpu':
            tensor_img = tensor_img.cuda(self.args.proc_idx).float()
            tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
        if np.random.random() < 0.4:
            tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
        else:
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
        return tensor_img, tensor_lab
    
    def random_crop_on_tumor(self, tensor_img, tensor_lab, d, h, w, fallback_crop=False):
        
        forg=[]
        for c in self.tumor_class_names:
            if 'pancrea' in c:
                forg.append('pancreas')
            elif 'kidney' in c:
                forg.append('kidney_right')
                forg.append('kidney_left')
            elif 'gall' in c:
                forg.append('gall_bladder')
            else:
                forg.append(c)
        forg = list(set(forg))
        #now we need the indexes
        if tensor_lab.shape[1] == len(self.classes_UFO):
            cls = self.classes_UFO
            lesion_classes = []
        elif tensor_lab.shape[1] == len(self.classes):
            cls = self.classes
            lesion_classes = self.lesion_classes
        else:
            raise ValueError('Label tensor must have %s channels, but got %s channels'%(len(self.classes_UFO), tensor_lab.shape[1]))
        forg = [cls.index(c) for c in forg]#we have only atlas here
        
        if fallback_crop:
            tumor_case = False
            print('Fallback crop', flush=True, file=sys.stderr)
        else:
            tumor_case = tensor_lab[:,self.lesion_classes].sum()>0
        
        if np.random.random() < 0.4:
            #crop large, then rotate and crop small
            assert len(tensor_lab.shape) == 5
            tensor_img, tensor_lab = augmentation.random_crop_on_tumor(tensor_img, tensor_lab, lesion_classes, d+20, h+40, w+40,tumor_case,
                                 foreground_classes=forg)
            if self.args.aug_device == 'gpu':
                tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
            tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
            ##print('Shape of tensor after rotate tumor crop:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)

        else:
            #just crop on tumor
            assert len(tensor_lab.shape) == 5
            tensor_img, tensor_lab = augmentation.random_crop_on_tumor(tensor_img, tensor_lab, lesion_classes, d, h, w,tumor_case,
                                                                       foreground_classes=forg)
            if self.args.aug_device == 'gpu':
                tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
            ##print('Shape of tensor after tumor crop:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)
        ##print('Crop on tumor successful for:', self.img_list[idx], flush=True, file=sys.stderr)
        return tensor_img, tensor_lab
    
    def clean_subseg_list(self, tumor_segments):
        #split tumor segments that have /
        tmp=[]
        for segment in tumor_segments:
            if pd.isna(segment) or segment == 'u':
                continue
            else:
                sublist=segment.split(' / ')
                if sublist not in tmp:
                    tmp.append(sublist)
        tumor_segments = tmp
        tumor_segments_flat = list(set([item for sublist in tmp for item in sublist]))
        return tumor_segments, tumor_segments_flat
    
    def get_tumor_segment_labels(self, idx):
        """
        This function reads the LLM output for a given report, and its most importat outputs are subseg_with_only_known_sizes and organs_with_only_known_sizes_n_segments.
        These outputs represent organ/organ subsegments that contain tumors but do not contain tumors with unknown size.  
        """
        tumors=self.read_report(idx)
        if tumors is None:
            #no tumor, just do random crop
            retur = {'tumor_segments':[],
                    'tumor_segments_flat':[],
                    'tumor_organs':[],
                    'organs_with_unk_tumor_segment':[],
                    'organs_with_unk_tumor_size':[],
                    'organs_with_only_known_sizes_n_segments':[],
                    'subseg_with_only_known_sizes':[],
                    'subseg_with_unk_tumor_size':[],
                    'subsegs_in_organs_with_unk':[]}
            #print('No tumor found for:', self.img_list[idx], flush=True, file=sys.stderr)
            return retur,tumors
        else:
            #tumor is present
            tumor_segments = tumors['Standardized Location'].tolist()
            if self.args.no_pancreas_subseg:
                #print(f'Segments before removing pancreas subseg: {tumor_segments}', flush=True, file=sys.stderr)
                #print(f'tumor organs: {tumors["Standardized Organ"].tolist()}', flush=True, file=sys.stderr)
                tumor_segments = []
                for s in tumors['Standardized Location'].tolist():
                    if isinstance(s,str) and (('pancrea' in s) or ('head' in s) or ('tail' in s) or ('body' in s)):
                        tumor_segments.append('pancreas')
                    else:
                        tumor_segments.append(s)
                print('Pancreas subseg removed', flush=True, file=sys.stderr)
                if not self.args.pancreas_only:
                    raise ValueError('no_pancreas_subseg only implemented for pancreas_only. If we remove kidney segment we will have a problem, as right and left are seen as segments')
            tumor_sizes = tumors['Tumor Size (mm)'].tolist()
            tumor_organs = tumors['Standardized Organ'].tolist()
            
            #print('UFO: No lesion Case?', tumors['no lesion'].values[0], flush=True, file=sys.stderr)


            #check which organs have tumors with unknown size or segment
            organs_with_unk_tumor_segment = []
            organs_with_unk_tumor_size = []
            #and which subsegments have unknown size
            subseg_with_unk_tumor_size = []
            for i in list(range(len(tumor_organs))):
                if pd.isna(tumor_sizes[i]) or tumor_sizes[i] == 'u' or tumor_sizes[i]=='multiple':
                    organs_with_unk_tumor_size.append(tumor_organs[i])
                    subseg_with_unk_tumor_size.append(tumor_segments[i])
                if pd.isna(tumor_segments[i]) or tumor_segments[i] == 'u':
                    organs_with_unk_tumor_segment.append(tumor_organs[i])

            #check which segments are in an organ with some unknown tumor size or segment
            subsegs_in_organs_with_unk = []
            for i in list(range(len(tumor_organs))):
                #check if the organ is not in the list of organs with unknown tumor segment
                if tumor_organs[i] in organs_with_unk_tumor_segment or tumor_organs[i] in organs_with_unk_tumor_size:
                    subsegs_in_organs_with_unk.append(tumor_segments[i])

            

            tumor_segments, tumor_segments_flat = self.clean_subseg_list(tumor_segments)
            subseg_with_unk_tumor_size, subseg_with_unk_tumor_size_flat = self.clean_subseg_list(subseg_with_unk_tumor_size)
            subsegs_in_organs_with_unk, subsegs_in_organs_with_unk_flat = self.clean_subseg_list(subsegs_in_organs_with_unk)

            tumor_organs = list(set(organ for organ in tumor_organs if not pd.isna(organ) and organ != 'u'))
            organs_with_unk_tumor_segment = list(set(organ for organ in organs_with_unk_tumor_segment if not pd.isna(organ) and organ != 'u'))
            organs_with_unk_tumor_size = list(set(organ for organ in organs_with_unk_tumor_size if not pd.isna(organ) and organ != 'u'))

            #subsegments with only known sizes
            subseg_with_only_known_sizes = list(set(tumor_segments_flat) - set(subseg_with_unk_tumor_size_flat) - set(subsegs_in_organs_with_unk_flat))
            #organs with only known sizes and locations of tumors
            organs_with_only_known_sizes_n_segments = list(set(tumor_organs) - set(organs_with_unk_tumor_segment) - set(organs_with_unk_tumor_size))

            #for subseg_with_only_known_sizes, you must check tumor_segments, and consider segments that come in pairs
            #check if some sub-segment is in more than one item in the list, if so, merge the items
            tmp=[]
            for segment in subseg_with_only_known_sizes:
                #get all items that contain the segment in the list tumor_segments
                items = [item for item in tumor_segments if segment in item]
                #flatten
                items = list(set([item for sublist in items for item in sublist]))
                #items represent a list of sub-segments that share tumors with segment
                #check if any of them is in the list of prohibted segments
                if any(item in subseg_with_unk_tumor_size_flat for item in items) or \
                   any(item in subsegs_in_organs_with_unk_flat for item in items):
                    continue
                else:
                    tmp.append(items)
            subseg_with_only_known_sizes=tmp
            

            #create a big dict with the variables here
            retur = {'tumor_segments':tumor_segments,
                    'tumor_segments_flat':tumor_segments_flat,
                    'tumor_organs':tumor_organs,
                    'organs_with_unk_tumor_segment':organs_with_unk_tumor_segment,
                    'organs_with_unk_tumor_size':organs_with_unk_tumor_size,
                    'organs_with_only_known_sizes_n_segments':organs_with_only_known_sizes_n_segments,
                    'subseg_with_only_known_sizes':subseg_with_only_known_sizes,
                    'subseg_with_unk_tumor_size':subseg_with_unk_tumor_size,
                    'subsegs_in_organs_with_unk':subsegs_in_organs_with_unk}
            #raise ValueError('You must change the handling of this function output everywhere it is used')
            #print('Tumor Dict:', tumors[['Standardized Location','Tumor Size (mm)','Standardized Organ']])
            #print('subseg_with_only_known_sizes:', retur['subseg_with_only_known_sizes'], flush=True, file=sys.stderr)
            #print('organs_with_only_known_sizes_n_segments:', retur['organs_with_only_known_sizes_n_segments'], flush=True, file=sys.stderr)
            #print('XXXXXXXX Tumor Found for:', self.img_list[idx], flush=True, file=sys.stderr)
            return retur,tumors

    def get_random_tumor_seg_mask(self, tensor_lab, tumor_segment, exclude=None):
        #print('Selected tumor segment:', tumor_segment, flush=True, file=sys.stderr)
        #get the mask for a given segment/organ or segment list
        
        if not isinstance(tumor_segment, list):
            tumor_segment = [tumor_segment]

        if len(tumor_segment)==1 and tumor_segment[0] == 'pancreas':
            #pancreas is a special case, we have pancreas labels but they are not in the atlas format
            #we assign all pancreas labels to 1
            tumor_segment = ['head','body','tail']
        if len(tumor_segment)==1 and tumor_segment[0] == 'liver':
            #liver is a special case, we have liver labels but they are not in the atlas format
            #we assign all liver labels to 1
            tumor_segment = ['segment 1','segment 2','segment 3','segment 4','segment 5','segment 6','segment 7','segment 8']
        
        #get the labels of the tumor segment
        segment_labels=[seg.replace('segment ','liver_segment_').replace('head','pancreas_head').replace('body','pancreas_body').replace('tail','pancreas_tail').replace('left','kidney_left').replace('right','kidney_right') for seg in tumor_segment]

        #print('Segment labels are:', segment_labels, flush=True, file=sys.stderr)
        for label in segment_labels:
            if label not in self.classes_UFO:
                raise ValueError('Label %s not in classes_UFO'%label)

        if len(tensor_lab.shape) == 4:
            tensor_lab = tensor_lab.unsqueeze(0)
        assert len(tensor_lab.shape) == 5, f'Label tensor must have 5 dimensions, but got {len(tensor_lab.shape)} dimensions and shape {tensor_lab.shape}'
        if tensor_lab.shape[1] == len(self.classes_UFO):
            classes = self.classes_UFO
        elif tensor_lab.shape[1] == len(self.classes):
            classes = self.classes
        else:
            raise ValueError('Label tensor must have %s channels, but got %s channels'%(len(self.classes_UFO), tensor_lab.shape[1]))
        
        tumor_segment_labels = []
        for i,clss in enumerate(classes,0):
            if clss in segment_labels:
                tumor_segment_labels.append(i)

        #print('Label indices of tumor segment are:', tumor_segment_labels, flush=True, file=sys.stderr)
        
        #get the tumor segment mask
        tumor_segment_mask=[]
        #print('The shape of tensor_lab is:', tensor_lab.shape, flush=True, file=sys.stderr)
        for i in range(tensor_lab.shape[1]):
            if i in tumor_segment_labels:
                tumor_segment_mask.append(tensor_lab[:,i])
        tumor_segment_mask=torch.stack(tumor_segment_mask,axis=0)
        tumor_segment_mask=tumor_segment_mask.sum(0)
        #binarize
        tumor_segment_mask[tumor_segment_mask>0]=1
        #assert tumor_segment_mask.sum().item()!=0.0, f'problem in case {self.current_sample}, tumor segment mask is empty, crop is in {tumor_segment}'
        return tumor_segment_mask

    def get_chosen_segment_mask(self, tensor_lab, tumor_segment):
        if tumor_segment == 'random':
            return torch.zeros_like(tensor_lab).type_as(tensor_lab)
        
        print('Chosen segment:', tumor_segment, flush=True, file=sys.stderr)
        segment_mask = self.get_random_tumor_seg_mask(tensor_lab, tumor_segment).squeeze(0)
        assert segment_mask.sum().item()!=0.0, f'problem in case {self.current_sample}, segment_mask is empty, crop is in {tumor_segment}'
        #apply it to the lesion classes
        segment_mask_lesion_ch = []
        #print('Segment is:', tumor_segment, flush=True, file=sys.stderr)
        for c in self.classes:
            if (any('segment' in item for item in tumor_segment) or any('liver' in item for item in tumor_segment)) and 'liver_lesion' in c:
                segment_mask_lesion_ch.append(segment_mask)
                #print('Segment added to class:', c, flush=True, file=sys.stderr)
            elif (any('head' in item for item in tumor_segment) or any('body' in item for item in tumor_segment) or any('tail' in item for item in tumor_segment) or any('pancreas' in item for item in tumor_segment))\
                  and 'pancreatic_lesion' in c:
                segment_mask_lesion_ch.append(segment_mask)
                #print('Segment added to class:', c, flush=True, file=sys.stderr)
            elif (any('left' in item for item in tumor_segment) or any('right' in item for item in tumor_segment) or any('kidney' in item for item in tumor_segment)) and 'kidney_lesion' in c:
                segment_mask_lesion_ch.append(segment_mask)
                #print('Segment added to class:', c, flush=True, file=sys.stderr)
            else:
                segment_mask_lesion_ch.append(torch.zeros_like(tensor_lab[0]).type_as(tensor_lab))
        segment_mask_lesion_ch = torch.stack(segment_mask_lesion_ch,axis=0)
        assert segment_mask_lesion_ch.sum().item()!=0.0, f'problem in case {self.current_sample}, chosen segment mask is empty, crop is in {tumor_segment}'
        return segment_mask_lesion_ch
            
    
    def crop(self, tensor_img, tensor_lab, idx, d, h, w):
        if self.tumor_annotated_seg[self.img_list[idx]]:
            #print('This is an image with per-voxel annotations:'+self.img_list[idx], flush=True, file=sys.stderr)
            #for data with per-voxel tumor annotations
            #10% random crop probability is already inside the augmentation.random_crop_on_tumor
            error=False
            try:
                tensor_img, tensor_lab = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w)
            except:
                error=True
                #print('Error cropping on tumor for:', self.img_list[idx], flush=True, file=sys.stderr)

            if not self.crop_on_tumor or error:
                tensor_img, tensor_lab = self.random_crop(tensor_img, tensor_lab, d, h, w)
                ##print('Shape of tensor after random crop:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)
            return tensor_img, tensor_lab, None, None
        else:
            #print('This is an image with tumor annotations from reports:'+self.img_list[idx], flush=True, file=sys.stderr)
            #data without per-voxel tumor annotations, just reports mentioning tumors
            segments,tumor_dict=self.get_tumor_segment_labels(idx)

            if len(segments['subseg_with_only_known_sizes'])>0:
                segment_options=segments['subseg_with_only_known_sizes']
            elif len(segments['organs_with_only_known_sizes_n_segments'])>0:
                segment_options=segments['organs_with_only_known_sizes_n_segments']
            elif len(segments['subseg_with_only_known_sizes'])>0:
                segment_options=segments['subseg_with_only_known_sizes']
            else:
                #no tumor, do random cropping
                tensor_img, tensor_lab = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w,fallback_crop=True)
                return tensor_img, tensor_lab, tumor_dict, 'random'
            
            #crop around the tumor organ/subsegment 
            # 90% chance of cropping on tumor:
            if np.random.random() < 0.1:
                #random crop, low chance
                tensor_img, tensor_lab = self.random_crop(tensor_img, tensor_lab, d, h, w)
                print('Random crop by chance', flush=True, file=sys.stderr)
                return tensor_img, tensor_lab, tumor_dict, 'random'
            else:
                #randomly pick a segment
                tumor_segment = random.choice(segment_options)
                #print('Chosen segment a:', tumor_segment, flush=True, file=sys.stderr)
                #get the mask for the tumor segment
                tumor_segment_mask=self.get_random_tumor_seg_mask(tensor_lab, tumor_segment)
                
                if tumor_segment_mask.sum().item()==0.0:
                    self.zero_masks[self.current_sample]=tumor_segment
                    #save as yaml
                    with open('zero_masks.yaml', 'w') as f:
                        yaml.dump(self.zero_masks, f)
                    #remove tumor_segment from segment_options
                    segment_options = [seg for seg in segment_options if seg not in [tumor_segment]]
                    if len(segment_options)==0:
                        tensor_img, tensor_lab = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, fallback_crop=True)
                        print('Random crop because of empty mask', flush=True, file=sys.stderr)
                        return tensor_img, tensor_lab, tumor_dict, 'random'
                    else:
                        tumor_segment = random.choice(segment_options)
                        tumor_segment_mask=self.get_random_tumor_seg_mask(tensor_lab, tumor_segment)
                        if tumor_segment_mask.sum().item()==0.0:
                            #random crop
                            tensor_img, tensor_lab = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, fallback_crop=True)
                            print('Random crop because of empty mask', flush=True, file=sys.stderr)
                            return tensor_img, tensor_lab, tumor_dict, 'random'

                out = augmentation.crop_foreground_3d(tensor_ct=tensor_img, tensor_lab=tensor_lab, foreground=tumor_segment_mask,
                                                      crop_size=[d, h, w])
                if isinstance(out, tuple):
                    tensor_img, tensor_lab, cropped_forg = out
                    print('>>>>>>>>>>Crop on tumor successful for:', self.img_list[idx], flush=True, file=sys.stderr)
                    return tensor_img, tensor_lab, tumor_dict, tumor_segment
                else:
                    print('Error cropping around tumor for:'+self.img_list[idx]+'---'+out, flush=True, file=sys.stderr)
                    if len(segment_options)==1:
                        #random crop 
                        print('Error cropping around tumor for:'+self.img_list[idx]+'---'+out, flush=True, file=sys.stderr)
                        tensor_img, tensor_lab = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, fallback_crop=True)
                        return tensor_img, tensor_lab, tumor_dict, 'random'
                    #select another tumor segment
                    #get a random segment that is not in tumor_segment
                    segment_options = [seg for seg in segment_options if seg not in [tumor_segment]]
                    if len(segment_options)==0:
                        print('Error cropping around tumor for:'+self.img_list[idx]+'---'+out, flush=True, file=sys.stderr)
                        tensor_img, tensor_lab = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, fallback_crop=True)
                        return tensor_img, tensor_lab, tumor_dict, 'random'
                    tumor_segment = random.choice(segment_options)
                    #get the mask for the tumor segment
                    tumor_segment_mask=self.get_random_tumor_seg_mask(tensor_lab, tumor_segment)
                    out = augmentation.crop_foreground_3d(tensor_ct=tensor_img, tensor_lab=tensor_lab, foreground=tumor_segment_mask, crop_size=[d, h, w])
                    if isinstance(out, tuple):
                        tensor_img, tensor_lab, cropped_forg = out
                        print('>>>>>>>>>>Crop on tumor successful for:', self.img_list[idx], flush=True, file=sys.stderr)
                        return tensor_img, tensor_lab, tumor_dict, tumor_segment
                    else:
                        print('Error cropping around tumor for:'+self.img_list[idx]+'---'+out, flush=True, file=sys.stderr)
                        #random crop
                        tensor_img, tensor_lab = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, fallback_crop=True)
                        return tensor_img, tensor_lab, tumor_dict, 'random'

    def save(self, tensor_img, tensor_lab, idx, tumor_dict=None, dta=None, unk_channels_tensor=None,
            tumor_volumes_in_crop=None, chosen_segment_mask=None,tumor_diameters=None):
        """
        Saves the augmented image/label pair to disk if a destination was specified.
        Uses numpy .npy format and keeps the original naming scheme.
        """
        os.makedirs(self.save_destination, exist_ok=True)

        # Keep the same filenames as the original
        base_img_name = os.path.basename(self.img_list[idx])   # e.g. "xxx.npy"
        base_lab_name = os.path.basename(self.lab_list[idx])   # e.g. "xxx_gt.npy"

        img_filename = os.path.join(self.save_destination, base_img_name)
        lab_filename = os.path.join(self.save_destination, base_lab_name)

        np_img = tensor_img.cpu().numpy()
        np_lab = tensor_lab.cpu().numpy().astype(np.bool_)  

        #print('Number of labels:',np_lab.shape[0], flush=True, file=sys.stderr)
        np_lab = np.packbits(np_lab, axis=0) #from bool to uint8 - reduce the channels dimension by 8. Each voxel is saved a a byte anyway. This reduce the size of the file by 8.
        ##print('Shape of label after packing:', np_lab.shape)
        
        img_filename = img_filename.replace('.npz','.npy')
        lab_filename = lab_filename.replace('.npz','.npy')

        # Save as .npy
        np.save(img_filename, np_img)
        np.save(lab_filename, np_lab)
        print('Saved:',img_filename, flush=True, file=sys.stderr)
        print('Saved:',lab_filename, flush=True, file=sys.stderr)


        if unk_channels_tensor is not None:
            unk_ch = unk_channels_tensor.cpu().numpy().astype(np.bool_)
            unk_channels_tensor = np.packbits(unk_ch, axis=0)
            np.save(lab_filename.replace('.npy','_unk.npy'), unk_channels_tensor)

        if chosen_segment_mask is not None:
            chosen_segment_mask = chosen_segment_mask.cpu().numpy().astype(np.bool_)
            chosen_segment_mask = np.packbits(chosen_segment_mask, axis=0)
            np.save(lab_filename.replace('.npy','_chosen_tumor_segment.npy'), chosen_segment_mask)

        if tumor_dict is not None:
            tumor_dict.to_csv(os.path.join(self.save_destination, img_filename.replace('.npy','.csv')), index=False)
        if dta is not None:
            with open(os.path.join(self.save_destination, img_filename.replace('.npy','.json')), "w") as f:
                json.dump(dta, f)
        if tumor_volumes_in_crop is not None:
            with open(os.path.join(self.save_destination, img_filename.replace('.npy','_tumor_volumes.json')), "w") as f:
                json.dump(tumor_volumes_in_crop, f)
        
        if tumor_diameters is not None:
            tumor_diameters = tumor_diameters.cpu().numpy().tolist()
            with open(os.path.join(self.save_destination, img_filename.replace('.npy','_tumor_diameters.json')), "w") as f:
                json.dump(tumor_diameters, f)

        self.save_counter += 1

    def load_augmented_data(self, idx):
        # We'll assume the user has already run the dataset once to save the augmented data.
        if self.save_destination is None:
            raise ValueError("load_augmented=True but save_destination=None. Cannot load augmented data.")
        
        #print('Loading augmented data for:', self.img_list[idx], flush=True, file=sys.stderr)

        start = time.time()

        # Derive the filenames from the original naming scheme
        base_img_name = os.path.basename(self.img_list[idx])    # e.g. "xxx.npy"
        base_lab_name = os.path.basename(self.lab_list[idx])    # e.g. "xxx_gt.npy"
        
        # Replace npz by npy
        base_img_name = base_img_name.replace('.npz', '.npy')
        base_lab_name = base_lab_name.replace('.npz', '.npy')

        aug_img_path = os.path.join(self.save_destination, base_img_name)
        aug_lab_path = os.path.join(self.save_destination, base_lab_name)
        
        # Load the augmented data
        np_img = np.load(aug_img_path, allow_pickle=False)  # shape as saved
        tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0).float()
        ##print('Time to load augmented image:', time.time() - start, flush=True, file=sys.stderr)
        start = time.time()
        ##print shapes
        ##print('Shape:', np_img.shape, np_lab.shape)

        # Convert to torch
        # The code expects image to be float32 and label int8 (for checking).
        np_lab = np.load(aug_lab_path, allow_pickle=False)  # uint8

        # 4. Unpack the bits along the same axis.
        if np_lab.shape[0] != len(self.classes):
            #print('Shape before unpack:', np_lab.shape, flush=True, file=sys.stderr)
            start_unpack = time.time()
            # 4. Unpack the bits along the same axis.
            np_lab = np.unpackbits(np_lab, axis=0)
            assert np_lab.shape[0] < self.num_classes +10
            assert np_lab.shape[0] >= self.num_classes
            np_lab = np_lab[:self.num_classes]
            #print('Label unpacked:', np_lab.shape, flush=True, file=sys.stderr)
            ##print('Time to unpack:', time.time() - start_unpack, flush=True, file=sys.stderr)

        tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)

        ##print('Time to load augmented label:', time.time() - start, flush=True, file=sys.stderr)
        aug_start = time.time()

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        
        if self.mode == 'train':
            #this augmentation is online.
            if np.random.random() < 0.3:
                tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
            if np.random.random() < 0.3:
                tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
            if np.random.random() < 0.3:
                tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
            if np.random.random() < 0.3:
                tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
            if np.random.random() < 0.3:
                tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.5])
            if np.random.random() < 0.3:
                std = np.random.random() * 0.2 
                tensor_img = augmentation.gaussian_noise(tensor_img, std=std)
            ##print('Applied augmentation online!')
        
        ##print('Augmentation deactivated!')

        # You can still call save_sanity_check if desired
        #self.save_sanity_check(tensor_img, tensor_lab, idx)

        ##print('Time augmenting data:', time.time() - aug_start, flush=True, file=sys.stderr)

        tensor_img = tensor_img.squeeze(0)

        ##print('Shapes:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)

        if self.mode == 'train':
            if self.img_list[idx] not in self.UFO_paths:
                #annotated per-voxel, no unknnown voxel
                unk_channels_list=torch.zeros(tensor_lab.shape).type_as(tensor_lab)
                tumor_volumes_in_crop=[0,0,0,0,0,0,0,0,0,0]
                tumor_diameters=torch.zeros((10,3)).float()
                chosen_segment_mask=torch.zeros_like(tensor_lab).type_as(tensor_lab)
            else:
                #try loading unk_channels_list if saved
                unk_pth=aug_lab_path.replace('_gt.npy','_gt_unk.npy')
                if os.path.exists(unk_pth):
                    unk_channels_tensor = np.load(unk_pth, allow_pickle=False)
                    if unk_channels_tensor.shape[0] != len(self.classes):
                        unk_channels_tensor = np.unpackbits(unk_channels_tensor, axis=0)
                        unk_channels_tensor = unk_channels_tensor[:len(self.classes)]
                    unk_channels_list = torch.from_numpy(unk_channels_tensor)
                    #print(f'----------------UNK WAS LOADED FROM {unk_pth}', flush=True, file=sys.stderr)
                else:
                    unk_channels_list=self.define_unknown_voxels(tensor_lab,idx)
                    #print('----------------UNK WAS CREATED', flush=True, file=sys.stderr)
                #load the json file
                with open(os.path.join(self.save_destination, base_img_name.replace('.npy','.json')), "r") as f:
                    dta=json.load(f)
                tumor_volumes_in_crop,tumor_diameters=self.estimate_tumor_volume(idx,tumor_segment_crop=dta['tumor_in_crop'])
                if os.path.exists(aug_lab_path.replace('.npy','_chosen_tumor_segment.npy')):
                    chosen_segment_mask = np.load(aug_lab_path.replace('.npy','_chosen_tumor_segment.npy'), allow_pickle=False)
                    if chosen_segment_mask.shape[0] != len(self.classes):
                        chosen_segment_mask = np.unpackbits(chosen_segment_mask, axis=0)
                        chosen_segment_mask = chosen_segment_mask[:self.num_classes]
                    chosen_segment_mask = torch.from_numpy(chosen_segment_mask)
                else:
                    chosen_segment_mask=self.get_chosen_segment_mask(tensor_lab, dta['tumor_in_crop'])
            #print('LOADED AUGMENTED DATA', tensor_lab.shape, 'From:', os.path.join(self.save_destination, base_lab_name), flush=True, file=sys.stderr)
            self.SanityAssertOutput(tensor_lab, unk_channels_list, torch.tensor(tumor_volumes_in_crop).float(), chosen_segment_mask.float())
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            retur = {"image":           tensor_img,
                    "label":           tensor_lab,
                    "unk_channels":    unk_channels_list,
                    "volumes":         torch.tensor(tumor_volumes_in_crop).float(),
                    "mask":            chosen_segment_mask.float(),
                    "diameters":       tumor_diameters.type_as(tensor_img),}
            return retur
            
            #return tensor_img, tensor_lab, unk_channels_list, torch.tensor(tumor_volumes_in_crop).float(), chosen_segment_mask.float(), tumor_diameters.type_as(tensor_img)
        else:
            #print('LOADED AUGMENTED DATA', tensor_lab.shape, 'From:', os.path.join(self.save_destination, base_lab_name), flush=True, file=sys.stderr)
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            return tensor_img, tensor_lab, np.array(self.spacing_list[idx])

    def save_sanity_check(self, img, lab, idx):
        """Save the image and labels to NIfTI format for sanity checking."""
        if self.saved_count < 10:
            save_dir = './SanityCheck'
            os.makedirs(save_dir, exist_ok=True)

            img_folder = os.path.join(save_dir, f'img{self.saved_count + 1}')
            os.makedirs(img_folder, exist_ok=True)

            # Save the image
            img_nifti = sitk.GetImageFromArray(img.squeeze().cpu().numpy())
            ##print shape
            ##print('Shape:', img.squeeze().cpu().numpy().shape)
            img_nifti.SetSpacing(self.spacing_list[idx])
            sitk.WriteImage(img_nifti, os.path.join(img_folder, 'CT.nii.gz'))

            # Save the labels
            for i, cls in enumerate(self.classes):
                label_array = (lab[i].squeeze().cpu().numpy()).astype(np.int8)
                if label_array.max() > 0:  # Save only if the label exists
                    label_nifti = sitk.GetImageFromArray(label_array)
                    label_nifti.SetSpacing(self.spacing_list[idx])
                    sitk.WriteImage(label_nifti, os.path.join(img_folder, f'{cls}.nii.gz'))

            self.saved_count += 1
    
    def assign_labels(self, tensor_lab, idx):
        """
        UFO data is not annotated per-voxel for some classes, making classes and classes_UFO missmatch. This function adds zero channels for missing classes, and creates a unk_channels dict explining which class is UNKNOWN.
        Some classes are missing but we know they are truly zero, so we do not add them to unk_channels.
        If the missing class is not a lesion (e.g., an organ we do not have pseudo-annotations for) we assign it to unk_channels.
        If it is a lesion, we check tumor_dict. tumor_dict, extracted from the report, explains which organ/segments present tumors. We check if these organs/segments are present in tensor_lab (cropped).
        Tumor labels with a corresponding tumor segment in the crop -> assign to unk_channels (we do not know where the tumor is).
        Tumor labels withour corresponding tumor segment in the crop -> assign label 0 (negative for tumor in the crop).
        """
        clss_to_idx = {clss: i for i, clss in enumerate(self.classes)}
        clss_UFO_to_idx = {clss: i for i, clss in enumerate(self.classes_UFO)}
        all_data,tumor_dict=self.get_tumor_segment_labels(idx)
        tumor_segments=all_data['tumor_segments']
        for tumor_organ in all_data['tumor_organs']:
            if isinstance(tumor_organ,str) and tumor_organ=='liver':
                if not any('segment' in item for item in tumor_segments):
                    if 'liver' not in tumor_segments:
                        tumor_segments.append('liver')
            elif isinstance(tumor_organ,str) and tumor_organ=='pancreas':
                if not any('head' in item for item in tumor_segments) and not any('body' in item for item in tumor_segments) and not any('tail' in item for item in tumor_segments):
                    if 'pancreas' not in tumor_segments:
                        tumor_segments.append('pancreas')
            elif isinstance(tumor_organ,str) and tumor_organ=='kidney':
                if not any('left' in item for item in tumor_segments) and not any('right' in item for item in tumor_segments):
                    if 'kidney' not in tumor_segments:
                        tumor_segments.append('kidney')

        #flatten the list of lists
        tmp=[]
        for item in tumor_segments:
            if isinstance(item, list):
                for subitem in item:
                    tmp.append(subitem)
            else:
                if item == 'pancreas':
                    for it in ['head','body','tail']:
                        tmp.append(it)
                elif item == 'liver':
                    for it in ['segment 1','segment 2','segment 3','segment 4','segment 5','segment 6','segment 7','segment 8']:
                        tmp.append(it)
                elif item =='kidney':
                    for it in ['left','right']:
                        tmp.append(it)
                else:
                    tmp.append(item)
        tumor_segments=tmp

        tumor_segments=list(set(tumor_segments))
        #convert to standard label names:
        tumor_segments=[seg.replace('segment ','liver_segment_').replace('head','pancreas_head').replace('body','pancreas_body').replace('tail','pancreas_tail').replace('left','kidney_left').replace('right','kidney_right') for seg in tumor_segments]
        #tumor_segments represents all organ/subsegments with tumors in the whole ct
        #which lesion classes to add unk? check which of the tumor_segments are in the crop.
        unk_segments={'liver':torch.zeros((tensor_lab.shape[-3],tensor_lab.shape[-2],tensor_lab.shape[-1])).type_as(tensor_lab),
                      'pancreas':torch.zeros((tensor_lab.shape[-3],tensor_lab.shape[-2],tensor_lab.shape[-1])).type_as(tensor_lab),
                      'kidney':torch.zeros((tensor_lab.shape[-3],tensor_lab.shape[-2],tensor_lab.shape[-1])).type_as(tensor_lab)}
        #this variable will create a mask of the segments in the crop that have tumors in unknown locations (report annotation)
        
        unk_lesions=[]
        for seg in tumor_segments:
            seg_idx=clss_UFO_to_idx[seg]
            if tensor_lab[seg_idx].max()>0:
                if 'liver' in seg:
                    unk_segments['liver'][tensor_lab[seg_idx]>0]=1
                elif 'pancreas' in seg:
                    unk_segments['pancreas'][tensor_lab[seg_idx]>0]=1
                elif 'kidney' in seg:
                    unk_segments['kidney'][tensor_lab[seg_idx]>0]=1
                else:
                    raise ValueError('Unrecognized segment:',seg)
                #there is a tumor segment in the crop
                #what is the organ of the tumor segment?
                if '_segment' in seg:
                    organ=seg[:seg.rfind('_segment')]
                else:
                    organ=seg
                organ=organ.replace('_head','').replace('_body','').replace('_tail','').replace('pancreas','pancreatic')
                unk_lesions.append(organ)
        unk_lesions=list(set(unk_lesions))
        #print('unk lesions:', unk_lesions, flush=True, file=sys.stderr)

        unk_channels={}
        unk_channels_list=[]
        label=[]
        #print('Shape of tensor_lab before assigning labels:', tensor_lab.shape, flush=True, file=sys.stderr)
        assert len(tensor_lab.shape) == 4
        for j,clss in enumerate(self.classes,0):
            if clss in self.classes_UFO:
                label.append(tensor_lab[clss_UFO_to_idx[clss]])
                unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
            else:
                if 'lesion' not in clss.lower():
                    if clss=='liver':
                        #join all liver segments
                        l=torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0])
                        for i in [1,2,3,4,5,6,7,8]:
                            l=torch.logical_or(l,tensor_lab[clss_UFO_to_idx['liver_segment_%i'%i]])
                        label.append(l)
                        unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))#this channel is knwon, assign zero to unk_channels_list
                    elif clss=='pancreas':
                        #join all pancreas segments
                        l=torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0])
                        for i in ['head','body','tail']:
                            l=torch.logical_or(l,tensor_lab[clss_UFO_to_idx['pancreas_%s'%i]])
                        label.append(l)
                        unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))#this channel is knwon, assign zero to unk_channels_list
                    else:
                        label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                        unk_channels[clss]=j
                        unk_channels_list.append(torch.ones(tensor_lab[0].shape).type_as(tensor_lab[0]))#no pixel is known for this channel, assign 1 to unk_channels_list

                else:
                    #check if there is a tumorous segment for this lesion in the crop
                    tumor_present=False
                    for organ in unk_lesions:
                        if organ in clss:
                            label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                            unk_channels[clss]=j
                            if 'liver' in clss:
                                unk_channels_list.append(unk_segments['liver'])#make only the pixels with unknown tumor location be 1, background pixels are 0
                            elif 'pancreatic' in clss:
                                unk_channels_list.append(unk_segments['pancreas'])#make only the pixels with unknown tumor location be 1, background pixels are 0
                            elif 'kidney' in clss:
                                unk_channels_list.append(unk_segments['kidney'])#make only the pixels with unknown tumor location be 1, background pixels are 0
                            else:
                                raise ValueError('Organ not recognized:',clss)
                            tumor_present=True
                            break
                    #if not:
                    #assign label 0
                    if not tumor_present:
                        #negative for the tumor
                        label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                        unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
        label=torch.stack(label,dim=0)
        unk_channels_list=torch.stack(unk_channels_list,0)
        assert len(label.shape) == 4
        #print('Shape of tensor_lab after assigning labels:', label.shape, flush=True, file=sys.stderr)
        #print('Unk channels:', unk_channels, flush=True, file=sys.stderr)
        assert unk_channels_list.sum().item()!=0
        return label,unk_channels,unk_channels_list.type_as(label)
    
    def define_unknown_voxels(self, label, idx):
        """
        Defines the unknown voxels in the image. Unlike assign_labels, this function assumes your labels (tensor_lab) are already in the final format, with the correct number of channels, in the order of self.classes.
        unk_channels is a dictionary with the classes that are unknown.
        """

        #we must first re-create the tensor_lab (input of the assign_labels function), from label, the output of the assign_labels function.
        clss_to_idx = {clss: i for i, clss in enumerate(self.classes)}
        clss_UFO_to_idx = {clss: i for i, clss in enumerate(self.classes_UFO)}
        tensor_lab = []
        for j,clss in enumerate(self.classes_UFO,0):
            ##print('j:',j,flush=True, file=sys.stderr)
            ##print('clss:',clss,flush=True, file=sys.stderr)
            if clss=='background':
                #add zeros as placeholder
                tensor_lab.append(torch.zeros(label[0].shape).type_as(label[0]))
                bkg=j
            else:
                tensor_lab.append(label[clss_to_idx[clss]])
        tensor_lab=torch.stack(tensor_lab,dim=0)
        #add to background the opposite of all other classes
        tensor_lab[bkg]=(tensor_lab.sum(dim=0)>0).type_as(tensor_lab[0])
        

        #now we can use the assign_labels function
        #to define the unknown voxels
    
        #convert to atlas format
        label_out,unk_channels,unk_channels_list=self.assign_labels(tensor_lab,idx)
        #sanity check: see if label_out matches label
        assert (torch.equal(label_out,label))
        
        return unk_channels_list


    def estimate_tumor_volume(self, idx, tumor_segment_crop):
        """
        Estimates tumor volume from reports. For the segment in the crop.
        Always returns a list of 10 items, padding with 0.
        """
        _,tumor_dict=self.get_tumor_segment_labels(idx)
        #print('Tumor dict:', tumor_dict)
        #print all column names in tumor_dict
        #print(tumor_dict.columns)
        #print('Sizes:',tumor_dict['Tumor Size (mm)'])
        #print('Cropped on tumor segment:', tumor_segment_crop)
        if tumor_segment_crop is None or tumor_segment_crop=='random':
            return [0,0,0,0,0,0,0,0,0,0], torch.zeros((10,3)).float() #CT not cropped around a tumor segment
        
        if isinstance(tumor_segment_crop, list):
            pass
        elif isinstance(tumor_segment_crop, str):
            tumor_segment_crop=[tumor_segment_crop]
        else:
            raise ValueError('tumor_segment_crop must be a list or a string.')
        
        #is our tumor_segment_crop organ or segment:
        if 'liver' in "".join(tumor_segment_crop) or 'kidney' in "".join(tumor_segment_crop) or 'pancreas' in "".join(tumor_segment_crop):
            tpe='organ'
            col='Standardized Organ'
        elif 'segment' in "".join(tumor_segment_crop) or 'head' in "".join(tumor_segment_crop) or 'body' in "".join(tumor_segment_crop) or 'tail' in "".join(tumor_segment_crop) or 'left' in "".join(tumor_segment_crop) or 'right' in "".join(tumor_segment_crop):
            tpe='segment'
            col='Standardized Location'
        else:
            raise ValueError('tumor_segment_crop does not contain organs or segments:', tumor_segment_crop)
        
        tumors_in_crop=[]
        for row in tumor_dict.iterrows():
            location=row[1][col]
            #print('Location:',location)
            if not isinstance(location, str) or location.lower()=='u':
                continue
            if '/' in location:
                location=location.split(' / ')
            if not isinstance(location, list):
                location=[location]
            in_crop=True
            for loc in location:
                if loc not in tumor_segment_crop:
                    in_crop=False
                    break
            if in_crop:
                tumors_in_crop.append(row[1]['Tumor Size (mm)'])

            #print('Tumors in crop:', tumors_in_crop)#list of strings with sizes

            #print('Tumor dict:',tumor_dict[['Standardized Organ','Standardized Location','Tumor Size (mm)']])
                
        #estimate volumes for each tumor size
        volumes=[]
        diameters=[]
        for size in tumors_in_crop:
            if 'x' not in size:
                diameter=float(size)
                volume=(4/3) * math.pi * ((diameter/2) ** 3)#sphere. volume in mm3 (voxels)
                volumes.append(volume)
                diameters.append([diameter,diameter,diameter])
            else:
                #ellipsoid
                sizes=size.split(' x ')
                sizes=[float(s) for s in sizes]
                if len(sizes)==2:
                    #assume 3rd axis is the average of the other two
                    sizes.append(sum(sizes)/2)
                #ellipsoid volume
                volume=(4/3) * math.pi * ((sizes[0]/2) * (sizes[1]/2) * (sizes[2]/2))
                volumes.append(volume)
                diameters.append(sizes)

        #print('Estimated volumes:',volumes)

        for i in range(len(volumes),10):
            volumes.append(0)
            diameters.append([0,0,0])
            
        return volumes,torch.tensor(diameters).float()
    
    def SanityAssertOutput(self, tensor_lab, unk_channels_tensor,tumor_volumes_in_crop,chosen_segment_mask):
                                #tensor_lab, unk_channels_list, torch.tensor(tumor_volumes_in_crop).float(), chosen_segment_mask.float()
        classes=sorted(self.classes)
        #assert shapes
        assert len(tensor_lab.shape)==4 , 'tensor_lab must have 4 dimensions'
        assert tensor_lab.shape[0]==len(classes), 'Number of classes in tensor_lab (%i) does not match number of classes (%i)'%(tensor_lab.shape,len(classes))
        assert unk_channels_tensor.shape[0]==len(classes), f'Number of classes in unk_channels_tensor ({unk_channels_tensor.shape}) does not match number of classes ({len(classes)})'
        assert chosen_segment_mask.shape[0]==len(classes), f'Number of classes in chosen_segment_mask ({chosen_segment_mask.shape}) does not match number of classes ({len(classes)})'
        assert (tensor_lab.shape==unk_channels_tensor.shape) and (tensor_lab.shape==chosen_segment_mask.shape), f'tensor_lab, unk_channels_tensor and chosen_segment_mask must have the same shape. tensor_lab: %s, unk_channels_tensor: %s, tumor_volumes_in_crop: %s'%(tensor_lab.shape,unk_channels_tensor.shape,tumor_volumes_in_crop.shape)

        
        #save examples
        sample=self.current_sample
        sample=sample[sample.rfind('BDMAP_'):sample.rfind('.')]
        if self.counter<10:
            debug_save_labels(tensor_lab,sample+'_y',self.classes)
            debug_save_labels(chosen_segment_mask,sample+'_chosen_segment_mask',self.classes)
            debug_save_labels(unk_channels_tensor,sample+'_unk_voxels',self.classes)
            self.counter+=1

        #assert that unk_channels_tensor and chosen_segment_mask are 0 for all non lesion classes
        missing_classes=set(classes)-set(self.classes_UFO)-{'liver','pancreas'}
        missing_classes=list(missing_classes)
        #print('Missing classes:', missing_classes,flush=True, file=sys.stderr)
        unk_cls=[]
        known_cls=[]
        for i,clss in enumerate(classes):
            if 'lesion' in clss.lower() or clss in missing_classes: 
                unk_cls.append(i)
            else:
                known_cls.append(i)
        if not unk_channels_tensor[known_cls].sum().item()==0:
            for i,clss in enumerate(classes,0):
                if i in unk_cls:
                    continue
                else:
                    if unk_channels_tensor[i].sum().item()!=0:
                        print('Class with unk voxels:',clss,'Sample is:',sample)
        assert unk_channels_tensor[known_cls].sum().item()==0
        assert chosen_segment_mask[known_cls].sum().item()==0

        #print('Assertions passed!',flush=True, file=sys.stderr)



        

        

def npy_to_nii(npy_path, nii_path, spacing=(1.0, 1.0, 1.0),labels=None):
    """
    Reads a .npy file, converts it to a SimpleITK image, 
    sets spacing, and saves as .nii.gz.

    :param npy_path:    Path to the input .npy file.
    :param nii_path:    Path to the output .nii.gz file.
    :param spacing:     Tuple or list specifying the (z, y, x) spacing. 
                        Default is (1.0, 1.0, 1.0).
    """
    # Load the NumPy array
    array = np.load(npy_path)
    #print('Shape of array:', array.shape)
    #squeeze
    array = array.squeeze()
    #print('Shape after squeeze:',array.shape)

    if labels is not None:
        #load yaml labels
        with open(labels, 'r') as f:
            labels = yaml.load(f, Loader=yaml.SafeLoader)
        #print('Yaml loaded')
        #sort
        labels = sorted(labels)
       #print('Labels:',labels)
       #print('Shape of array:',array.shape)
        if len(array.shape) == 4:
            #label
            if array.shape[0] < len(labels):
                #unpack
                array = np.unpackbits(array, axis=0)
                array = array[:len(labels)]
            os.makedirs(nii_path.replace('.nii.gz',''), exist_ok=True)
            for label in labels:
                #save each label
                sitk_image = sitk.GetImageFromArray(array[labels.index(label)])
                sitk_image.SetSpacing(spacing)
                sitk.WriteImage(sitk_image, os.path.join(nii_path.replace('.nii.gz',''),label+'.nii.gz'))
               #print('Saved:', os.path.join(nii_path.replace('.nii.gz',''),label+'.nii.gz'))
    else:
       #print('No labels provided, saving as a single volume')
        # Convert NumPy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(array)

        # Optionally set image spacing (if known)
        sitk_image.SetSpacing(spacing)

        # Write to .nii.gz
        sitk.WriteImage(sitk_image, nii_path)


def debug_save_labels(labels: torch.Tensor,
                      name='',
                      label_names = '/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml',
                      out_dir: str = "./DatasetSanity",
                      batch_idx: int = 0):
    """
    Saves each channel of the specified batch index in `labels` as a .nii.gz file.
    
    Args:
        labels (torch.Tensor): A tensor of shape (B, C, H, W, D).
        label_names_yaml (str): Path to a YAML file containing a list of label names.
                                The list will be sorted alphabetically and used
                                to name the channels.
        out_dir (str): Output directory to save the .nii.gz files. Defaults to "LossSanity".
        batch_idx (int): Which batch element to save. Defaults to 0.
    """
    import nibabel as nib
    # 1. Create output folder if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Load and sort label names
    if not isinstance(label_names, list):
        with open(label_names, "r") as f:
            label_names = yaml.safe_load(f)  # e.g. ["liver", "kidney", "pancreas", ...]
        
    label_names_sorted = sorted(label_names)  # sort alphabetically
    
    # 3. Basic shape check
    if len(labels.shape)==4:
        labels = labels.unsqueeze(0)
    assert len(labels.shape) == 5
    B, C, H, W, D = labels.shape
    assert batch_idx < B, f"batch_idx={batch_idx} is out of range for B={B}."
    assert C == len(label_names_sorted), (
        f"Number of channels (C={C}) does not match the number of label names "
        f"(={len(label_names_sorted)})."
    )
    
    # 4. Extract just the batch element we want
    #    This will have shape (C, H, W, D).
    label_slice = labels[batch_idx]
    
    # 5. Loop over channels, save each one as a nii.gz
    for c in range(C):
        # Move channel c to CPU numpy for saving
        channel_data = label_slice[c].detach().cpu().numpy()
        
        # Build a simple identity affine; if you have real metadata, replace it
        affine = np.eye(4, dtype=np.float32)
        
        # Convert to float32 (or int16, float64, etc.)
        channel_data = channel_data.astype(np.float32)
        
        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(channel_data, affine)
        
        # Derive a filename from the label name
        channel_label_name = label_names_sorted[c]
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}/{channel_label_name}.nii.gz")
        
        # Save
        nib.save(nifti_img, out_path)
        
    #print(f"Saved to {out_path}")
