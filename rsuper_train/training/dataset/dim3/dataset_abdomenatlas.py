import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import math
import random
import pdb
from training import augmentation
import yaml
import time
import sys
import  importlib
from pathlib import Path

#python dataset_abdomenatlas.py --dataset abdomenatlas --model medformer --dimension 3d --batch_size 2 --crop_on_tumor --save_destination /fastwork/psalvador/JHU/data/atlas_300_medformer_augmented_npy_augmented_multich_crop_on_tumor/ --crop_on_tumor --multi_ch_tumor --workers_overwrite 10

def balance_classes(class1, class2):
    """
    Balances two lists of strings by repeating the smaller one until its length
    matches the larger one and then shuffling both lists.
    
    Parameters:
        class1 (list of str): The first class.
        class2 (list of str): The second class.
        
    Returns:
        tuple: A tuple (balanced_class1, balanced_class2) with both lists balanced.
    """
    # Determine which list is smaller
    if len(class1) < len(class2):
        # Compute how many times to repeat class1 to match class2's size
        times = len(class2) // len(class1)
        remainder = len(class2) % len(class1)
        balanced_class1 = class1 * times + class1[:remainder]
        balanced_class2 = class2[:]  # Make a copy of class2
    elif len(class2) < len(class1):
        times = len(class1) // len(class2)
        remainder = len(class1) % len(class2)
        balanced_class2 = class2 * times + class2[:remainder]
        balanced_class1 = class1[:]  # Make a copy of class1
    else:
        # If they are already equal in size, just copy them
        balanced_class1, balanced_class2 = class1[:], class2[:]
    
    # Shuffle both lists in place
    random.shuffle(balanced_class1)
    random.shuffle(balanced_class2)
    
    return balanced_class1, balanced_class2

def get_class_proportions(meta,sample_list,lesion_class_names):
    """
    Get class weights based on the sample list and the meta information.
    This function will return a weight for each class in the sample list.
    :param meta: pandas dataframe with the meta information
    :param sample_list: list of sample names to consider
    :param lesion_classes: list of lesion classes to consider
    :return: list of weights for each class in the sample list
    """
    import pandas as pd
    # Get the meta information for the samples in the sample list
    if isinstance(meta, str):
        meta = pd.read_csv(meta)
    
    # For each sample in sample_list, we add one row to the meta dataframe, accept duplicates!
    tmp = []
    for sample in sample_list:
        # If the sample is not in the meta, add it with zeros for all classes
        tmp.append(meta[meta['BDMAP ID'] == sample]) # get the row for the sample
    meta = pd.concat(tmp, ignore_index=True) # concatenate all the rows for the sample list
    if len(meta) == 0:
        raise ValueError("No samples found in the meta file for the provided sample list.")
    
    print('Lesion class names:', lesion_class_names, flush=True)
    #organs
    organs_lesion_classes = {n.replace('_lesion','').replace('_','').replace('adrenal','adrenal gland'): n for n in lesion_class_names} # remove lesion suffix to get organs

    # Get the counts for each class
    cols = [f'number of {organ} lesion instances' for organ in list(organs_lesion_classes.keys())] # get the columns for each organ lesion instance
    
    #print cols missing from meta
    for col in cols:
        if col not in meta.columns:
            lesion_cols_meta = sorted([col for col in meta.columns if 'number of' in col.lower() and 'lesion instances' in col.lower()])
            raise ValueError(f"Column '{col}' not found in the meta dataframe. Lesion columns found in meta: {lesion_cols_meta}. Please check the meta file or the lesion class names provided.")
    
    # Get the counts for each class in the sample list
    meta = meta[cols]
    
    #make int
    meta = meta.fillna(0).astype(int) # fill NaN with 0 and convert to int
    
    #binarize
    meta = (meta >= 1).astype(int) # convert to binary, 1 if there is at least one instance, 0 otherwise
    
    #sum
    counts = meta.sum(axis=0)  # Sum across all samples for each class
    
    #create a dict of class proportions
    total = len(meta)  # Total number of samples in the sample list
    assert total == len(sample_list), f"Total number of samples in the meta ({total}) does not match the sample list ({len(sample_list)})"
    proportions = {}
    for i, organ in enumerate(list(organs_lesion_classes.keys())):
        proportions[organs_lesion_classes[organ]] = counts[f'number of {organ} lesion instances'] / total if total > 0 else 0  # Avoid division by zero
    #now calculate how many samples have no lesion
    meta['no_lesion'] = (meta[cols].sum(axis=1) == 0).astype(int)
    no_lesion_count = meta['no_lesion'].sum() # how many samples have no lesions
    # Add the no lesion count to the proportions
    proportions['healthy'] = no_lesion_count / total if total > 0 else 0  # Proportion of healthy samples
    
    #print the proportions for debugging
    print('Class proportions:', proportions, flush=True)
    print('Number of samples:', total, flush=True)
    
    return proportions

def get_sample_weight(labels,proportions,class_names,balancer=None,loading_augmented=False):
    weights = []
    tumors = []
    eps = 1e-4
    if balancer is not None:
        tumor_prop = 1-proportions['healthy'] 
        if loading_augmented:
            #read yaml with tumor proportions
            #with open(os.path.join(self.save_destination, 'tumor_proportions.yaml'), 'w') as f:
            with open(os.path.join(balancer.save_destination, 'tumor_proportions.yaml'), 'r') as f:
                proportions = yaml.load(f, Loader=yaml.SafeLoader)
                if proportions is None:
                    raise ValueError('Tumor proportions could not be loaded from yaml file!')
        else:
            #get proportions from balancer
            proportions = balancer.tumor_proportions
        for k,v in proportions.items():
            proportions[k] = v * tumor_prop 
        
    for i, c in enumerate(class_names):
        if c in proportions.keys():
            if labels[i].sum() > 0: #positive sample for class
                weights.append(1.0 / (eps + proportions[c]))
                tumors.append(c) # keep track of which tumors are present in the labels
            else: #negative sample for class
                weights.append(1.0 / (eps + (1-proportions[c])))
        else:
            # If the class is not in proportions, assign a default weight
            weights.append(1.0)
    # Normalize weights to sum to 1*number of classes
    weights = torch.tensor(weights)
    weights = weights / weights.sum() # Normalize to sum to 1
    weights = weights*len(class_names) # Scale by number of classes to keep the relative weights
    
    #print
    #print('Sample tumors:',tumors,' ; sample weights:', weights, flush=True)
    return weights


class AbdomenAtlasDataset(Dataset):
    def __init__(self, args, mode='train', k_fold=10, k=0, seed=0, all_train=False,
                crop_on_tumor=False,
                 save_destination=None,  
                 load_augmented=False,
                 gigantic_length=True,
                 save_augmented=False,
                 id_list=None):    
        
        self.mode = mode
        self.args = args
        self.load_augmented = load_augmented   
        self.save_counter = 0 
        self.save_destination = save_destination
        self.load_augmented = load_augmented
        self.gigantic_length=gigantic_length
        self.save_augmented = save_augmented
        assert mode in ['train', 'test']

        if save_destination is not None:
            try:
                with open(os.path.join(args.save_destination, 'list', 'dataset.yaml'), 'r') as f:
                    img_name_list = yaml.load(f, Loader=yaml.SafeLoader)

                with open(os.path.join(args.save_destination, 'list', 'label_names.yaml'), 'r') as f:
                    classes = yaml.load(f, Loader=yaml.SafeLoader)
                    #sort--we sorted when saving in nii2npy.py
                    classes = sorted(classes)
                    print('Classes are:',classes)
                    print('Got the classes from:',os.path.join(args.save_destination, 'list', 'label_names.yaml'))
            except:
                with open(os.path.join(args.data_root, 'list', 'dataset.yaml'), 'r') as f:
                    img_name_list = yaml.load(f, Loader=yaml.SafeLoader)
                with open(os.path.join(args.data_root, 'list', 'label_names.yaml'), 'r') as f:
                    classes = yaml.load(f, Loader=yaml.SafeLoader)
                    #sort--we sorted when saving in nii2npy.py
                    classes = sorted(classes)
        else:
            with open(os.path.join(args.data_root, 'list', 'dataset.yaml'), 'r') as f:
                    img_name_list = yaml.load(f, Loader=yaml.SafeLoader)
            with open(os.path.join(args.data_root, 'list', 'label_names.yaml'), 'r') as f:
                classes = yaml.load(f, Loader=yaml.SafeLoader)
                #sort--we sorted when saving in nii2npy.py
                classes = sorted(classes)


        random.Random(seed).shuffle(img_name_list)

        if id_list is not None and mode == 'train':
            import pandas as pd
            subset=pd.read_csv(id_list,header=None)[0].tolist()
            img_name_list = [i for i in img_name_list if i in subset]
            print('Using id_list:',id_list)
            print('Length of img_name_list:',len(img_name_list))
            #raise ValueError('Not implemented yet')


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

        print('case list:',img_name_list)
        print('Start loading %s data'%self.mode)
        
        
        if args.balance_pos_neg and mode == 'train':
            atlas_meta = pd.read_csv(args.atlas_meta)
            cols = [col for col in atlas_meta.columns if 'number of' in col.lower() or 'instances' in col.lower()]
            # Filter the rows where all selected columns are 0
            atlas_healthy = atlas_meta[(atlas_meta[cols] == 0).all(axis=1)]
            atlas_diasease = atlas_meta[(atlas_meta[cols] > 0).any(axis=1)]
            #get only the cases in img_name_list
            atlas_healthy = [i for i in atlas_healthy['BDMAP ID'].tolist() if i in img_name_list]
            atlas_diasease = [i for i in atlas_diasease['BDMAP ID'].tolist() if i in img_name_list]
            assert len(atlas_healthy) > 0, 'No healthy cases found in atlas metadata!'
            assert len(atlas_diasease) > 0, 'No disease cases found in atlas metadata!'
            print('Atlas healthy cases:', len(atlas_healthy))
            atlas_healthy, atlas_diasease = balance_classes(atlas_healthy, atlas_diasease)
            print('After balancing Atlas, healthy cases:', len(atlas_healthy))
            print('After balancing Atlas, disease cases:', len(atlas_diasease))
            
            # Combine JHH and Atlas
            img_name_list = atlas_healthy + atlas_diasease
            print('After balancing JHH and Atlas, total image name list length:', len(img_name_list))

        path = args.data_root

        self.img_list = []
        self.lab_list = []
        self.spacing_list = []

        for name in img_name_list:
                
            img_name = name + '.npy'
            lab_name = name + '_gt.npy'
            #check is the file exists, if not, try .npz
            if not os.path.exists(os.path.join(path, img_name)):
                img_name = name + '.npz'
                lab_name = name + '_gt.npz'
            if not os.path.exists(os.path.join(path, img_name)):
                raise ValueError('File does not exist, neither as npy nor as npz:', os.path.join(path, img_name))

            img_path = os.path.join(path, img_name)
            lab_path = os.path.join(path, lab_name)

            spacing = np.array((1.0, 1.0, 1.0)).tolist()
            self.spacing_list.append(spacing[::-1])  # itk axis order is inverse of numpy axis order

            self.img_list.append(img_path)
            self.lab_list.append(lab_path)

        self.crop_on_tumor = crop_on_tumor
        
        
        self.classes = classes
        self.num_classes = len(classes)
        print('Classes:')
        for i, c in enumerate(classes):
            print(i, c)

        if self.crop_on_tumor:
            lesion_classes = []
            lession_class_names = []
            for i, c in enumerate(classes):
                if 'lesion' in c.lower():
                    lesion_classes.append(i)
                    lession_class_names.append(c)
            self.lesion_classes = lesion_classes
            print('Lesion classes:', lesion_classes)

        self.saved_count = 0  # Reset the saved count on instantiation
        print('Load done, length of dataset:', len(self.img_list))
        
        if args.class_weights:
            meta = args.atlas_meta
            meta_ufo = args.ufo_meta
            import pandas as pd
            meta = pd.read_csv(meta)
            meta_ufo = pd.read_csv(meta_ufo)
            meta = pd.concat([meta, meta_ufo], ignore_index=True) # combine JHH and UFO meta
            self.class_proportions = get_class_proportions(
                meta=meta, 
                sample_list=img_name_list,
                lesion_class_names=lession_class_names
            )
        else:
            self.class_proportions = None
            
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
                    print('FAILED TO LOAD AUGMENTED DATA:', self.img_list[idx], self.lab_list[idx], flush=True, file=sys.stderr)
            #    print('FAILED TO LOAD AUGMENTED DATA:', self.img_list[idx], self.lab_list[idx])
            #    pass
        try:
            np_img = np.load(self.img_list[idx], mmap_mode='r', allow_pickle=False)
            if 'npz' in self.img_list[idx]:
                np_img = np_img['arr_0']
        except:
            print('Error loading:', self.img_list[idx])
            try:
                np_img = np.load(self.img_list[idx])
                if 'npz' in self.img_list[idx]:
                    np_img = np_img['arr_0']
            except:
                raise ValueError('Error loading:', self.img_list[idx])
        try:
            np_lab = np.load(self.lab_list[idx], mmap_mode='r', allow_pickle=False)
            if 'npz' in self.lab_list[idx]:
                np_lab = np_lab['arr_0']
        except:
            print('Error loading:', self.lab_list[idx])
            try:
                np_lab = np.load(self.lab_list[idx])
                if 'npz' in self.lab_list[idx]:
                    np_lab = np_lab['arr_0']
            except:
                raise ValueError('Error loading:', self.lab_list[idx])

        #print('Label shape:', np_lab.shape, flush=True, file=sys.stderr)

        if np_lab.shape[0] != len(self.classes):
            #print('Shape before unpack:', np_lab.shape, flush=True, file=sys.stderr)
            start_unpack = time.time()
            # 4. Unpack the bits along the same axis.
            np_lab = np.unpackbits(np_lab, axis=0)
            assert np_lab.shape[0] < (len(self.classes)+10)
            assert np_lab.shape[0] >= (len(self.classes))
            np_lab = np_lab[:self.num_classes]
            #print('Label unpacked:', np_lab.shape, flush=True, file=sys.stderr)
            #print('Time to unpack:', time.time() - start_unpack, flush=True, file=sys.stderr)


        if self.mode == 'train':
            d, h, w = self.args.training_size
            #np_img, np_lab = augmentation.np_crop_3d(np_img, np_lab, [d+20, h+40, w+40], mode='random')

            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)
            tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)
            #print('Time to load data:', time.time() - start, flush=True, file=sys.stderr)
            aug_start = time.time()

            del np_img, np_lab

            #pad with zeros if the image is smaller than the training patch size + a little margin
            tensor_img, tensor_lab = augmentation.pad_volume_pair(tensor_img, tensor_lab, d+20, h+40, w+40)

            error=False
            try:
                if np.random.random() < 0.4:
                    #crop large, then rotate and crop small
                    #print('Shape of tensor_lab:', tensor_lab.shape, flush=True, file=sys.stderr)
                    assert len(tensor_lab.shape) == 5
                    tumor_case = tensor_lab[:,self.lesion_classes].sum()>0
                    tensor_img, tensor_lab = augmentation.random_crop_on_tumor(tensor_img, tensor_lab, self.lesion_classes, d+20, h+40, w+40,tumor_case)
                    if self.args.aug_device == 'gpu':
                        tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                        tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
                    tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
                    tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
                    #print('Shape of tensor after rotate tumor crop:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)

                else:
                    #just crop on tumor
                    assert len(tensor_lab.shape) == 5
                    tumor_case = tensor_lab[:,self.lesion_classes].sum()>0
                    tensor_img, tensor_lab = augmentation.random_crop_on_tumor(tensor_img, tensor_lab, self.lesion_classes, d, h, w,tumor_case)
                    if self.args.aug_device == 'gpu':
                        tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                        tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
                    #print('Shape of tensor after tumor crop:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)
                #print('Crop on tumor successful for:', self.img_list[idx], flush=True, file=sys.stderr)

            except:
                error=True
                print('Error cropping on tumor for:', self.img_list[idx], flush=True, file=sys.stderr)

            
            if not self.crop_on_tumor or error:
                tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, [d+20, h+40, w+40], mode='random')
                if self.args.aug_device == 'gpu':
                    tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                    tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
                if np.random.random() < 0.4:
                    tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
                    tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
                else:
                    tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
                #print('Shape of tensor after random crop:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)

            tensor_img, tensor_lab = tensor_img.contiguous(), tensor_lab.contiguous()

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
            #assert tensor_lab.dtype == torch.int8
            assert tensor_img.dtype == torch.float32
            del np_img, np_lab
            

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        assert tensor_img.shape[1:] == tensor_lab.shape[1:]
        #print('Shapes:',tensor_img.shape, tensor_lab.shape)

        # Save for sanity check
        #self.save_sanity_check(tensor_img, tensor_lab, idx)

        # If a save_destination is given, store the augmented sample there as .npy
        if self.save_augmented:
            #print('Shape:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)
            self.save(tensor_img, tensor_lab, idx)
        #print('Time to augment data:', time.time() - aug_start, flush=True, file=sys.stderr)

        if self.mode == 'train':
            #print('Shapes:', tensor_img.shape, tensor_lab.shape)
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            return tensor_img, tensor_lab, get_sample_weight(tensor_lab,self.class_proportions,self.classes,
                                                             balancer=None) if self.class_proportions else torch.ones_like(tensor_lab) # return the sample weight for balancing classes
        else:
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            return tensor_img, tensor_lab, np.array(self.spacing_list[idx])

    
    def save(self, tensor_img, tensor_lab, idx):
        """
        Saves the augmented image/label pair to disk if a destination was specified.
        Uses numpy .npy/.npz format and keeps the original naming scheme.
        """
        os.makedirs(self.save_destination, exist_ok=True)

        # Keep the same filenames as the original
        base_img_name = os.path.basename(self.img_list[idx])   # e.g. "xxx.npy"
        base_lab_name = os.path.basename(self.lab_list[idx])   # e.g. "xxx_gt.npy"

        img_filename = os.path.join(self.save_destination, base_img_name)
        lab_filename = os.path.join(self.save_destination, base_lab_name)

        np_img = tensor_img.cpu().numpy()
        np_lab = tensor_lab.cpu().numpy().astype(np.bool_)  

        np_lab = np.packbits(np_lab, axis=0) #from bool to uint8 - reduce the channels dimension by 8. Each voxel is saved a a byte anyway. This reduce the size of the file by 8.
        #print('Shape of label after packing:', np_lab.shape)

        # Save as .npy---we can save the crops as npy, they are small, and we can load them faster. But we save the full images as npz.
        np.save(img_filename.replace('.npz', '.npy'), np_img)
        np.save(lab_filename.replace('.npz', '.npy'), np_lab)

        self.save_counter += 1

    def load_augmented_data(self, idx):
        #print('Loading augmented data:', self.img_list[idx], self.lab_list[idx], flush=True, file=sys.stderr)
        # We'll assume the user has already run the dataset once to save the augmented data.
        if self.save_destination is None:
            raise ValueError("load_augmented=True but save_destination=None. Cannot load augmented data.")
        
        

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
        np_img = np.load(aug_img_path, mmap_mode='r', allow_pickle=False)  # shape as saved
        tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0).float()
        #print('Time to load augmented image:', time.time() - start, flush=True, file=sys.stderr)
        start = time.time()
        #print shapes
        #print('Shape:', np_img.shape, np_lab.shape)

        # Convert to torch
        # The code expects image to be float32 and label int8 (for checking).
        np_lab = np.load(aug_lab_path, mmap_mode='r', allow_pickle=False)  # uint8

        # 4. Unpack the bits along the same axis.
        if np_lab.shape[0] != len(self.classes):
            #print('Shape before unpack:', np_lab.shape, flush=True, file=sys.stderr)
            start_unpack = time.time()
            # 4. Unpack the bits along the same axis.
            np_lab = np.unpackbits(np_lab, axis=0)
            #print('Shape after unpack:', np_lab.shape, flush=True, file=sys.stderr)
            #print('Len classes:', len(self.classes), flush=True, file=sys.stderr)
            #self.num_classes
            #print('Num classes:', self.num_classes, flush=True, file=sys.stderr)
            assert np_lab.shape[0] < (len(self.classes)+10)
            assert np_lab.shape[0] >= (len(self.classes))
            np_lab = np_lab[:self.num_classes]
            
            #print('Label unpacked:', np_lab.shape, flush=True, file=sys.stderr)
            #print('Time to unpack:', time.time() - start_unpack, flush=True, file=sys.stderr)

        tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)
        #print('Shape of tensor_lab:', tensor_lab.shape, flush=True, file=sys.stderr)

        #print('Time to load augmented label:', time.time() - start, flush=True, file=sys.stderr)
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
            #print('Applied augmentation online!')
        
        #print('Augmentation deactivated!')

        # You can still call save_sanity_check if desired
        #self.save_sanity_check(tensor_img, tensor_lab, idx)

        #print('Time augmenting data:', time.time() - aug_start, flush=True, file=sys.stderr)

        tensor_img = tensor_img.squeeze(0)

        #print('Shapes:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)

        #print('', flush=True, file=sys.stderr)
        #print('Loaded augmented data:', self.img_list[idx], self.lab_list[idx], 'Shape:', tensor_lab.shape, flush=True, file=sys.stderr)
        #print('', flush=True, file=sys.stderr)
        
        if self.mode == 'train':
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            return tensor_img, tensor_lab, get_sample_weight(tensor_lab,self.class_proportions,self.classes,
                                                             balancer=None,loading_augmented=True) if self.class_proportions else torch.ones_like(tensor_lab) 
        else:
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
            #print shape
            #print('Shape:', img.squeeze().cpu().numpy().shape)
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

def npy_to_nii(npy_path, nii_path, spacing=(1.0, 1.0, 1.0)):
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

    # Convert NumPy array to SimpleITK image
    sitk_image = sitk.GetImageFromArray(array)

    # Optionally set image spacing (if known)
    sitk_image.SetSpacing(spacing)

    # Write to .nii.gz
    sitk.WriteImage(sitk_image, nii_path)
