import builtins
import logging
import os
import random
import time
import training.losses_foundation as lf
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
#mp.set_sharing_strategy('file_system')
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from training.dataset.dim3.sampler import ChunkedSampler
from training.dataset.dim3.sampler_clip import one_organ_per_batch_sampler

import gc


from training.utils import update_ema_variables
from training.validation import validation_ddp as validation
from training.utils import (
    exp_lr_scheduler_with_warmup, 
    log_evaluation_result, 
    get_optimizer, 
    filter_validation_results,
    unwrap_model_checkpoint,
)
import yaml
import argparse
import time
import math
import sys
import pdb
import warnings
import matplotlib.pyplot as plt
import copy



from utils import (
    configure_logger,
    save_configure,
    is_master,
    AverageMeter,
    ProgressMeter,
    resume_load_optimizer_checkpoint,
    resume_load_model_checkpoint,
)
warnings.filterwarnings("ignore", category=UserWarning)

counter_mg = 0


def train_net(net, trainset, testset, args, ema_net=None, fold_idx=0):
    
    ########################################################################################
    # Dataloader Creation
    #train_sampler = DistributedSampler(trainset) if args.distributed else None
    try:
        leng = len(trainset.img_list)
    except:
        assert trainset.gigantic_length==False, 'You must set gigantic_length to False in the dataset if you want to use the dataloader with a sampler'
        leng = trainset.__len__()
    
    if args.clip_pretrain:
        train_sampler = one_organ_per_batch_sampler(
            dataset_size=leng,#real size of the dataset
            samples_per_epoch=args.iter_per_epoch*args.batch_size*args.ngpus_per_node,
            shuffle=True,
            seed=42,
            rank=dist.get_rank() if args.distributed else 0,
            world_size=dist.get_world_size() if args.distributed else 1,
            dataset = trainset,
            batch_size=args.batch_size_global)
           
        trainLoader = data.DataLoader(
            trainset, 
            batch_sampler=train_sampler,
            pin_memory=(args.aug_device != 'gpu'),
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers>0),
        )
    elif args.model_genesis_pretrain:
        train_sampler = DistributedSampler(trainset) if args.distributed else None
        trainLoader = data.DataLoader(
            trainset, 
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            pin_memory=(args.aug_device != 'gpu'),
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers>0),
        )
    else:
        train_sampler = ChunkedSampler(
            dataset_size=leng,#real size of the dataset
            samples_per_epoch=args.iter_per_epoch*args.batch_size*args.ngpus_per_node,
            shuffle=True,
            seed=42,
            rank=dist.get_rank() if args.distributed else 0,
            world_size=dist.get_world_size() if args.distributed else 1)
        
        trainLoader = data.DataLoader(
            trainset, 
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            pin_memory=(args.aug_device != 'gpu'),
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers>0),
        )
    
    test_sampler = DistributedSampler(testset) if args.distributed else None
    testLoader = data.DataLoader(
        testset,
        batch_size=1,  # has to be 1 sample per gpu, as the input size of 3D input is different
        shuffle=(test_sampler is None), 
        sampler=test_sampler,
        pin_memory=True,
        num_workers=args.num_workers
    )
    
    logging.info(f"Created Dataset and DataLoader")

    ########################################################################################
    # Initialize tensorboard, optimizer, amp scaler and etc.
    writer = SummaryWriter(f"{args.log_path}{args.unique_name}/fold_{fold_idx}") if is_master(args) else None

    optimizer = get_optimizer(args, net)
    
    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args)
        

    #criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weight).cuda().float())
    #criterion = nn.BCEWithLogitsLoss()
    #criterion_dl = DiceLossMultiClass()

    if args.multi_ch_tumor:
        raise ValueError('Not implemented yet')
    else:
        matcher=None
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    ########################################################################################
    # Start training
    best_Dice = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000
    
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)#this shuffles the dataset
        if hasattr(trainLoader.dataset, 'shuffle_atlas'):
            trainLoader.dataset.shuffle_atlas()

        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        #exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=args.warmup, max_epoch=args.epochs)
        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, epoch=epoch, warmup_epoch=args.warmup, max_epoch=args.epochs)
        logging.info(f"Current lr: {exp_scheduler:.4e}")
       
        train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, scaler, args,
                    matcher=matcher)
        
        ##################################################################################
        # Evaluation, save checkpoint and log training info
        
        
        if is_master(args):
            # save the latest checkpoint, including net, ema_net, and optimizer
            net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': net_state_dict,
                'ema_model_state_dict': ema_net_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_latest.pth")

            if (epoch+1) % 25 == 0:
                # save the model
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': net_state_dict,
                    'ema_model_state_dict': ema_net_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_epoch_{epoch+1}.pth")

        #if False:
        if (epoch+1) % args.val_freq == 0 and (not args.clip_pretrain):
            net_for_eval = ema_net if args.ema else net

            dice_list_test, ASD_list_test, HD_list_test = validation(net_for_eval, testLoader, args, matcher=matcher)
            if is_master(args):
                dice_list_test, ASD_list_test, HD_list_test = filter_validation_results(dice_list_test, ASD_list_test, HD_list_test, args) # filter results for some dataset, e.g. amos_mr
                log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, 'test', epoch, args)
            
                if dice_list_test.mean() >= best_Dice.mean():
                    best_Dice = dice_list_test
                    best_HD = HD_list_test
                    best_ASD = ASD_list_test

                    # Save the checkpoint with best performance
                    net_state_dict, ema_net_state_dict = unwrap_model_checkpoint(net, ema_net, args)

                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': net_state_dict,
                        'ema_model_state_dict': ema_net_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_best.pth")

                logging.info("Evaluation Done")
                logging.info(f"Dice: {dice_list_test.mean():.4f}/Best Dice: {best_Dice.mean():.4f}")

                writer.add_scalar('LR', exp_scheduler, epoch+1)

        

    return best_Dice, best_HD, best_ASD



def train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, scaler, args, matcher=None):
    gc.collect()
    elapsed_time_meter = AverageMeter("Elapsed Time", ":6.2f")
    
    net.train()
    start_epoch_time = time.time()  # Track epoch start time
    loss_meters = OrderedDict()
    progress=None
    iter_num_per_epoch = 0
    for i, inputs in enumerate(trainLoader):
        report_embeddings=None
        if 'ufo' in args.dataset:
            img = inputs["image"]
            label = inputs["label"]
            unk_voxels = inputs["unk_channels"].float()
            tumor_volumes_in_crop = inputs["volumes"].float()
            chosen_segment_mask = inputs["mask"].float()
            tumor_diameters = inputs["diameters"].float()
            if "weights" in inputs:
                class_weights = inputs["weights"].float()
            else:
                class_weights = None
            if args.clip_pretrain:
                report_embeddings = inputs['clip_embedding'].float()
                report_embeddings = report_embeddings.cuda(non_blocking=True)
            if not args.model_genesis_pretrain:
                label = label.long()
            try:
                names = inputs['name']
            except KeyError:
                names = None
                
            #print('Tumor volumes in crop returned:', tumor_volumes_in_crop, flush=True, file=sys.stderr)
        else:
            img, label, class_weights = inputs[0], inputs[1], inputs[2].float()
            if not args.model_genesis_pretrain:
                label = label.long()
            unk_voxels, tumor_volumes_in_crop, chosen_segment_mask, tumor_diameters = None, None, None, None
            tumor_volumes_in_crop_per_voxel = None
            tumor_diameters_per_voxel = None
            names=None
            
        if args.model_genesis_pretrain:
            #moved to dataset
            #print(f'Shape of img: {img.shape}')
            #img, label = mg.generate_one_pair(img.cpu().numpy())
            #print(f'Shape of img after model genesis: {img.shape}, label: {label.shape}')
            #img=torch.from_numpy(img).cuda(non_blocking=True)
            #label=torch.from_numpy(label).cuda(non_blocking=True)
            #print('generated pair for model genesis')
            assert img.shape == label.shape, 'Image and label must have the same shape, do you apply model genesis in your dataset?'
            global counter_mg
            if counter_mg<10:
                counter_mg+=1
                #save samples for debugging
                os.makedirs('debug_model_genesis/',exist_ok=True)
                lf.save_tensor_as_nifti(img[0,0],f'debug_model_genesis/{counter_mg}_x.nii.gz')
                lf.save_tensor_as_nifti(label[0,0],f'debug_model_genesis/{counter_mg}_y.nii.gz')
            #we sustitute the image and label by the pair generated by model genesis
        
        #print('Label max and shape:', label.max(), label.shape)
        if args.aug_device != 'gpu':
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            if unk_voxels is not None:
                unk_voxels = unk_voxels.cuda(non_blocking=True)
            if tumor_volumes_in_crop is not None:
                tumor_volumes_in_crop = tumor_volumes_in_crop.cuda(non_blocking=True)
            if chosen_segment_mask is not None:
                chosen_segment_mask = chosen_segment_mask.cuda(non_blocking=True)
            if tumor_diameters is not None:
                tumor_diameters = tumor_diameters.cuda(non_blocking=True)
       
        step = i + epoch * len(trainLoader) # global steps
        
        optimizer.zero_grad()
        assert not torch.isnan(img).any(), 'Input is nan'
        assert torch.max(img)<=100, f'Input is bigger than 100: {torch.max(img)}'
        assert torch.min(img)>=-100, f'Input is smaller than -100: {torch.min(img)}'

        if args.amp:
            raise ValueError('MedFormer seems unstable with amp, please use float32 precision')
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = net(img) 

            loss_all=lf.calculate_loss(model_output=result, label=label, unk_voxels=unk_voxels, args=args,
                                matcher=matcher,chosen_segment_mask=chosen_segment_mask,tumor_volumes_report=tumor_volumes_in_crop, 
                                tumor_diameters=tumor_diameters,
                                classes=trainLoader.dataset.classes,input_tensor=img,
                                class_weights=class_weights if 'class_weights' in locals() else None,
                                model_genesis=args.model_genesis_pretrain,
                                clip_only = args.clip_pretrain, report_embeddings=report_embeddings, dist=dist,
                                ) # pass class_weights if available, otherwise None
            loss=loss_all['overall']

            scaler.scale(loss).backward()
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            # Clip gradients: norm will be at most 1.0.
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            result = net(img) 
            loss_all=lf.calculate_loss(model_output=result, label=label, unk_voxels=unk_voxels, args=args,
                                matcher=matcher,chosen_segment_mask=chosen_segment_mask,tumor_volumes_report=tumor_volumes_in_crop, 
                                tumor_diameters=tumor_diameters,
                                classes=trainLoader.dataset.classes,input_tensor=img,
                                class_weights=class_weights if 'class_weights' in locals() else None,
                                model_genesis=args.model_genesis_pretrain,
                                clip_only = args.clip_pretrain, report_embeddings=report_embeddings, dist=dist,
                                ) # pass class_weights if available, otherwise None

            loss=loss_all['overall']
            loss.backward()
            
            # Clip gradients before stepping the optimizer.
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            

        if args.ema:
            update_ema_variables(net, ema_net, args.ema_alpha, step)

        if len(loss_meters) == 0:
            loss_meters = {k: AverageMeter(k, ":6.4f") for k in loss_all.keys()}
            loss_meters['Elapsed Time'] = AverageMeter("Elapsed Time", ":6.2f")

        for k, v in loss_all.items():
            loss_meters[k].update(v.item(), img.shape[0])

        elapsed_time = time.time() - start_epoch_time

        loss_meters['Elapsed Time'].update(elapsed_time, n=1)

        if progress is None:
            progress = ProgressMeter(
                                    len(trainLoader) if args.dimension == '2d' else args.iter_per_epoch,
                                    list(loss_meters.values()),
                                    prefix=f"{args.unique_name} epoch: [{epoch + 1}]",
                                )

        if i % args.print_freq == 0:
            progress.display(i)
        
        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break

        #torch.cuda.empty_cache()

    if is_master(args):
        for key, meter in loss_meters.items():
            writer.add_scalar(f"Train/{key}", meter.avg, epoch+1)


def get_parser():
    parser = argparse.ArgumentParser(description='CBIM Meidcal Image Segmentation')
    parser.add_argument('--dataset', type=str, default='abdomenatlas_ufo', help='dataset name')
    parser.add_argument('--reports', default=None, help='path to reports')
    parser.add_argument('--model', type=str, default='unet', help='model name')
    parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
    parser.add_argument('--amp', action='store_true', help='if use the automatic mixed precision for faster training')
    
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./exp/', help='the path to save checkpoint and logging info')
    parser.add_argument('--log_path', type=str, default='./log/', help='the path to save tensorboard log')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')
    parser.add_argument('--all_train', action='store_true', help='Uses all dataset in training')
    parser.add_argument('--crop_on_tumor', action='store_true', help='Uses all dataset in training')#use this!
    parser.add_argument('--multi_ch_tumor', action='store_true', help='Use when predicting tumor instances, uses Hungarian algorithm for matching predictions')
    parser.add_argument('--multi_ch_tumor_data_root', type=str, default='/projects/bodymaps/Pedro/data/atlas_300_medformer_multi_ch_tumor_npy/', help='data root for multi channel tumor dataset')
    parser.add_argument('--multi_ch_tumor_classes', type=int, default=61, help='number of classes for multi channel tumor dataset') 
    parser.add_argument('--debug_val',  action='store_true', help='Runs validation before training')
    parser.add_argument('--workers', type=int, default=None, help='overwrites number of workers in config file') 
    parser.add_argument('--load_augmented',  action='store_true', help='Loads pre-saved crops for training. Should speed up things.')  
    parser.add_argument('--save_destination', type=str, default=None, help='destination to save augmented data or to load it')  
    parser.add_argument('--save_augmented', action='store_true', help='Saves after agumentation.')   
    parser.add_argument('--data_root', type=str, default=None, help='data root for dataset')
    parser.add_argument('--UFO_root', type=str, default=None, help='data root for UFO dataset')
    parser.add_argument('--jhh_root', type=str, default=None, help='data root for JHH dataset')
    parser.add_argument('--ucsf_ids', type=str, default=None, help='location of a csv file with the UFO IDs to use for training')

    # NEW DDP arguments
    parser.add_argument('--world_size', type=int, default=1, help='number of nodes for multi-node training')
    parser.add_argument('--rank', type=int, default=0, help='node rank for multi-node training')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:8001', help='url used to set up distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
    
    #report_volume_loss_basic
    parser.add_argument('--report_volume_loss_basic', type=float, default=1, help='weight for the volume loss basic')
    parser.add_argument('--seg_loss', type=float, default=1, help='weight for the volume loss basic')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path') 
    parser.add_argument('--warmup', type=int, default=5, help='number of warmup epochs') 
    parser.add_argument('--loss', type=str, default='ball_dice_last', help='type of loss function to use in reports') 
    parser.add_argument('--classification_branch', action='store_true', help='adds a classification branch to the model bottleneck')
    
    #use the arguments below to load a pre-trained model and fine-tune it with a different class list. It uses output neuron keeping, which preserves weights for common classes across the old and new class lists.
    parser.add_argument('--update_output_layer', action='store_true', help='update the output layer to have the same number of classes as the number of classes in the class_list')
    parser.add_argument('--old_classes', type=str, default=None, help='old classes, we will keep weights/kernels of the old classes. This parameter should be a location of a yaml file with the old classes, we will sort them!')
    
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs to train')
    parser.add_argument('--classes_number', type=int, default=None, help='number of classes')
    parser.add_argument('--ball_bce_weight', type=float, default=1, help='weight for the BCE loss of the ball loss')
    parser.add_argument('--ball_dice_weight', type=float, default=1, help='weight for the Dice loss of the ball loss')
    parser.add_argument('--stardard_ce_ball', action='store_true', help='use standard cross entropy averaging inside the ball loss. Otherwise, we take the average loss for forground and background pixels independently and sum them, giving more weight to avoiding FN.')
    parser.add_argument('--lr', type=float, default=0.0006, help='learning rate')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--balance_pos_neg', action='store_true', help='balance healthy and disease cts')    
    parser.add_argument('--class_weights', action='store_true', help='balance classes by their frequency in the dataset. This will use the inverse frequency of each class to weight the loss function.')
    
    parser.add_argument('--clip_pretrain', action='store_true', help='pretrains with the clip loss')
    parser.add_argument('--clip_source', type=str, default='/projects/bodymaps/Pedro/data/report_embeddings_clinical_longformer/', help='pretrains with the clip loss')
    
    
    parser.add_argument('--no_mask', action='store_true', help='uses no segmentation mask for training, only reports')
    
    #pretrain model genesis
    parser.add_argument('--model_genesis_pretrain', action='store_true', help='skips ALL other losses, just uses model-genesis pre-training')

    parser.add_argument('--pancreas_only', action='store_true', help='trains only on the pancreas')
    parser.add_argument('--kidney_only', action='store_true', help='trains only on the kidney')
    parser.add_argument('--UFO_only', action='store_true', help='trains only on the pancreas')
    parser.add_argument('--Atlas_only', action='store_true', help='trains only on the kidney')
    parser.add_argument('--no_pancreas_subseg', action='store_true', help='blances positives and negatives')
    parser.add_argument('--ball_volume_margin', type=float, default=0.2, help='Margin of tolerance for tumor volume and diameter in the ball loss')
    parser.add_argument('--volume_loss_tolerance', type=float, default=0.2, help='Margin of tolerance for tumor volume and diameter in the ball loss')
    
    #extra classifiers on top of the segmentation output
    parser.add_argument('--tumor_classes',nargs='+',
                        default=None,
                        help="List of tumor types to process"
                        )
    
    parser.add_argument('--crop_size', default=None, type=int, help='If not None, uses a subset of the training set of the specified size')



    args = parser.parse_args()

    reports = args.reports
    dr = args.data_root
    epochs = args.epochs
    ufo_root = args.UFO_root
    jhh_root = args.jhh_root
    w = args.workers
    lr = args.lr
    classes_number = args.classes_number
    
    args.clip_loss = False
    args.load_clip = False

    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    print('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    if args.multi_ch_tumor:
        #overwrites the arguments in config file
        args.classes = args.multi_ch_tumor_classes
        args.data_root = args.multi_ch_tumor_data_root
        print('Using multi channel tumor dataset')
        print(f'Using multi channel tumor, overwriting classes to {args.classes}')
        print(f'Using multi channel tumor, overwriting data root to {args.data_root}')

    if w is not None:
        args.num_workers = w
        print(f'Overwriting number of workers to {w}')
    if dr is not None:
        args.data_root = dr
    if epochs is not None:
        args.epochs = epochs
    if ufo_root is not None:
        args.UFO_root = ufo_root
    if jhh_root is not None:
        args.jhh_root = jhh_root
    if classes_number is not None:
        args.classes = classes_number
    if lr is not None:
        args.base_lr = lr
        print(f'Overwriting learning rate to {lr}')
    if reports is not None:
        args.reports = reports
        print(f'Overwriting reports to {reports}')
        
    if args.model_genesis_pretrain:
        #disable deep supervision
        args.aux_loss = False
        args.classes = 1
        args.classes_number = 1
        
    if args.clip_pretrain:
        #disable deep supervision
        args.clip_loss = True
        args.load_clip = True
        
    if args.crop_size is not None:
        args.training_size = [args.crop_size, args.crop_size, args.crop_size] 
        
    args.batch_size_global = args.batch_size
        
    return args
    


def init_network(args,classes=None,old_classes=None):
    if args.model_genesis_pretrain:
        c = old_classes
        classes = ['model_genesis']
        print('set classes as model_genesis')
    elif args.update_output_layer and (args.pretrained is None):
        c = old_classes # we must load the checkpoint with the old classes
    else:
        c = classes
        
        
    print('Old classes:', old_classes)
    
    net = get_model(args, pretrain=args.pretrain,classes=c)
    

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain,classes=c)
        logging.info("Use EMA model for evaluation")
    else:
        ema_net = None
        
    if args.update_output_layer:
        from model.dim3.medformer import update_output_layer_onk
        print('Classes for onk:', classes)
        net=update_output_layer_onk(net, original_classes=old_classes, new_classes=classes, copy_pancreas=args.no_mask)

        #also update the ema net
        ema_net=update_output_layer_onk(ema_net, original_classes=old_classes, new_classes=classes, copy_pancreas=args.no_mask)
    
    if args.resume:
        resume_load_model_checkpoint(net, ema_net, args)

    #print(net)

    return net, ema_net 





def main_worker(proc_idx, ngpus_per_node, fold_idx, args, result_dict=None, trainset=None, testset=None):
    # seed each process
    if args.reproduce_seed is not None:
        random.seed(args.reproduce_seed)
        np.random.seed(args.reproduce_seed)
        torch.manual_seed(args.reproduce_seed)

        if hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # set process specific info
    args.proc_idx = proc_idx
    args.ngpus_per_node = ngpus_per_node

    # suppress printing if not master
    if args.multiprocessing_distributed and args.proc_idx != 0:
        def print_pass(*args, **kwargs):
            pass

        #builtins.print = print_pass
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + proc_idx
        
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=f"{args.dist_url}",
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.proc_idx)

        # adjust data settings according to multi-processing
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = int((args.num_workers + args.ngpus_per_node - 1) / args.ngpus_per_node)


    args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
    os.makedirs(args.cp_dir, exist_ok=True)
    configure_logger(args.rank, args.cp_dir+f"/fold_{fold_idx}.txt")
    save_configure(args)

    logging.info(
        f"\nDataset: {args.dataset},\n"
        + f"Model: {args.model},\n"
        + f"Dimension: {args.dimension}"
    )
    
    if args.old_classes is not None:
        with open(args.old_classes, 'r') as f:
            old_classes = yaml.load(f, Loader=yaml.SafeLoader)
            #sort--we sorted when saving in nii2npy.py
        args.old_classes = sorted(old_classes)
        old_classes=args.old_classes
    else:
        old_classes = None
    net, ema_net = init_network(args,classes=trainset.classes,old_classes=old_classes)
      
    
    net.to('cuda')
    if args.ema:
        ema_net.to('cuda')
    if args.distributed:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DistributedDataParallel(net, device_ids=[args.proc_idx], find_unused_parameters=False)
        # set find_unused_parameters to True if some of the parameters is not used in forward
        
        if args.ema:
            ema_net = nn.SyncBatchNorm.convert_sync_batchnorm(ema_net)
            ema_net = DistributedDataParallel(ema_net, device_ids=[args.proc_idx], find_unused_parameters=False)
            
            for p in ema_net.parameters():
                p.requires_grad_(False)


    logging.info(f"Created Model")
    best_Dice, best_HD, best_ASD = train_net(net, trainset, testset, args, ema_net, fold_idx=fold_idx)
    
    logging.info(f"Training and evaluation on Fold {fold_idx} is done")
    
    if args.distributed:
        if is_master(args):
            # collect results from the master process
            result_dict['best_Dice'] = best_Dice
            result_dict['best_HD'] = best_HD
            result_dict['best_ASD'] = best_ASD
    else:
        return best_Dice, best_HD, best_ASD
        

        



if __name__ == '__main__':
    # parse the arguments
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.multiprocessing.set_start_method('spawn')
    args.log_path = args.log_path + '%s/'%args.dataset

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed


    ngpus_per_node = torch.cuda.device_count()
    
    
    Dice_list, HD_list, ASD_list = [], [], []
    the_folds=[0]
    for fold_idx in the_folds:
        if args.multiprocessing_distributed:
            with mp.Manager() as manager:
            # use the Manager to gather results from the processes
                result_dict = manager.dict()
                    
                # Since we have ngpus_per_node processes per node, the total world_size
                # needs to be adjusted accordingly
                args.world_size = ngpus_per_node * args.world_size
                trainset = get_dataset(args, mode='train', fold_idx=fold_idx, all_train=args.all_train, crop_on_tumor=args.crop_on_tumor,
                           load_augmented=args.load_augmented, save_destination=args.save_destination,
                           save_augmented=args.save_augmented) 
                testset = get_dataset(args, mode='test', fold_idx=fold_idx)
                # Use torch.multiprocessing.spawn to launch distributed processes:
                # the main_worker process function
                mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, fold_idx, args, result_dict, trainset, testset))
                best_Dice = result_dict['best_Dice']
                best_HD = result_dict['best_HD']
                best_ASD = result_dict['best_ASD']
            args.world_size = 1
        else:
            trainset = get_dataset(args, mode='train', fold_idx=fold_idx, all_train=args.all_train, crop_on_tumor=args.crop_on_tumor,
                           load_augmented=args.load_augmented, save_destination=args.save_destination)  
            testset = get_dataset(args, mode='test', fold_idx=fold_idx)
            # Simply call main_worker function
            best_Dice, best_HD, best_ASD = main_worker(0, ngpus_per_node, fold_idx, args, trainset=trainset, testset=testset)



        Dice_list.append(best_Dice)
        HD_list.append(best_HD)
        ASD_list.append(best_ASD)
    
    #############################################################################################
    # Save the cross validation results
    total_Dice = np.vstack(Dice_list)
    total_HD = np.vstack(HD_list)
    total_ASD = np.vstack(ASD_list)
    

    with open(f"{args.cp_path}/{args.dataset}/{args.unique_name}/cross_validation.txt",  'w') as f:
        np.set_printoptions(precision=4, suppress=True) 
        f.write('Dice\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {Dice_list[i]}\n")
        f.write(f"Each Class Dice Avg: {np.mean(total_Dice, axis=0)}\n")
        f.write(f"Each Class Dice Std: {np.std(total_Dice, axis=0)}\n")
        f.write(f"All classes Dice Avg: {total_Dice.mean()}\n")
        f.write(f"All classes Dice Std: {np.mean(total_Dice, axis=1).std()}\n")

        f.write("\n")

        f.write("HD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {HD_list[i]}\n")
        f.write(f"Each Class HD Avg: {np.mean(total_HD, axis=0)}\n")
        f.write(f"Each Class HD Std: {np.std(total_HD, axis=0)}\n")
        f.write(f"All classes HD Avg: {total_HD.mean()}\n")
        f.write(f"All classes HD Std: {np.mean(total_HD, axis=1).std()}\n")

        f.write("\n")

        f.write("ASD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {ASD_list[i]}\n")
        f.write(f"Each Class ASD Avg: {np.mean(total_ASD, axis=0)}\n")
        f.write(f"Each Class ASD Std: {np.std(total_ASD, axis=0)}\n")
        f.write(f"All classes ASD Avg: {total_ASD.mean()}\n")
        f.write(f"All classes ASD Std: {np.mean(total_ASD, axis=1).std()}\n")



        
    print(f'All {args.k_fold} folds done.')

    sys.exit(0)


