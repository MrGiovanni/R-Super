"""
Here is the core of R-Super. This file contains the new Volume Loss and Ball Loss, which convert tumor information from
radiology reports into per-voxel supervision for semantic segmentation.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import yaml
import nibabel as nib
import math
import importlib
from . import info_nce as nce
import random
from functools import reduce
import copy

def dilate_volume(volume, kernel_size, full_pass_radius=3):
    # ensure odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # for small kernels, just do one pass
    if kernel_size <= (2*full_pass_radius+1):
        return dilate_volume_conv(volume, kernel_size)

    # compute how many "radius‑3" (kernel=7) passes we need
    # radius = (kernel_size‑1)//2  (an integer number of voxels)
    radius = (kernel_size - 1) // 2

    num_full = radius // full_pass_radius  # integer division
    remainder = radius % full_pass_radius  # 0, 1, or 2

    # apply all full radius‑3 passes
    for _ in range(num_full):
        volume = dilate_volume_conv(volume, 2*full_pass_radius + 1)

    # handle the leftover radius if any (1→kernel=3, 2→kernel=5)
    if remainder > 0:
        volume = dilate_volume_conv(volume, 2*remainder + 1)

    return volume



def dilate_volume_conv(volume, kernel_size):
    """
    Applies binary dilation to a 3D binary volume using max pooling.

    Parameters:
        volume (torch.Tensor): The input binary volume with shape
            [batch, channels, depth, height, width]. The volume should be binary (0 or 1).
        kernel_size (int): The size of the cubic structuring element (must be an odd number).

    Returns:
        torch.Tensor: The dilated binary volume with the same shape as the input.
    """
    reduce=0
    if len(volume.shape) == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
        reduce=2
    if len(volume.shape) == 4:
        volume = volume.unsqueeze(0)
        reduce=1
    assert len(volume.shape) == 5, f"Input tensor should be 5D, got {volume.shape}"

    # Ensure the kernel size is odd for proper centering.
    if kernel_size % 2 == 0:
        kernel_size+=1



    # Apply max pooling with stride=1 and the computed padding.
    # This will output a 1 if any voxel in the kernel window is 1 (binary dilation).
    #we can use a maxpool or a ball convolution to dilate the volume. Maxpool should be faster, but it uses a cube kernel, while the ball kernel is more accurate.
    #dilated = F.max_pool3d(volume, kernel_size=kernel_size, stride=1, padding=padding)
    ball_kernel = create_ball_kernel(kernel_size).type_as(volume).unsqueeze(0).unsqueeze(0).repeat(volume.shape[1],1, 1, 1, 1)

    # Calculate padding such that the output size is the same as the input size.
    kernel_size = ball_kernel.shape[-1]
    padding = kernel_size // 2

    dilated = F.conv3d(volume, ball_kernel, padding=padding, groups=volume.shape[1])
    #binarize
    dilated = (dilated > 0).float()

    assert dilated.shape == volume.shape, "Output shape must match input shape."

    if reduce == 1:
        dilated = dilated.squeeze(0)
    elif reduce == 2:
        # Reduce back to original shape if we added extra dimensions.
        dilated = dilated.squeeze(0).squeeze(0)

    return dilated

def dilate_volume_maxpool(volume, kernel_size):
    """
    Applies binary dilation to a 3D binary volume using max pooling.

    Parameters:
        volume (torch.Tensor): The input binary volume with shape
            [batch, channels, depth, height, width]. The volume should be binary (0 or 1).
        kernel_size (int): The size of the cubic structuring element (must be an odd number).

    Returns:
        torch.Tensor: The dilated binary volume with the same shape as the input.
    """
    kernel_size = max(1,int(kernel_size/(2**(0.5))))#compensates for the fact that maxpool is not a round kernel
    if kernel_size%2==0:
        kernel_size+=1

    reduce=0
    if len(volume.shape) == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
        reduce=2
    if len(volume.shape) == 4:
        volume = volume.unsqueeze(0)
        reduce=1
    assert len(volume.shape) == 5, f"Input tensor should be 5D, got {volume.shape}"

    # Ensure the kernel size is odd for proper centering.
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd number for proper alignment.")

    # Calculate padding such that the output size is the same as the input size.
    padding = kernel_size // 2


    # Apply max pooling with stride=1 and the computed padding.
    # This will output a 1 if any voxel in the kernel window is 1 (binary dilation).
    dilated = F.max_pool3d(volume, kernel_size=kernel_size, stride=1, padding=padding)

    assert dilated.shape == volume.shape, "Output shape must match input shape."

    if reduce == 1:
        dilated = dilated.squeeze(0)
    elif reduce == 2:
        # Reduce back to original shape if we added extra dimensions.
        dilated = dilated.squeeze(0).squeeze(0)

    return dilated

counter = 0

def get_known_voxels(y: torch.Tensor, unk_voxels: torch.Tensor, dilation=5,sanity=True, classes = None):
    """
    We cannot calculate the BCE loss for voxels we do not know the ground-truth for.
    This function will output a per-voxel masks showing the known voxels. You can use it to mask the loss (or the output and label).
    Args:
        y (torch.Tensor): Tensor of shape (B, C, H, W, D).
        unk_voxels (torch.Tensor): Tensor of shape (B, C, H, W, D) indicating the regions that have tumors not annotated per voxel for each class. I.e., in this tensor, 1 represents voxels we do not know the per-voxel ground-truth. 
        Zero representas voxels we do know the per-voxel ground-truth.
        dilation (int): Size of the cubic structuring element for dilation. Default is 5.
    """
    unk_voxels=unk_voxels.float()
    assert torch.equal(unk_voxels.bool().float(),unk_voxels), 'unk_voxels must be binary'

    if dilation>0:
        #dilate unk voxels: adds a margin around the unknown voxels
        unk_voxels = dilate_volume(unk_voxels, dilation)

    #print("unk_voxels unique values:", torch.unique(unk_voxels), flush=True)
    #print("unk_voxels sum:", unk_voxels.sum(), flush=True)
    one = torch.ones(unk_voxels.shape).type_as(unk_voxels)
    known_voxels = one-unk_voxels
    known_voxels = known_voxels.type_as(y).float()
    assert torch.equal(known_voxels + unk_voxels,one)

    #print('Sum of known voxels:',known_voxels.sum())
    #print('Sum of unknown voxels:',unk_voxels.sum())
    #print('Sum of all voxels:',one.sum(),'matches?',torch.equal(known_voxels + unk_voxels,one))

    if sanity:
        global counter
        if counter<10:
            debug_save_labels(y,str(counter)+'_y',label_names=classes) 
            debug_save_labels(known_voxels,str(counter)+'_known_voxels',label_names=classes)
            debug_save_labels(unk_voxels,str(counter)+'_unk_voxels',label_names=classes)
            print('Saved to '+ str(counter)+'_known_voxels')
            counter+=1



    #print number of channels with unknown voxels
    #num_unknown_channels = unk_voxels.float().sum(dim=(-1,-2,-3))>0
    #num_unknown_channels = num_unknown_channels.float().sum(-1)
    #num_unknown_channels = num_unknown_channels.mean(0)
    #print("---------Number of channels with unknown voxels: ", num_unknown_channels, flush=True, file=sys.stderr)
    #print("Number of known voxels: ", known_voxels.sum(), flush=True, file=sys.stderr)

    #with open(os.path.join(args.data_root, 'list', 'label_names.yaml'), 'r') as f:
    #    classes = yaml.load(f, Loader=yaml.SafeLoader)

    return known_voxels




def get_lesion_channels(out, classes, assertion = False, return_class_names = False):
    #merge lesion channels if they are in the same organ. Outputs will have only lesion channels, removes organ channels.
    assert out.shape[1] == len(classes)
    #print('Shapes here: ', out.shape, chosen_segment_mask.shape, flush=True, file=sys.stderr)

    lesion_out = {}

    for i,clss in enumerate(classes,0):
        #print('Class is:',clss,'Mask sum is:',chosen_segment_mask[:,i].sum())
        for suffix in ['lesion','cyst','pdac','pnet']:
            if suffix in clss:
                name = clss[:clss.index('_'+suffix)+len('_'+suffix)].replace('pancreatic','pancreas')
                if name not in lesion_out:
                    lesion_out[name] = []
                lesion_out[name].append(out[:,i])

    for key in lesion_out.keys():#this combines multi-channel outputs into a single channel
        lesion_out[key] = torch.stack(lesion_out[key],dim=0).max(dim=0).values
        

    #from dicts to tensor
    kys=list(lesion_out.keys())
    lesion_out = torch.stack([lesion_out[key] for key in kys],dim=1).type_as(out)
    
    if assertion:
        for i in range(lesion_out.shape[0]):
            # For sample i, lo has shape (num_lesion_channels, ...spatial dimensions...)
            lo = lesion_out[i]
            # Sum over all dimensions except the channel, regardless of the number of spatial dims.
            lo_sum = lo.sum(dim=(-1,-2,-3))
            # Create a boolean mask for channels with any nonzero value.
            active_mask = lo_sum > 0
            active_count = active_mask.sum().item()
            if active_count > 1:  # If more than one lesion channel is active
                # Prepare the names of the lesion channels that are active.
                active_names = [kys[j] for j in range(len(kys)) if active_mask[j]]
                raise ValueError(
                    f"Error: For sample index {i}, more than one lesion channel has active elements. "
                    f"Active lesion channels: {active_names}"
                    f"lo.sum(dim=(-1,-2,-3)): {lo.sum(dim=(-1,-2,-3))}"
                )
    if return_class_names:
        return lesion_out, kys
    else:
        return lesion_out

def volume_loss_basic(out,chosen_segment_mask,tumor_volumes, 
                      labels,unk_voxels,
                      classes='/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml',
                      dilation_segment=31, dilation_unk=7, tolerance=0.1,loss_function='deprecated',n='huber',
                      sigmoid=True, class_weights=None):
    """
    Computes the basic tumor volume loss. This loss compares the total predicted tumor volume inside 
    a subsegment with the tumor volume from the report.
    
    Parameters
    ----------
    out : Tensor
        Network segmentation output. Must match *labels* shape.
    chosen_segment_mask : Tensor
        Same shape; 1 where the report’s lesion lies.
    tumor_volumes : Tensor
        `(B,T)` – *T* ≤ 10; volume of tumors in the chosen_segment_mask region.
    tolerance : float
        Relative dead-zone. 0.1 ⇒ no penalty if |V̂−V| ≤ 10 %·V.
    dilation_segment / dilation_unk : int
        Grow masks to compensate for imperfect sub-segment borders and
        annotation gaps (empirically 31 / 7 gave the most stable
        training).
    class_weights : Optional[Tensor]
        `(B,C)` or `(B,C,1,1,1)`; combats class imbalance.
    sigmoid : bool
        Whether to apply sigmoid activation to the output. Set to True if out are logits.

    Returns
    -------
    dict  {"dice_volume_loss": scalar tensor}
    """
    #total tumor volume from the report
    #print('Volume in reports:', tumor_volumes)
    assert len(tumor_volumes.shape) == 2 #batch and maximum of 10 tumors
    assert len(out.shape) == 5
    assert chosen_segment_mask.shape == out.shape
    assert unk_voxels.shape == out.shape
    assert labels.shape == out.shape
    
    if class_weights is not None:
        assert len(class_weights.shape) == 5
        assert class_weights.shape[1] == out.shape[1], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        assert class_weights.shape[0] == out.shape[0], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        #repeat class weights to match the batch size of out
        class_weights = class_weights.repeat(1, 1, out.shape[2], out.shape[3], out.shape[4]) # B,C,H,W,D
        
    #get only the channels with lesions---apply this in the beginning to reduce computational cost of dilation
    out = get_lesion_channels(out, classes)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask, classes,assertion=False)
    labels = get_lesion_channels(labels, classes)
    unk_voxels = get_lesion_channels(unk_voxels, classes)

    #activation
    if sigmoid:
        out = torch.sigmoid(out)

    #dilate the chosen segment mask
    chosen_segment_mask = dilate_volume(chosen_segment_mask,dilation_segment)
    #dilate the unk voxels
    unk_voxels = dilate_volume(unk_voxels,dilation_unk)

    #remove from this loss any channel with a tumor that is annotated per-voxel
    per_voxel_positives = (labels.sum((-1,-2,-3),keepdim=True)>0).float()#B,C, which elements we have a tumor annotated per voxel
    #labels = labels * (1-per_voxel_positives)
    out = out * (1-per_voxel_positives)
    
    #voxels we are sure have no tumor:
    negative_voxels = 1 - ((labels + unk_voxels + chosen_segment_mask) > 0).float() #B,C

    
    if class_weights is not None:
        class_weights = get_lesion_channels(class_weights, classes)
        class_weights = class_weights.mean(dim=(-1,-2,-3)) #reduce to B,C, this will be used to weight the loss for each channel.
    

    #let's get only the subsegment voxels
    assert out.shape == chosen_segment_mask.shape
    assert out.shape == negative_voxels.shape
    out_in_subsegment = out * chosen_segment_mask
    out_in_negative_voxels = out * negative_voxels

    #we have 1 report volume per batch item, but to what class does it refer to? we can use chosen_segment_mask to figure that out
    report_volume = tumor_volumes.sum(-1) # shape B, we sum the multiple tumors we can have
    report_volume = report_volume.unsqueeze(-1).repeat(1,chosen_segment_mask.shape[1])#B,3
    gate=(chosen_segment_mask.sum(dim=(-1,-2,-3))>0).float()#B,3, one in the lesion channel the report volume refers to, 0 otherwise
    #assert gate.shape[-1]==3
    report_volume = report_volume * gate #B,C, only non-zero for lesion we care about in each CT patch


    loss=dice_based_volume_loss(out_in_subsegment,report_volume,tolerance=tolerance,E=500,cross_entropy=False)
    #shape of loss should be B,C
    if class_weights is not None:
        #apply class weights to the loss
        loss = loss * class_weights
    loss = loss.mean()
    loss={'dice_volume_loss':loss}
    #print('Using dice volume loss')
    assert not torch.isnan(loss['dice_volume_loss']).any(), 'loss is nan'
    return loss


def dice_based_volume_loss(x,y,tolerance=0.1,E=500,cross_entropy=False):
    """
    This is the function shown in the miccai paper Figure 1.
    """
    #assert no negative values
    assert torch.min(y).item()>=0
    assert torch.min(x).item()>=0

    #assert no nan
    assert not torch.isnan(x).any(), 'Output is nan'
    assert not torch.isnan(y).any(), 'label is nan'

    #tolerance: return 0 if x/y is within 1+/- tolerance
    if len(x.shape)==5:
        x = x.sum((-1,-2,-3))
    assert len(x.shape)==2, f'shape of x is: {x.shape}'

    predicted_volume = x
    target_volume = y

    assert predicted_volume.shape == target_volume.shape

    loss=torch.abs(predicted_volume-target_volume)/(predicted_volume+target_volume+E)
    #E allows this to work when the ground-truth is zero.

    #subtract the loss at tolerance, for continuity
    v=(1-tolerance)*target_volume
    mini = target_volume.clamp(max=100)
    v    = torch.max(v, mini)
    loss_at_tolerance=torch.abs(v-target_volume)/(v+target_volume+E)

    loss=loss-loss_at_tolerance

    #clamp at zero
    loss=torch.clamp(loss,min=0,max=1)

    if cross_entropy:
        #print('Using cross-entropy')
        loss = -torch.log(torch.ones(loss.shape).type_as(loss)-loss+1e-5)
    else:
        #print('Using dice volume without cross-entropy')
        pass

    return loss

def plot_dice_based_volume_loss(y_value=1000, tolerance=0.1, E=500, num_points=100, x_min=0, x_max=10000,
                                cross_entropy=False):
    """
    Plots the loss for a fixed ground truth volume (y_value) as the predicted volume (x) varies.
    
    y_value : float
        The fixed ground truth volume.
    tolerance : float
        Tolerance percentage (default 0.1 means ±10%).
    E : float
        Offset constant in the denominator.
    num_points : int
        Number of points to sample for predicted volumes.
    x_min, x_max : float
        The range of predicted volumes to consider. If x_max is None, it defaults to 2*y_value.
    """
    import matplotlib.pyplot as plt
    if x_max is None:
        x_max = 2 * y_value  # Default range if not provided

    # Create a series of predicted volume values
    x_values = torch.linspace(x_min, x_max, num_points)
    
    # Create a dummy tensor "x" of shape (num_points, 1, 1, 1)
    # so that summing over the last three dims gives the predicted volume
    x_tensor = x_values.view(num_points, 1, 1, 1)
    
    # Create a target tensor "y" with the same predicted volume for each sample
    y_tensor = torch.full((num_points,), y_value)
    
    # Compute the individual loss values
    loss = dice_based_volume_loss(x_tensor, y_tensor, tolerance=tolerance, E=E, cross_entropy=cross_entropy)
    
    # Plot the loss as a function of the predicted volume
    plt.figure(figsize=(8, 6))
    plt.plot(x_values.numpy(), loss.numpy(), label='Dice-Based Volume Loss')
    plt.xlabel("Predicted Volume (x)")
    plt.ylabel("Loss")
    plt.title(f"Loss vs. Predicted Volume for Ground Truth y = {y_value}")
    plt.legend()
    plt.grid(True)
    plt.show()



def GlobalWeightedRankPooling(x, N=1000, c=0.75, inverse=False, concentrate=1, return_weights=False,hard_cutoff=False):
    """
    Performs Global Weighted Rank Pooling (GWRP). The weights decay exponentially so that
    the top N voxels receive c% of the total weight.
    Ps: the raw weight at voxel N will be 1-c. 
    So, the inverse weight will be c.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        N (int or torch.Tensor): Number of top voxels to concentrate. If an integer, a scalar
                                 value is used; if a tensor of shape (B, C), each (B,C) pair 
                                 uses its own N.
        c (float): Fraction (e.g. 0.9 for 90%) of the total weight to be concentrated in the top N voxels.
    
    Returns:
        torch.Tensor: The pooled tensor of shape (B, C).
    """
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0).unsqueeze(0)
        reduce=True
    assert len(x.shape) == 5, f"Input tensor should be 5D, got {x.shape}"

    B, C, H, W, D = x.shape
    L = H * W * D  # total number of voxels per (B, C)
    
    # Sort the spatial elements in descending order.
    x_sorted, sort_indices = torch.sort(x.view(B, C, L), dim=-1, descending=True)
    
    # Compute the decay factor d.
    # If N is a scalar, convert it to a tensor of shape (B, C) with that constant.
    if not torch.is_tensor(N):
        N_tensor = torch.full((B, C), N, dtype=torch.float32, device=x.device)
    else:
        N_tensor = N.to(x.device).float()
    # Ensure N is at least 1.
    N_tensor = torch.clamp(N_tensor, min=1)
    
    # Compute d elementwise: d = (1-c)^(1/N).
    d = (1 - c) ** (1.0 / N_tensor)  # shape (B, C)
    # Reshape d to (B, C, 1) so it can broadcast.
    d = d.unsqueeze(-1)
    
    # Create an index tensor of shape (1, 1, L).
    indices = torch.arange(L, dtype=torch.float32, device=x.device).view(1, 1, L)
    
    # Compute weights: each weight is d^(i), broadcasting over (B, C).
    weights_raw = d ** indices  # shape (B, C, L)
    weights = weights_raw / weights_raw.sum(dim=-1, keepdim=True)  # normalize to sum to 1

    #assert that, for a random B,C element, the sum of the first N weights is equal to c
    #rand_b=torch.randint(0,B,(1,))
    #rand_c=torch.randint(0,C,(1,))
    #summed = weights[rand_b, rand_c, :int(N_tensor[rand_b, rand_c].item())].sum()  
    #assert abs(summed.item() - c) < 0.2

    if inverse:
        # For the inverse case we want to ignore the top N voxels.
        # Create a mask that is 0 for indices < N and 1 for indices >= N.
        mask_inv = (indices >= N_tensor.unsqueeze(-1)).float()  # shape (B, C, L)
        # Use the complementary weights for the background: here we use (1 - weights_raw)
        weights = mask_inv * (1 - weights_raw)
        # Note: We do not normalize these weights to sum to 1 because the goal here is to measure
        # the background (i.e. the voxels outside the top N).
    elif concentrate!=1:
        assert concentrate>1, 'concentrate must be greater than 1'
        # Create two masks: one for the top N voxels and one for the rest.
        mask_top = (indices < N_tensor.unsqueeze(-1)).float()      # 1 for indices < N, 0 otherwise
        mask_rest = (indices >= N_tensor.unsqueeze(-1)).float()     # 1 for indices >= N, 0 otherwise
        # Leave top N voxels unchanged and scale the rest by (1/concentrate)
        new_weights = mask_top * weights + mask_rest * (weights / concentrate)
        # Renormalize the weights so they sum to 1.
        weights = new_weights / new_weights.sum(dim=-1, keepdim=True)

    if return_weights:
        if hard_cutoff:
            #make all weights after N zero and re-normalize
            mask_top = (indices < N_tensor.unsqueeze(-1)).float()
            weights = mask_top * weights
            weights = weights / weights.sum(dim=-1, keepdim=True)
        # We need to return the weights reorganized into the original spatial order.
        # sort_indices tells us, for each (B, C, i), which voxel in the unsorted order that value came from.
        # Compute the inverse permutation.
        inverse_indices = sort_indices.argsort(dim=-1)
        # unsort the weights so that they align with the original order.
        weights_unsorted = weights.gather(dim=-1, index=inverse_indices)
        # Reshape to original spatial dimensions.
        weights_unsorted = weights_unsorted.view(B, C, H, W, D)
        if reduce:
            weights_unsorted = weights_unsorted.squeeze(0).squeeze(0)
        return weights_unsorted
    
    # Compute weighted sum and normalize by the sum of weights.
    pooled = (x_sorted * weights).sum(dim=-1)

    return pooled



def DiceLossMultiClass(preds, targets, known_voxels, alpha = 0.5, beta=0.5, size_average=True, reduce=True, sigmoid=True, class_weights=None):

    if len(preds.shape)==3:
        preds=preds.unsqueeze(0).unsqueeze(0)
    if len(targets.shape)==3:
        targets=targets.unsqueeze(0).unsqueeze(0)
    if len(known_voxels.shape)==3:
        known_voxels=known_voxels.unsqueeze(0).unsqueeze(0)

    if len(preds.shape)==4:
        preds=preds.unsqueeze(0)
        targets=targets.unsqueeze(0)
        known_voxels=known_voxels.unsqueeze(0)

    assert len(preds.shape)==5
    assert (preds.shape == targets.shape) and (targets.shape == known_voxels.shape), f"Shapes do not match, pred, target and unk are: {preds.shape}, {targets.shape}, {known_voxels.shape}"

    N = preds.size(0)
    C = preds.size(1)
    
    if sigmoid:
        P = torch.sigmoid(preds)
    else:
        P = preds

    P = P * known_voxels
    targets = targets * known_voxels

    smooth = 1e-5

    class_mask = targets

    ones = torch.ones(P.shape).to(P.device)
    P_ = ones - P 
    class_mask_ = ones - class_mask

    TP = P * class_mask
    FP = P * class_mask_
    FN = P_ * class_mask

    alpha = FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) / ((FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) + FN.transpose(0, 1).reshape(C, -1).sum(dim=(1))) + smooth)
    alpha = alpha.unsqueeze(0).repeat(N, 1) # repeat for each batch item, now alpha is B,C

    alpha = torch.clamp(alpha, min=0.2, max=0.8) 
    #print('alpha:', alpha)
    beta = 1 - alpha
    num = torch.sum(TP, dim=(-1,-2,-3)).float()
    den = num + alpha * torch.sum(FP, dim=(-1,-2,-3)).float() + beta * torch.sum(FN, dim=(-1,-2,-3)).float()

    dice = num / (den + smooth)
    loss = 1 - dice
    if class_weights is not None:
        class_weights = class_weights.mean(dim=(-1,-2,-3))
        while len(class_weights.shape) < len(loss.shape):
            class_weights = class_weights.unsqueeze(0)
        assert class_weights.shape == loss.shape, f'Class weights shape {class_weights.shape} does not match the shape of dice loss {loss.shape}'
        # Apply class weights
        loss = loss * class_weights
    
    if not reduce:
        return loss

    if size_average:
        assert len(loss.shape) == 2, f'Loss should be 2D after reduction, but got {loss.shape}.'
        loss = loss.mean()  # Average over the batch size

    return loss

counter2=0


    

def classification_loss(cls_out, label, unk_voxels, args, chosen_segment_mask, classes, class_weights=None):
    #calculate classification loss
        
    if False:
        lesion_idx = [i for i, class_name in enumerate(classes) if (('background' in class_name) or ('pdac' in class_name) or ('pnet' in class_name) or ('cyst' in class_name))]
        lesion_labels = label[:, lesion_idx].float()
        if chosen_segment_mask is not None:
            lesion_labels += chosen_segment_mask[:, lesion_idx].float()
        #print('Lesion labels shape:', lesion_labels.shape)
        #class should be the class of the center voxel
        lesion_labels = lesion_labels[:, :, lesion_labels.shape[2]//2, lesion_labels.shape[3]//2, lesion_labels.shape[4]//2]
        #assert single label
        assert len(lesion_labels.shape)==2, f'Lesion labels shape is: {lesion_labels.shape}'
        assert lesion_labels.sum(dim=1).max()<=1, f'Lesion labels should be single label, but got {lesion_labels.sum(dim=1).max()}'
        #print('Lesion labels:', lesion_labels)
        target_idx = lesion_labels.argmax(dim=1).long() # (B,)
        #print('Lesion label:', target_idx)
        #print('cls_out out:', cls_out.shape)
        #background class? no cyst? lesion? what!?
    else:
        lesion_idx = [i for i, class_name in enumerate(classes) if ('lesion' in class_name)]
        #print(f'Classification loss for classes: {[classes[i] for i in lesion_idx]}')
        lesion_labels = label[:, lesion_idx].float()
        #multi-class
        if chosen_segment_mask is not None:
            lesion_labels += chosen_segment_mask[:, lesion_idx].float()
        lesion_labels = (lesion_labels.sum(dim=(-1,-2,-3))>0).float()
    #now check chosen_segment_mask
    assert len(cls_out.shape)==2 and cls_out.shape[0]==label.shape[0], f'Classification output shape is: {cls_out.shape}, label shape is: {label.shape}'
    if False:
        #softmax
        #for i in range(target_idx.shape[0]):
        #    print('Target idx:', target_idx[i], 'cls_out:', cls_out[i],
        #           'class:', classes[target_idx[i]])
        cls_loss   = F.cross_entropy(cls_out, target_idx,reduction='none')
    else:
        #sigmoid
        cls_loss = F.binary_cross_entropy_with_logits(cls_out, lesion_labels, reduction='none', weight=class_weights)
        #print(f'Labels: {lesion_labels[0]}')
        #print(f'cls_out: {cls_out[0]}')
        #print(f'cls_loss: {cls_loss[0]}')
    #if channels with unknown voxels are present and their label is 0, remove them from the loss (multiply by 0)
    if unk_voxels is not None:  
        unk_labels = (unk_voxels[:, lesion_idx].sum(dim=(-1,-2,-3))>0).float()
        #where unk_labels is 1 and label is 0:
        known_labels = (1-unk_labels)+lesion_labels
        known_labels = (known_labels>0).float()
        cls_loss = cls_loss * known_labels
    cls_loss = cls_loss.mean()
    #print('Classification loss:', cls_loss)
    return cls_loss


def model_genesis_loss(result,label):
    #MSE voxel-wise loss
    if isinstance(result, tuple) or isinstance(result, list):
        raise ValueError('Turn off deep supervision for model genesis pretraining')
    l = torch.nn.functional.mse_loss(result,label, reduction='mean')
    loss={'genesis_loss': l,
          'overall': l}
    return loss
    
def all_gather_tensor(x, dist):
    """
    Gathers `x` across all DDP ranks *with* autograd support.
    If we're not in a distributed context or only one GPU, just returns x.
    """
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        # Single‐GPU / non‐DDP path: no gather needed
        return x

    world = dist.get_world_size()
    # Prepare list of tensors to gather into
    with torch.no_grad():
        gathered = [torch.zeros_like(x) for _ in range(world)]
        # This call *does* record the collective in autograd, so gradients will flow
        dist.all_gather(gathered, x)
    #print(f'Sum of gathered tensors: {sum([g.sum() for g in gathered])}')
    rank = dist.get_rank()
    gathered[rank] = x #the grad flows through here
    # Concatenate along the batch dimension
    return torch.cat(gathered, dim=0)

def merge_no_overlap(d1, d2):
    overlap = d1.keys() & d2.keys()
    if overlap:
        raise KeyError(f"Cannot merge: duplicate keys found: {overlap}")
    return {**d1, **d2}


def calculate_loss(model_output, label, unk_voxels, args, matcher,chosen_segment_mask,
                   tumor_volumes_report,tumor_diameters,
                   classes,input_tensor=None, class_weights=None, model_genesis=False,
                   clip_only=False,report_embeddings=None, dist=None):
    """
    This function calculates all of our loss functions, i.e., the segmentation loss (dice and BCE), and the report supervision
    losses (volume and ball loss), and any baseline loss (clip, classification, models genesis).
    
    Compute *all* losses used by R-Super for a single training step.

    The routine aggregates five possible loss terms
    (segmentation, volume, ball, classification, contrastive) and
    returns them in a dictionary **plus** an `"overall"` key that is the
    weighted sum of all active losses. You can simply back-propagate this 
    overall loss.

    ----------
    Mandatory inputs
    ----------------
    model_output : dict
        What the network returned for the current batch.
        * **segmentation** – required. Either  
          `(B, C, H, W, D)` **or** a *tuple/list* of such tensors when
          deep supervision is enabled.  B is batch size, C is number of classes.
        * **classification** – present only when
          `args.classification_branch` is *True*; shape `(B, Nc)`.  
          This is for the MTL baseline.
        * **clip** – present only when *clip-only* contrastive learning
          is requested. This is for the CLIP-Like baseline.

    label : torch.FloatTensor  
        Voxel-wise one-hot labels with shape `(B, C, H, W, D)` where
        `C == len(classes)`.

    unk_voxels : torch.FloatTensor  
        Same shape as *label*. 1 → the voxel is **unknown** (should be
        *masked-out* from standard segmentation losses), 0 → ground-truth is
        known. Must be binary. unk_voxels marks organs where reports mention
        tumors, but we have no per-voxel annotations for these tumors.
        These are the cases to be optimized by the volume and ball losses.

    args : argparse.Namespace  
        Experiment hyper-parameters and feature flags.  Only the
        following fields are accessed by this pared-down version:  
        `loss`, `aux_weight`, `seg_loss`, `report_volume_loss_basic`,
        `volume_loss_tolerance`, `ball_bce_weight`, `ball_dice_weight`,
        `multi_ch_tumor`, `stardard_ce_ball`.

    matcher : Callable  
        Hungarian matcher used when `args.multi_ch_tumor` is *True*.  
        Signature `out_ids, label_ids = matcher(pred, gt)`. 
        Not used in the MICCAI paper.

    chosen_segment_mask : torch.FloatTensor  
        Binary mask of voxels belonging to the organ sub-segment in
        which tumors resides. Same shape as *label*. This is the organ
        with tumor(s) where our data loader cropped at. We will use the
        volume and ball losses to make the AI find the tumor inside 
        this organ/sub-segment.

    tumor_volumes_report : torch.FloatTensor  
        `(B, T)` where *T* is the max # tumours per crop, set to 10. 
        Each entry is the radiology-report
        volume in voxels. From reports, our data loader calculates volumes in
        mm3. If you use 1x1x1 spacing like we did, this is fine. If not, 
        you must use your voxel spacing to convert mm3 to voxels.

    tumor_diameters : torch.FloatTensor  
        `(B, T, 3)` – three orthogonal diameters (mm) for every tumour.
        Again, we use 1x1x1 spacing. If you use other spacing, you will need
        to convert these diameters to your voxel spacing in the x y axes.
        (using 1x1x1 makes this easier).

    classes : list[str]  
        Alphabetically-sorted class names; length **C**.  Used for
        consistency checks and to locate lesion channels.

    ----------
    Optional
    ------------------------

    input_tensor : torch.Tensor, optional  
        Raw CT patch `(B, 1, H, W, D)` only needed for saving debug
        images when `counter3 < 10`.

    class_weights : torch.FloatTensor, optional  
        Per-sample, per-class weights `(B, C)`.  Will be broadcast to
        `(B, C, H, W, D)` internally.  **Must** be either *None* or a
        true weight map (the code removes it if it equals `torch.ones`).  

    model_genesis : bool, default=False  
        When *True* the function returns a simple voxel-wise MSE loss
        (`model_genesis_loss`) and skips every other path. This is 
        used for the Model Genesis pretraining.

    clip_only : bool, default=False  
        Activates the symmetric InfoNCE contrastive loss between CT
        patch embeddings and report embeddings. Requires
        `model_output['clip']`, *report_embeddings*, and a valid
        `torch.distributed` *dist* object.
        This is used for the CLIP-Like pretraining.

    report_embeddings : torch.Tensor, dist : torch.distributed  
        Only used when *clip_only* is *True*.
        
    -------------
    Important `args` fields still used
    ----------------------------------
    loss (str)               – Selects report-loss type:  
                             ball_dice_last: default, ball (last layer) and volume (deep supervision) losses
                             dice: volume loss only (the name is strange, but it is because we were somewhat inspired by the dice loss to develop this loss)
                             ball: ball loss only

    aux_weight (list[float]) – Per-decoder weight when deep supervision
                            is enabled (one weight per head).

    seg_loss (float)         – Weight for the BCE+Dice
                            segmentation loss.

    report_volume_loss_basic (float)
                            – Weight for all report-driven losses (ball and volume);
                            0 ⇒ those losses are skipped.

    volume_loss_tolerance (float)
                            – ±relative band where the dice-volume loss = 0 (e.g. 0.1 → ±10 %). 
                            Use 0.1 for 10% (what we used).

    ball_bce_weight (float)  – Relative weight of the BCE term inside
                            `ball_loss`. Deafult is 1.

    ball_dice_weight (float) – Relative weight of the Dice term inside
                            `ball_loss`. Deafult is 1.

    multi_ch_tumor (bool)    – Not used in the paper, set to False.

    stardard_ce_ball (bool)  – If True, ball loss uses the “standard”
                            single mean for BCE; otherwise it averages
                            foreground & background separately.

    ----------
    Returned value
    --------------
    dict
        Every active loss under a descriptive key **plus**:
        * `'segmentation'` – BCE-with-logits + Dice (masked)  
        * `'report'` (or the two keys produced by *ball loss*)  
        * `'classification'` (only if `args.classification_branch`)  
        * `'contrastive_loss'` (only in *clip-only* mode)  
        * `'overall'` – sum of all above (or custom wrapper output)
    """
    global counter2
    #print('Unk voxels:', unk_voxels)
    
    if model_genesis:
        return model_genesis_loss(model_output['segmentation'],label)
    
    if clip_only:
        seg_result = model_output['segmentation']
        result_embedding = model_output['clip']
        if isinstance(seg_result, tuple) or isinstance(seg_result, list):
            tmp = 0
            for i in range(len(seg_result)):
                tmp = tmp + seg_result[i].sum()*0
        result_embedding = result_embedding+tmp*0#this is to avoid unused parameter error
        result_embedding = all_gather_tensor(result_embedding, dist)
        report_embeddings = all_gather_tensor(report_embeddings, dist)
        #assert same shape (all dimensions)
        assert result_embedding.shape == report_embeddings.shape, f'Result embedding shape is: {result_embedding.shape}, report embedding shape is: {report_embeddings.shape}'
        loss_ct2rep = nce.info_nce(result_embedding, report_embeddings)
        loss_rep2ct = nce.info_nce(report_embeddings, result_embedding)
        sym_loss = 0.5*(loss_ct2rep + loss_rep2ct)
        sym_loss = sym_loss*dist.get_world_size() #this compensated for the all_gather
        return {'contrastive_loss': sym_loss,
                'overall': sym_loss}
       
    y_class=None
    result = model_output['segmentation']
    if args.classification_branch:
        y_class = model_output['classification']
        #print('Classification shape:', y_class.shape)
    
    if chosen_segment_mask is not None and chosen_segment_mask.sum()>0:
        for b in range(chosen_segment_mask.shape[0]):
            if unk_voxels[b].sum()==0 and chosen_segment_mask[b].sum()>0:
                raise ValueError('unk_voxels should not be all zeros if chosen_segment_mask is not all zeros')
            if tumor_volumes_report[b].sum() == 0 and chosen_segment_mask[b].sum()>0:
                raise ValueError('tumor_volumes_report should not be all zeros if chosen_segment_mask is not all zeros')
    
    #raise ValueError(f'Number of classes in classes: {len(classes)}. Number of classes in label: {label.shape[1]}. Number of classes in result: {result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1]}')
    assert len(classes) == label.shape[1], f'Number of classes in classes: {len(classes)} does not match the number of channels in label: {label.shape[1]}'
    assert len(classes) == (result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1]), \
    f'Number of classes in result: {(result[0].shape[1] if isinstance(result, (tuple, list)) else result.shape[1])} does not match the number of channels in label: {label.shape[1]}'
    
    if class_weights is not None and torch.equal(class_weights, torch.ones_like(class_weights)):
        class_weights = None
        
    if class_weights is not None:
        class_weights = class_weights.to(label.device) #make sure class weights are the same size as cls_out
        assert  class_weights.shape[0] == label.shape[0], f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        assert class_weights.shape[1] == label.shape[1], f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        assert len(class_weights.shape) == 2, f'Class weights should be 2D, but got {class_weights.shape}'
        
    cls_loss = None
    if (y_class is not None):
        cls_loss = classification_loss(y_class, label, unk_voxels, args, chosen_segment_mask, classes, class_weights)

    loss = 0
    loss_report = 0
    loss_segmentation = 0
    if class_weights is not None:
        class_weights = class_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        assert len(class_weights.shape)==len(label.shape), f'Class weights shape {class_weights.shape} does not match label shape {label.shape}'
        
    if isinstance(result, tuple) or isinstance(result, list):
        # if use deep supervision, add all loss together---outpput: [final output, hidden output]
        if unk_voxels is not None:
            known_voxels = get_known_voxels(label,unk_voxels,classes=classes)#this will remove (substitute by 0) any channels we are unsure if about the label
            assert torch.equal((known_voxels*label).float().sum(),label.float().sum()), f'The unknown region should not cover channels where our label is different from 0---knwon. we got{(known_voxels*label).float().sum()} and {label.float().sum()}'
            known_voxels_original = known_voxels.clone()
            #print('Assertion successful')
        else:
            known_voxels=torch.ones(label.shape).type_as(label)
        for j in range(len(result)):
            r,l = result[j],label
            if j==0 and args.multi_ch_tumor:
                #hungarian algorithm---run on the final output, use same indices on hidden layer outputs
                out_ids, label_ids = matcher(r,l)
            if args.multi_ch_tumor:
                #shuffle accorind to hungarian algo output
                r=r[out_ids]
                l=l[label_ids]
                known_voxels = known_voxels_original[label_ids]
                unk_voxels = unk_voxels[label_ids]  # Inside the multi_ch_tumor block
                chosen_segment_mask = chosen_segment_mask[label_ids]
                assert r.shape == known_voxels.shape, f'Label mismatch, known voxels is: {known_voxels.shape}, r is: {r.shape}'
                assert torch.equal(known_voxels[out_ids],known_voxels[label_ids]),'Known voxels should be the same accross the label channels, which are the ones the hungarian algo is shifting around'
            
            #assert no nan in output
            assert not torch.isnan(r).any(), 'Output is nan'
            if args.report_volume_loss_basic > 0:
                if ('ball' in args.loss or 'dynamic' in args.loss or 'dll' in args.loss) and not (j!=0 and 'last' in args.loss):
                    #j!=0 and 'last' in args.loss=>applies the ball loss only to the last layer
                    #print('Using the ball loss')
                    loss_r = ball_loss (out=r, labels=l, unk_voxels=unk_voxels, chosen_segment_mask=chosen_segment_mask, 
                                        tumor_volumes=tumor_volumes_report, tumor_diameters=tumor_diameters, classes=classes, 
                                        apply_dice_loss=('dice' in args.loss), input_tensor=input_tensor,
                                        sigmoid=True,
                                        standard_ce=args.stardard_ce_ball, class_weights = class_weights,
                                        single_class= False,
                                        diameter_margin=args.ball_volume_margin, volume_margin=args.ball_volume_margin)
                    if 'both' in args.loss:
                        loss_r = merge_no_overlap(loss_r,volume_loss_basic(r, chosen_segment_mask, tumor_volumes_report, l, unk_voxels, classes, loss_function=args.loss,
                                               sigmoid=True, class_weights = class_weights,tolerance=args.volume_loss_tolerance))
                        #print('Both')
                else:
                    loss_r = volume_loss_basic(r, chosen_segment_mask, tumor_volumes_report, l, unk_voxels, classes, loss_function=args.loss,
                                               sigmoid=True, class_weights = class_weights,tolerance=args.volume_loss_tolerance)
                    #print('Using the volume loss')
            else:
                loss_r = torch.tensor(0).type_as(r)


            loss_seg = F.binary_cross_entropy_with_logits(r, l.float(), reduction='none', weight=class_weights)
            
            assert loss_seg.shape == known_voxels.shape, f'Loss shape {loss_seg.shape} does not match known voxels shape {known_voxels.shape}'
            if counter2<5 and j==0:
                label_names = classes
                debug_save_labels(torch.sigmoid(r),str(counter2),out_dir='SanityOutputs',label_names=label_names)
                debug_save_labels(l.float(),str(counter2),out_dir='SanityLabelsBeforeLoss',label_names=label_names)
                debug_save_labels(loss_seg,str(counter2),out_dir='SanityLossBCE',label_names=label_names)
                debug_save_labels(loss_seg*known_voxels,str(counter2),out_dir='SanityLossBCEAfterKnownVoxels',label_names=label_names)
                counter2+=1
            loss_seg = loss_seg * known_voxels
            loss_seg = loss_seg.mean() + DiceLossMultiClass(r, l, known_voxels, sigmoid=True,class_weights=class_weights)
            loss_segmentation = loss_segmentation + args.aux_weight[j] * args.seg_loss * loss_seg

            if not isinstance(loss_r, dict):
                loss_report = loss_report + args.aux_weight[j] * args.report_volume_loss_basic * loss_r
            else:
                if isinstance(loss_report,int):
                    loss_report = {}
                    for key in loss_r.keys():
                        if key == 'ball_loss_bce':
                            weight = args.ball_bce_weight
                            #print(f'Using the ball bce weight: {weight}')
                        elif key == 'ball_loss_dice':
                            weight = args.ball_dice_weight
                            #print(f'Using the ball dice weight: {weight}')
                        else:
                            weight = 1
                        loss_report[key] = args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
                else:#dict
                    for key in loss_r.keys():
                        if key == 'ball_loss_bce':
                            weight = args.ball_bce_weight
                            #print(f'Using the ball bce weight: {weight}')
                        elif key == 'ball_loss_dice':
                            weight = args.ball_dice_weight
                            #print(f'Using the ball dice weight: {weight}')
                        else:
                            weight = 1
                        if key not in list(loss_report.keys()):
                            loss_report[key] = args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
                        else:
                            loss_report[key] = loss_report[key] + args.aux_weight[j] * args.report_volume_loss_basic * weight * loss_r[key]
    else:
        #raise ValueError('Result is not a tuple or list, you should be using deep supervision')
        if unk_voxels is not None:
            known_voxels = get_known_voxels(label,unk_voxels,classes=classes)#this will remove (substitute by 0) any channels we are unsure if about the label
            assert torch.equal((known_voxels*label).float().sum(),label.float().sum()), 'The unknown region should not cover channels where our label is different from 0---knwon'
        else:
            known_voxels=torch.ones(label.shape).type_as(label)

        if args.multi_ch_tumor:
            out_ids, label_ids = matcher(result,label)
            result=result[out_ids]
            label=label[label_ids]
            assert result.shape == known_voxels.shape
            known_voxels = known_voxels[out_ids]
            unk_voxels = unk_voxels[label_ids]  # Inside the multi_ch_tumor block
            chosen_segment_mask = chosen_segment_mask[label_ids]
            assert torch.equal(known_voxels[out_ids],known_voxels[label_ids]),'Known voxels should be the same accross the label channels, which are the ones the hungarian algo is shifting around'
        
        #assert no nan in output
        assert not torch.isnan(result).any(), 'Output is nan'
        if args.report_volume_loss_basic > 0:
            if 'ball' in args.loss or 'dynamic' in args.loss or 'dll' in args.loss:
                #j!=0 and 'last' in args.loss=>applies the ball loss only to the last layer
                loss_r = ball_loss (out=result, labels=label, unk_voxels=unk_voxels, chosen_segment_mask=chosen_segment_mask, 
                                    tumor_volumes=tumor_volumes_report, tumor_diameters=tumor_diameters, classes=classes, 
                                    apply_dice_loss=('dice' in args.loss),sigmoid=True,
                                    standard_ce=args.stardard_ce_ball,class_weights=class_weights,
                                    single_class= False,
                                    diameter_margin=args.ball_volume_margin, volume_margin=args.ball_volume_margin)
                if 'both' in args.loss:
                    loss_r = merge_no_overlap(loss_r,volume_loss_basic(result,chosen_segment_mask,tumor_volumes_report, 
                                           label, unk_voxels, classes, loss_function=args.loss,
                                           sigmoid=True, class_weights=class_weights,tolerance=args.volume_loss_tolerance))
                    #print('Both')
            else:
                loss_r = volume_loss_basic(result,chosen_segment_mask,tumor_volumes_report, 
                                           label, unk_voxels, classes, loss_function=args.loss,
                                           sigmoid=True, class_weights=class_weights,tolerance=args.volume_loss_tolerance)
        else:
            loss_r = torch.tensor(0).type_as(result)


        loss_seg = F.binary_cross_entropy_with_logits(result, label.float(), reduction='none', weight=class_weights) #use BCE with logits for the segmentation loss
    
        assert loss_seg.shape == known_voxels.shape

        loss_seg = loss_seg * known_voxels
        loss_seg = loss_seg.mean() + DiceLossMultiClass(result, label, known_voxels, sigmoid=True,class_weights=class_weights)
        loss_segmentation = loss_segmentation + args.seg_loss * loss_seg
        if not isinstance(loss_r, dict):
            loss_report = loss_report + args.report_volume_loss_basic * loss_r
        else:
            if isinstance(loss_report,int):
                loss_report = {}
            for key in loss_r.keys():
                if key == 'ball_loss_bce':
                    weight = args.ball_bce_weight
                    #print(f'Using the ball bce weight: {weight}')
                elif key == 'ball_loss_dice':
                    weight = args.ball_dice_weight
                    #print(f'Using the ball dice weight: {weight}')
                else:
                    weight = 1
                loss_report[key] = args.report_volume_loss_basic * weight * loss_r[key]
                
    loss={'segmentation':loss_segmentation}
    if isinstance(loss_report,dict):
        for key in loss_report.keys():
            loss[key] = loss_report[key]
    else:
        loss['report'] = loss_report
        
    if cls_loss is not None:
        loss['classification'] = cls_loss

    loss_overall = 0
    for key in loss.keys():
        #print('loss key:', key)
        loss_overall = loss_overall + loss[key]
    

    loss['overall']=loss_overall
    if torch.isnan(loss_overall).any():
        raise ValueError('loss is nan, propagating this can destroy the network weights, STOP!')

    #check if loss_overall requires grad
    assert loss_overall.requires_grad, 'Loss overall should require grad'

    return loss

def debug_save_labels(labels: torch.Tensor,
                      name='',
                      label_names = '/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml',
                      out_dir: str = "./LossChecking",
                      batch_idx = 0):
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
    #raise ValueError(f'Label names is: {label_names}')
    
    # 2. Load and sort label names
    if not isinstance(label_names, list):
        with open(label_names, "r") as f:
            label_names = yaml.safe_load(f)  # e.g. ["liver", "kidney", "pancreas", ...]
        
    label_names_sorted = sorted(label_names)  # sort alphabetically
    
    # 3. Basic shape check
    if len(labels.shape)==4:
        labels = labels.unsqueeze(0)

    if labels.shape[1]!=len(label_names_sorted):
        raise ValueError(f"Number of channels in labels ({labels.shape[1]}) does not match the number of label names ({len(label_names_sorted)}). Labels loaded from: {label_names}. ")
        label_names = '/projects/bodymaps/Pedro/data/atlas_300_medformer_multi_ch_tumor_npy/list/label_names.yaml'
        with open(label_names, "r") as f:
            label_names = yaml.safe_load(f)
        label_names_sorted = sorted(label_names)
    
    assert len(labels.shape) == 5
    B, C, H, W, D = labels.shape
    assert batch_idx < B, f"batch_idx={batch_idx} is out of range for B={B}."
    if C != len(label_names_sorted):
        label_names_sorted = [str(i) for i in list(range(C))]
    
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
        out_path = os.path.join(out_dir, f"{name}_{channel_label_name}.nii.gz")

        #print(f'Saving: {out_path}, its sum is {channel_data.sum()}')
        
        # Save
        nib.save(nifti_img, out_path)
        
    print(f"Saved to {out_path}")








############### BALL LOSS ####################

def create_ball_kernel(diameter, gaussian=False, gaussian_std=1.5):
    """
    Creates a 3D torch tensor (kernel) where there is a 'ball' of a given diameter.
    The diameter is first rounded up to the next odd integer. The kernel size is then
    computed to be 1.2 × (that odd diameter), rounded to the next odd integer.
    
    The ball is centered in this larger kernel. Inside the ball (hard cutoff at the
    ball boundary), values are set to 1 (or to a truncated Gaussian if `gaussian=True`).
    Outside the ball, values are 0. If `gaussian=True`, the Gaussian is centered at
    the ball center with standard deviation `gaussian_std * radius`.

    Parameters
    ----------
    diameter : float or int
        Desired diameter of the ball. Will be rounded up to the next odd integer.
    gaussian : bool, optional
        Whether to fill the ball with a Gaussian distribution, by default False.
    gaussian_std : float, optional
        Standard deviation factor (relative to the ball radius) if gaussian=True.
        For example, if the ball's radius is R and gaussian_std=1.5, the std is
        1.5*R, by default 1.5.

    Returns
    -------
    kernel : torch.FloatTensor
        A 3D tensor of shape (kernel_size, kernel_size, kernel_size) containing
        the ball (or Gaussian ball) centered in the kernel.
    """

    # --- Step 1: Round diameter to next odd integer ---
    diameter_ceil = math.ceil(diameter)
    if diameter_ceil % 2 == 0:
        diameter_ceil += 1
    diameter_odd = diameter_ceil  # The final odd diameter
    
    # --- Step 2: Compute kernel size as 1.2 * diameter_odd, also round up to next odd ---
    kernel_size_float = 1.2 * diameter_odd
    kernel_size_ceil = math.ceil(kernel_size_float)
    if kernel_size_ceil % 2 == 0:
        kernel_size_ceil += 1
    kernel_size = kernel_size_ceil  # The final odd kernel size
    
    # Ball radius (float)
    radius = diameter_odd / 2.0

    # --- Create 1D coordinate grid from 0..(kernel_size-1), shift so center is 0 ---
    center = (kernel_size - 1) / 2.0
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords_shifted = coords - center  # center at 0
    
    # --- Compute squared distance (3D) via broadcasting ---
    distance_squared = (coords_shifted[:, None, None] ** 2
                      + coords_shifted[None, :, None] ** 2
                      + coords_shifted[None, None, :] ** 2)
    
    # --- Hard cutoff mask for the ball ---
    mask = (distance_squared <= radius**2).float()
    
    if gaussian:
        # Scale std by the ball's actual radius
        std = gaussian_std * radius
        gaussian_values = torch.exp(-distance_squared / (2.0 * std**2))
        kernel = gaussian_values * mask
        # Normalize so that sum of kernel = 1
        kernel = kernel / kernel.sum()
    else:
        kernel = mask  # Binary ball kernel

    #assert the kernel size is odd
    assert kernel.shape[0] % 2 == 1, f'Kernel size should be odd, got {kernel.shape[0]}'
    
    return kernel


def save_ball_kernel(diameter, gaussian, gaussian_std, filename):
    """
    Wrapper function that creates a ball kernel using `create_ball_kernel`,
    prints the center and border values, and saves the kernel as a .nii.gz file.
    
    Args:
        diameter (int): Diameter of the ball.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball.
        gaussian_std (float): Standard deviation of the Gaussian.
        filename (str): Path for saving the NIfTI file (should end with .nii.gz).
    """
    # Create the kernel
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std)
    
    # Determine the center index (assuming symmetric kernel)
    center_idx = diameter // 2
    center_value = kernel[center_idx, center_idx, center_idx].item()
    
    # Determine the border value as the smallest nonzero value inside the ball.
    # (This should correspond roughly to the values at the edge.)
    border_value = kernel[kernel > 0].min().item()
    
    print(f"Center value: {center_value}")
    print(f"Border value: {border_value}")
    
    # Convert to numpy array (nibabel works with numpy)
    kernel_np = kernel.numpy()
    
    # Create a default affine (identity) matrix
    affine = np.eye(4)
    
    # Create and save the NIfTI image
    nii_img = nib.Nifti1Image(kernel_np, affine)
    nib.save(nii_img, filename)
    print(f"Saved ball kernel to {filename}")

def ball_convolution(x,diameter,gaussian, gaussian_std):
    """
    Performs a 3D convolution on the input tensor `x` using a ball kernel of diameter `diameter`.
    Optionally, the values inside the ball can follow a Gaussian distribution with standard deviation `gaussian_std`.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball.
        gaussian_std (float): Standard deviation of the Gaussian.
    
    Returns:
        torch.Tensor: Convolved tensor of shape (B, C, H, W, D).
    """
    #if diameter is not odd, add 1:
    if diameter%2==0:
        diameter+=1

    # Create the ball kernel
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std).type_as(x)
    
    # Convert kernel to 5D tensor (B=1, C=1, H, W, D)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Perform the 3D convolution
    out = F.conv3d(x, kernel, padding=kernel.shape[-1]//2)

    assert out.shape == x.shape, f'Output shape should be the same as input shape, got {out.shape} and {x.shape}'
    return out

def insert_ball_old(out_spatial,best_center,diameter,margin):
    # Use a binary (non-Gaussian) ball kernel.
    binary_ball_kernel = create_ball_kernel(diameter*(1+margin), gaussian=False)
    #we add the margin only here, we do not use the margin in the convolution, for better detection.
    
    # Create an empty volume for the ball mask with the same spatial shape as x.
    masked_volume = torch.zeros_like(out_spatial)
    H, W, D = masked_volume.shape
    d_half = binary_ball_kernel.shape[-1] // 2
    cx, cy, cz = best_center

    # For each dimension, compute the overlapping indices between the input volume and the ball kernel.
    # X-dimension:
    vol_x_min = max(0, cx - d_half)
    vol_x_max = min(H, cx + d_half + 1)
    mask_x_min = 0 if cx - d_half >= 0 else -(cx - d_half)
    mask_x_max = mask_x_min + (vol_x_max - vol_x_min)

    # Y-dimension:
    vol_y_min = max(0, cy - d_half)
    vol_y_max = min(W, cy + d_half + 1)
    mask_y_min = 0 if cy - d_half >= 0 else -(cy - d_half)
    mask_y_max = mask_y_min + (vol_y_max - vol_y_min)

    # Z-dimension:
    vol_z_min = max(0, cz - d_half)
    vol_z_max = min(D, cz + d_half + 1)
    mask_z_min = 0 if cz - d_half >= 0 else -(cz - d_half)
    mask_z_max = mask_z_min + (vol_z_max - vol_z_min)

    # Place the binary ball kernel into the masked_volume at the computed overlapping region.
    masked_volume[vol_x_min:vol_x_max, vol_y_min:vol_y_max, vol_z_min:vol_z_max] = \
        binary_ball_kernel[mask_x_min:mask_x_max, mask_y_min:mask_y_max, mask_z_min:mask_z_max]
    return masked_volume

def insert_ball(out_spatial, best_center, diameter, margin):
    """
    Places a 'ball' of size diameter * (1 + margin) into out_spatial at the 3D coordinate best_center.
    The 3D ordering is assumed to be (z, y, x).
    """
    # 1) Build the ball kernel for insertion
    binary_ball_kernel = create_ball_kernel(diameter*(1+margin), gaussian=False)

    # 2) Prepare an empty volume with same shape as out_spatial
    masked_volume = torch.zeros_like(out_spatial)
    
    # 3) Extract shape in (z, y, x) order
    Z, Y, X = masked_volume.shape
    
    # 4) The kernel half-width
    d_half = binary_ball_kernel.shape[-1] // 2
    
    # 5) Unpack best_center as (cz, cy, cx)
    cz, cy, cx = best_center
    
    # 6) Compute overlap in Z dimension
    vol_z_min = max(0, cz - d_half)
    vol_z_max = min(Z, cz + d_half + 1)
    mask_z_min = 0 if cz - d_half >= 0 else -(cz - d_half)
    mask_z_max = mask_z_min + (vol_z_max - vol_z_min)

    # 7) Compute overlap in Y dimension
    vol_y_min = max(0, cy - d_half)
    vol_y_max = min(Y, cy + d_half + 1)
    mask_y_min = 0 if cy - d_half >= 0 else -(cy - d_half)
    mask_y_max = mask_y_min + (vol_y_max - vol_y_min)

    # 8) Compute overlap in X dimension
    vol_x_min = max(0, cx - d_half)
    vol_x_max = min(X, cx + d_half + 1)
    mask_x_min = 0 if cx - d_half >= 0 else -(cx - d_half)
    mask_x_max = mask_x_min + (vol_x_max - vol_x_min)

    # 9) Place the kernel region into masked_volume
    masked_volume[
        vol_z_min:vol_z_max,
        vol_y_min:vol_y_max,
        vol_x_min:vol_x_max
    ] = binary_ball_kernel[
        mask_z_min:mask_z_max,
        mask_y_min:mask_y_max,
        mask_x_min:mask_x_max
    ]

    return masked_volume

def isolate_tumor(x, diameter, gaussian, gaussian_std, tumor_volume,
                  diameter_margin=0.5,volume_margin=0.5):
    """
    Uses a ball convolution over x and applies a maximum operation to find the best
    fitting ball center. Then, it multiplies the input by a volume with the same size
    as the input, but with a binary ball placed at the given object center coordinate.
    Finally, after the multiplication, we find the top N voxels inside the remaining volume.
    N is the tumor volume.

    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
        diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting inside the ball (for convolution).
        gaussian_std (float): Standard deviation of the Gaussian.
        tumor_volume (int): Number of voxels to select as the tumor volume.
    
    Returns:masked_volume should be within 
        torch.Tensor: A binary tumor mask of shape (H, W, D) with 1's in the top N voxels.
    """
    reduce=False
    if len(x.shape)==3:
        reduce=True
        x = x.unsqueeze(0).unsqueeze(0)
    assert len(x.shape) == 5, f"Input tensor should be 5D, got {x.shape}"

    #round diameter
    diameter = np.round(diameter).astype(int)
    #round tumor volume
    tumor_volume = np.round(tumor_volume).astype(int)

    # Ensure the diameter is odd.
    if diameter % 2 == 0:
        diameter += 1

    # Create the ball kernel for convolution.
    kernel = create_ball_kernel(diameter, gaussian, gaussian_std).type_as(x)
    # Convert kernel to a 5D tensor (shape: 1, 1, H, W, D).
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    #assert volume is not larger than the number of voxels in the ball
    if tumor_volume > 100000:
        assert tumor_volume <= (kernel>0).sum()*1.2, f'Tumor volume should be smaller than the number of voxels in the ball, got {tumor_volume} and {(kernel>0).sum()}'

    if (kernel>0).sum() > tumor_volume:
        #we tolerate numerical erros within a margin of 0.2
        tumor_volume = (kernel>0).sum()-1

    
    # Perform 3D convolution.
    out = F.conv3d(x, kernel, padding=kernel.shape[-1] // 2)

    assert out.shape == x.shape, f"Output shape should match input shape, got {out.shape} vs {x.shape}"

    # --- Step 1: Find the best fitting ball center ---
    # Assume x is of shape (1, 1, H, W, D); take the spatial part.
    out_spatial = out[0, 0]  # shape: (H, W, D)
    max_idx = torch.argmax(out_spatial)
    best_center = np.unravel_index(max_idx.item(), out_spatial.shape)  # (cx, cy, cz)
    
    # --- Step 2: Create a binary ball mask at the best center ---
    masked_volume = insert_ball(out_spatial,best_center,diameter,diameter_margin)
    new_dim = diameter
    while masked_volume.sum() < tumor_volume:
        #if the ball is in the border of the image, its volume may be less than the tumor volume, We increase the size of the ball until we reach the tumor volume.
        old_dim = new_dim
        new_dim = int(np.round(new_dim * 1.1))
        print(f'Increasing ball size to {new_dim}, current volume is {masked_volume.sum()}, tumor volume is {tumor_volume}')
        if old_dim == new_dim:
            new_dim += 1
        if new_dim % 2 == 0:
            new_dim += 1
        if new_dim >= max(x.shape[-1], x.shape[-2], x.shape[-3]):
            break
        masked_volume = insert_ball(out_spatial,best_center,new_dim,diameter_margin)
    if tumor_volume < (50**3):
        assert (masked_volume.sum() > tumor_volume*0.5), f'masked_volume should be within 20% of the tumor volume! got {masked_volume.sum()} and {tumor_volume}'
    if tumor_volume > (6**3):
        assert (masked_volume.sum() < tumor_volume*((1+diameter_margin)**3)*2), f'masked_volume should be within 20% of the tumor volume! got {masked_volume.sum()} and {tumor_volume} and diameter {diameter}'

    # --- Step 3: Multiply the input by the binary ball mask ---
    # x has shape (B, C, H, W, D); expand masked_volume to match.
    #assert no negative value in x
    assert (x >= 0).all(), f'Input tensor should not have negative values, got {x.min()}'
    masked_x = (x * masked_volume.unsqueeze(0).unsqueeze(0))

    # --- Step 4: Find the top N voxels in the masked volume ---
    # Remove batch and channel dimensions.
    masked_x_vol = masked_x[0, 0]
    flattened = masked_x_vol.view(-1)
    # Get indices of the top N voxel values.
    t=min(flattened.shape[-1]-1, tumor_volume)
    margin_small = min(0.5,volume_margin)
    t_small = int(t*(1-margin_small))
    t_small =  max(t_small, min(100,tumor_volume))  # Ensure at 4mm tumor
    t_big = min(flattened.shape[-1]-1,int(tumor_volume*(1+volume_margin)))
    topN_values, topN_indices = torch.topk(flattened, t)
    topN_values_small, topN_indices_small = torch.topk(flattened, t_small)
    topN_values_big, topN_indices_big = torch.topk(flattened, t_big)
    #how many indices? Assert this matches the tumor volume
    assert len(topN_indices) == t, f'Expected {tumor_volume} indices, got {len(topN_indices)}'
    # Create a binary volume: set top N positions to 1, rest to 0.
    tumor_mask_flat = torch.zeros_like(flattened)
    tumor_mask_flat[topN_indices] = 1
    tumor_mask_flat_small = torch.zeros_like(flattened)
    tumor_mask_flat_small[topN_indices_small] = 1
    tumor_mask_flat_big = torch.zeros_like(flattened)
    tumor_mask_flat_big[topN_indices_big] = 1
    
    # Reshape to original spatial dimensions.
    tumor_mask = tumor_mask_flat.view_as(masked_x_vol)
    tumor_mask_small = tumor_mask_flat_small.view_as(masked_x_vol)
    tumor_mask_big = tumor_mask_flat_big.view_as(masked_x_vol)
    # Assert the sum here still matches the tumor volume.
    assert tumor_mask.sum() == t, f'Tumor mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'

    
    #ensure no tumor_max value is outside the ball
    tumor_mask = tumor_mask * masked_volume
    tumor_mask_small = tumor_mask_small * masked_volume
    tumor_mask_big = tumor_mask_big * masked_volume

    if reduce:
        tumor_mask = tumor_mask.squeeze(0).squeeze(0)

    iters = 0
    while tumor_volume < (50**3) and tumor_mask.sum() < tumor_volume*0.7:
        #zero values inside the ball may not be chosen as the top N voxels. In such cases, we dilate the mask
        print(f'dilating tumor mask, iteration {iters}, current volume is {tumor_mask.sum()}, tumor volume is {tumor_volume}')
        if iters >5:
            return tumor_mask, tumor_mask_small, tumor_mask_big
        #dilate the mask
        tumor_mask = dilate_volume(tumor_mask, 7)*masked_volume
        tumor_mask_small = dilate_volume(tumor_mask_small, 7)*masked_volume
        tumor_mask_big = dilate_volume(tumor_mask_big, 7)*masked_volume
        iters+=1

    if tumor_volume < (50**3):
        assert (tumor_mask.sum() > tumor_volume*0.5), f'tumor_mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'
    if tumor_volume > (5**3):
        assert (tumor_mask.sum() < tumor_volume*((1+volume_margin)**3)*3), f'tumor_mask should have the same volume as the tumor volume, got {tumor_mask.sum()} and {tumor_volume}'

    #assert it is binary
    assert (tumor_mask == 0).sum() + (tumor_mask == 1).sum() == tumor_mask.numel(), f'Tumor mask should be binary, got {tumor_mask.sum()}'

    return tumor_mask, tumor_mask_small, tumor_mask_big


counter3=0

def ball_loss(out, labels, unk_voxels, chosen_segment_mask, tumor_volumes, tumor_diameters, classes, apply_dice_loss,
              diameter_margin=0.2, volume_margin=0.2, gaussian=True, 
              gaussian_std=1.5, gwrp=True, gwrp_concentration=0.5, dilation_for_background=7,
              subseg_dilation=31,input_tensor=None, unk_dilation=1,
              sigmoid=True, standard_ce=False, class_weights=None,
              single_class=False, use_small_pseudo_mask=True):
    """
    This funciton first uses a ball loss to isolate the tumor. Then, it selects the top N voxels inside the ball as a "pseudo-label" and applies BCE loss per-voxel.
    Optionally, we can average the per-voxel BCE loss using GWRP weights calculated for the isolated tumor voxels. This will give more emphasis in increasing high confidence voxels.
    Args:
    x is the model output
    tumor_diameter is a tensor of size B,T,3, batch, number of tumors in the crop, and 3 diameters
    diameter_margin: how much much we want the ball diameter to be bigger than the maximum tumor diameter
    gaussian: if a gaussian kernel is used in the ball convolution for better centering on the tumor
    gaussian_std: the higher, the smaller the difference between the ball kernel center and border values
    gwrp: wether to use GRWP to average each BCE loss. If so, more weight is given to increasing high confidence voxels.
    sigmoid: wether to apply sigmoid to the output.
    dilation_for_background: we apply a dilation kernel of this size to the tumor pseudo-mask, and define everything outside this mask as background, and use BCE loss to make the backgropund 0
    subseg_dilation: how much we dilate the tumor subsegment. Radiologists/AI may not be super precise when defining the subsegment, and tumors may grow out of organs, so we add a generous margin here.
    standard_ce: if True, we use a standard averaging for the BCE loss. Otherwise, we acerage the foreground and background voxel losses separately, and then sum the two losses.
    class_weights: optional 5D tensor to apply class weights. This is useful when dealing with imbalanced positives and negatives per class or datasets with many classes.
    Important: this loss assumes the output resolution is 1x1x1 mm, and that diammeters are in mm and volumes in mm^3. If the resolution is different, you should adjust the diameters and volumes accordingly or introduce a scaling factor.
    """
    global counter3

    #total tumor volume from the report
    #print('Volume in reports:', tumor_volumes)
    assert len(tumor_volumes.shape) == 2 #batch and maximum of 10 tumors
    assert len(out.shape) == 5
    assert chosen_segment_mask.shape == out.shape
    assert unk_voxels.shape == out.shape
    assert labels.shape == out.shape
    if class_weights is not None:
        assert class_weights.shape[1] == out.shape[1], f'Class weights shape {class_weights.shape} does not match output shape {out.shape}'
        assert len(class_weights.shape) == 5, f'Class weights should be 5D tensor, got {class_weights.shape}'
        #repeat channels to match the output shape
        class_weights = class_weights.repeat(out.shape[0], 1, out.shape[2], out.shape[3], out.shape[4])


    #get only the channels with lesions
    out = get_lesion_channels(out, classes)
    chosen_segment_mask = get_lesion_channels(chosen_segment_mask, classes, assertion=False)
    unk_voxels = get_lesion_channels(unk_voxels, classes)
    labels = get_lesion_channels(labels, classes)
    if class_weights is not None:
        class_weights = get_lesion_channels(class_weights, classes)

    chosen_segment_mask = dilate_volume(chosen_segment_mask,subseg_dilation)
    #dilate the unk voxels
    unk_voxels = dilate_volume(unk_voxels,unk_dilation)
    to_penalize = torch.ones_like(out)
    #remove the unk voxels from the penalization
    to_penalize = to_penalize * (1 - unk_voxels)
    #also remove the knwon labels
    to_penalize = to_penalize * (1 - labels)
    #but add back the chosen segment mask
    to_penalize = to_penalize + chosen_segment_mask
    #binarize
    to_penalize = (to_penalize > 0).float()


    #let's get only the subsegment voxels
    assert out.shape == chosen_segment_mask.shape

    losses = []
    losses_dice = []

    for B in range(out.shape[0]):#batch itens
        #assert diameters and violumes make sense
        assert torch.equal(tumor_diameters[B].sum(-1)>0, tumor_volumes[B]>0), f'Tumor diameters and volumes should be consistent, got {tumor_diameters[B]} and {tumor_volumes[B]}'
        
        #get correct batch and class
        x = out[B]
        tumor_seg = chosen_segment_mask[B]
        #current_x is still 4 D, with one class per tumor type. Assert at most one of these channels is non-zero (due to the chosen_segment_mask):
        assert (tumor_seg.sum((-1,-2,-3))>0).float().sum()<=1, f'Only one channel should be non-zero, got {tumor_seg.sum((-1,-2,-3))}'
        
        # if no tumor in this batch, create a zero pseudo label
        if tumor_seg.sum()==0 or tumor_volumes[B].sum()==0:
            # no tumor in this batch, create a zero pseudo label
            pseudo_mask = torch.zeros_like(x)
            if sigmoid:
                if not single_class:
                    #standard, use sigmoid
                    loss = F.binary_cross_entropy_with_logits(x, pseudo_mask, reduction='none')
                else:
                    #use softmax
                    loss = F.cross_entropy(x, pseudo_mask, reduction='none')
                #print('ball loss uses BCE with logits')
            else:
                if not single_class:
                    #assert x is in the range 0-1
                    assert (x>=0).all() and (x<=1).all(), f'Output is not in the range 0-1, its min is: {x.min()}, its max is: {x.max()}'
                    #assert pseudo_mask is in the range 0-1
                    assert (pseudo_mask>=0).all() and (pseudo_mask<=1).all(), f'Pseudo mask is not in the range 0-1, its min is: {pseudo_mask.min()}, its max is: {pseudo_mask.max()}'
                    loss = F.binary_cross_entropy(x, pseudo_mask, reduction='none')
                else:
                    #single class, but consider that softmax was already applied. Thus, use nll loss
                    #from one-hot to class indices: argmax
                    loss = F.nll_loss(x, pseudo_mask.argmax(dim=1), reduction='none')
            assert loss.shape == tumor_seg.shape
            loss = loss * to_penalize[B]
            if class_weights is not None:
                # apply class weights if provided
                loss = loss * class_weights[B]
            loss = loss.mean()
            if apply_dice_loss:
                if class_weights is not None:
                    w = class_weights[B]
                else:
                    w = None
                dice_loss = DiceLossMultiClass(preds=x, targets=pseudo_mask, known_voxels=to_penalize[B],sigmoid=sigmoid, class_weights=w).mean()
                losses_dice.append(dice_loss)
            losses.append(loss.mean())
            continue
        
        #get tumor class
        for c in range(x.shape[0]):
            if tumor_seg[c].sum()>0:
                x = x[c]
                penalize = to_penalize[B][c]
                if class_weights is not None:
                    c_weight = class_weights[B][c] #get the class weights for this batch and class
                else:
                    c_weight = None
                break
        tumor_seg = tumor_seg.sum(0)
        current_tumor_diameters = tumor_diameters[B]
        current_tumor_volumes = tumor_volumes[B]

        # Get the sort indices for tumor_volumes in descending order
        sorted_indices = torch.argsort(current_tumor_volumes, descending=True)

        # Filter indices to keep only those with volume > 0
        sorted_indices = sorted_indices[current_tumor_volumes[sorted_indices] > 0]
        #print('--------Sorted indices:', sorted_indices)
        #print('--------SORTED VOLUMES:', current_tumor_volumes[sorted_indices])
        #print('--------UNSORTED VOLUMES:', current_tumor_volumes)

        #Create the pseudo-mask
        pseudo_masks = []
        pseudo_masks_small = []
        pseudo_masks_big = []
        #update x for the next tumor: remove pseudo_mask, so that this tumor is not selected again.
        if sigmoid:
            x_iter = torch.sigmoid(x)*tumor_seg
        else:
            x_iter = x*tumor_seg
        for tumor_idx in sorted_indices:
            vol=current_tumor_volumes[tumor_idx].item()
            dia=current_tumor_diameters[tumor_idx]
            #get the maximum diameter
            max_diameter = torch.max(dia).item()
            assert max_diameter>0, f'Tumor diameter should be larger than 0, got {max_diameter}'
            assert vol>0, f'Tumor volume should be larger than 0, got {vol}'
            if vol==0 or max_diameter == 0:
                print('Found 0 tumor where it should not be')
                continue
            if max_diameter <= 1:
                print('Found 1mm diameter, increasing to 3')
                max_diameter = 3
            if vol <= 1:
                print('Found 1mm volume, increasing to 9')
                vol = 9
            #assert it is not zero
            #ball convolution: use isolate_tumor to get the top 'tumor_volume' voxels in the outpus, inside the best fitting ball position
            pseudo_mask,pseudo_mask_small,pseudo_mask_big = isolate_tumor(x_iter, diameter=max_diameter, 
                                                                          gaussian=gaussian, gaussian_std=gaussian_std, tumor_volume=vol,
                                                                          diameter_margin=diameter_margin,volume_margin=volume_margin)
            pseudo_masks.append(pseudo_mask)
            pseudo_masks_small.append(pseudo_mask_small)
            pseudo_masks_big.append(pseudo_mask_big)
            x_iter = x_iter * (1 - pseudo_mask) #remove the pseudo mask from the output, so that it is not selected again
        #stack the pseudo masks
        if use_small_pseudo_mask:
            pseudo_mask = torch.stack(pseudo_masks_small).sum(0)
        else:
            pseudo_mask = torch.stack(pseudo_masks).sum(0)
        pseudo_mask = (pseudo_mask > 0).float()
        dilated_pseudo_mask = torch.stack(pseudo_masks_big).sum(0)
        dilated_pseudo_mask = (dilated_pseudo_mask > 0).float()

        #we can add a tolerance margin around the pseudo mask, where we do not penalize the outputs for not being zero
        if dilation_for_background>0:
            dilated_pseudo_mask=dilate_volume(dilated_pseudo_mask, dilation_for_background)
            
        border = dilated_pseudo_mask - pseudo_mask
        #threshold at 0
        border = (border > 0).float()
            
        penalize=penalize * (1 - border)
        #penalize is a tensor with the voxels where we want to apply our losses to here

        #BCE loss with mask
        if sigmoid:
            if not single_class:
                BCE = F.binary_cross_entropy_with_logits(x, pseudo_mask, reduction='none')
            else:
                #single class
                BCE = F.cross_entropy(x, pseudo_mask, reduction='none')
        else:
            if not single_class:
                #assert x is in the range 0-1
                assert (x>=0).all() and (x<=1).all(), f'Output is not in the range 0-1, its min is: {x.min()}, its max is: {x.max()}'
                #assert pseudo_mask is in the range 0-1
                assert (pseudo_mask>=0).all() and (pseudo_mask<=1).all(), f'Pseudo mask is not in the range 0-1, its min is: {pseudo_mask.min()}, its max is: {pseudo_mask.max()}'
                BCE = F.binary_cross_entropy(x, pseudo_mask, reduction='none')
            else:
                #single class, but consider that softmax was already applied. Thus, use nll loss
                #from one-hot to class indices: argmax
                BCE = F.nll_loss(x, pseudo_mask.argmax(dim=1), reduction='none')
        assert (penalize.shape==BCE.shape), f'To penalize and BCE should have the same shape, got {penalize.shape} and {BCE.shape}'
        BCE = BCE * penalize #cut the loss gradient in the border. Remember that unk voxels were already removed from x

        #dice loss
        #dice loss
        if apply_dice_loss:
            #remove tumor surroundings, to avoid penalizing them: we are not super sure if this region is tumor or not.
            dice_loss = DiceLossMultiClass(preds=x, targets=pseudo_mask, known_voxels=penalize,sigmoid=sigmoid,class_weights=c_weight)
            #if sigmoid:
            #    print('Dice loss:',dice_loss, 'Mean prediction:',torch.sigmoid(x).mean())
            #else:
            #    print('Dice loss:',dice_loss, 'Mean prediction:',x.mean())
            #we make all voxels knwon because we alreay removed unknown voxels from x
            #print('Using dice loss inside the ball loss')

        if not standard_ce:
            #we separate foreground and background, calculate the average per-voxel loss for them separatelly, than sum it. We can use GRWP in the foreg. or not.
            if gwrp:
                #we do BCE for the entire channel, but we do not simply average it. We can use GWRP to average the tumor values (positive GT)
                #we add the pseudo-mask to boost its voxels values and concentrate GWRP there.
                assert pseudo_mask.sum() > 0, f'Pseudo mask should have at least one voxel, got {pseudo_mask.sum()}, volume is {vol} and diameter is {max_diameter}'
                if sigmoid:
                    foreg_weights = GlobalWeightedRankPooling(torch.sigmoid(x)*pseudo_mask+pseudo_mask, N=pseudo_mask.sum(), c=gwrp_concentration,return_weights=True,
                                                                hard_cutoff=True)
                else:
                    foreg_weights = GlobalWeightedRankPooling(x*pseudo_mask+pseudo_mask, N=pseudo_mask.sum(), c=gwrp_concentration,return_weights=True,
                                                                hard_cutoff=True)
                #print highest and lowest non-zero values in foreg_weights
                assert foreg_weights.sum() > 0.95 and foreg_weights.sum() < 1.05, f'GWRP weights should be normalized to 1, got {foreg_weights.sum()}'
                #renormlize gwrp weights so they sum to pseudo_mask.sum()
                foreg_weights = foreg_weights * pseudo_mask.sum()
                #print('GWRP Foreg weights range:', foreg_weights[foreg_weights>0].max(), foreg_weights[foreg_weights>0].min())
                #assert sum of foreg_weights is close to 1
                foreg_weights = foreg_weights*pseudo_mask
                assert BCE.shape == foreg_weights.shape, f'BCE and GWRP weights should have the same shape, got {BCE.shape} and {foreg_weights.shape}'
                loss_foreground = (BCE*foreg_weights)#.mean() #we can use mean here because 
            else:
                #print('Using simple mean for BCE loss')
                loss_foreground = (BCE*pseudo_mask)#.mean()
            
            #Background:
            bkg_weights = 1 - dilated_pseudo_mask
            loss_background = (BCE*bkg_weights)#.mean()
            
            if c_weight is not None:
                # apply class weights to the BCE loss
                assert len(c_weight.shape) == len(loss_background.shape), f'Class weights shape {c_weight.shape} does not match BCE shape {BCE.shape}'
                assert c_weight.shape[0] == loss_background.shape[0], f'Class weights {class_weights[B].shape} do not match loss_background shape {loss_background.shape}'
                loss_foreground = loss_foreground * c_weight
                loss_background = loss_background * c_weight
            loss_foreground = loss_foreground.mean()
            loss_background = loss_background.mean()

            loss = loss_foreground + loss_background
            losses.append(loss)#BCE loss
        else:
            #print('Using standard CE for BCE loss')
            if c_weight is not None:
                # apply class weights to the BCE loss
                assert len(c_weight.shape) == len(BCE.shape), f'Class weights shape {c_weight.shape} does not match BCE shape {BCE.shape}'
                assert c_weight.shape[0] == BCE.shape[0], f'Class weights {c_weight.shape} do not match BCE shape {BCE.shape}'
                BCE = BCE * c_weight
            BCE = BCE.mean()
            losses.append(BCE)#simple mean.

        if apply_dice_loss:
            losses_dice.append(dice_loss.mean())

        if counter3<10:

            counter3+=1
            os.makedirs('SanityBallLoss/'+str(counter3), exist_ok=True)
            if sigmoid:
                save_tensor_as_nifti(torch.sigmoid(x),'SanityBallLoss/'+str(counter3)+'/x')
            else:
                save_tensor_as_nifti(x,'SanityBallLoss/'+str(counter3)+'/x')
            save_tensor_as_nifti(pseudo_mask,'SanityBallLoss/'+str(counter3)+'/pseudo_mask')
            save_tensor_as_nifti(border,'SanityBallLoss/'+str(counter3)+'/border')
            save_tensor_as_nifti(tumor_seg,'SanityBallLoss/'+str(counter3)+'/tumor_segment')
            save_tensor_as_nifti((to_penalize[B].sum(0)>0).float(),'SanityBallLoss/'+str(counter3)+'/to_penalize')
            if input_tensor is not None:
                save_tensor_as_nifti(input_tensor[B].squeeze(),'SanityBallLoss/'+str(counter3)+'/input_volume')

            #save tumor volumes and diameters as yaml
            with open('SanityBallLoss/'+str(counter3)+'/tumor_volumes.yaml', 'w') as file:
                yaml.dump(tumor_volumes.tolist(), file)
            with open('SanityBallLoss/'+str(counter3)+'/tumor_diameters.yaml', 'w') as file:
                yaml.dump(tumor_diameters.tolist(), file)
            print('Saved to '+ 'SanityBallLoss/'+ str(counter3)+'/known_voxels')
            l=losses[-1].item()
            if apply_dice_loss:
                l+=losses_dice[-1].item()
            if sigmoid:
                info=f'Volume in output: {torch.sigmoid(x).sum().item()}, Volume in report: {vol}, Loss: {l}'
            else:
                info=f'Volume in output: {x.sum().item()}, Volume in report: {vol}, Loss: {l}'
            print(info)
            #save the loss as yaml
            with open('SanityBallLoss/'+str(counter3)+'/loss.yaml', 'w') as file:
                yaml.dump(l, file)
            #save the info as yaml
            with open('SanityBallLoss/'+str(counter3)+'/info.yaml', 'w') as file:
                yaml.dump(info, file)
            print('Saved to '+ 'SanityBallLoss/'+ str(counter3)+'/loss.yaml')

    return {'ball_loss_bce':torch.stack(losses).mean(),
            'ball_loss_dice':torch.stack(losses_dice).mean() if apply_dice_loss else torch.zeros_like(torch.stack(losses).mean())}


def save_tensor_as_nifti(tensor: torch.Tensor, filename: str):
    """
    Saves a torch tensor as a NIfTI file, assuming a voxel spacing of 1x1x1 mm.

    Args:
        tensor (torch.Tensor): A torch tensor of shape (H, W, D) or (1, H, W, D).
        filename (str): The output filename (should end with .nii or .nii.gz).
    """
    if 'nii.gz' not in filename:
        filename += '.nii.gz'
        
    assert len(tensor.squeeze(0).shape)==3, f"Input tensor should be 3D, got {tensor.shape}"

    # Ensure tensor is on CPU and convert to numpy array.
    np_array = tensor.detach().cpu().numpy()
    
    # If the tensor has an extra channel dimension, squeeze it.
    if np_array.ndim == 4 and np_array.shape[0] == 1:
        np_array = np_array.squeeze(0)
    
    # Create an identity affine (voxel sizes = 1 mm in all directions).
    affine = np.eye(4)
    
    # Create the NIfTI image and save.
    nifti_img = nib.Nifti1Image(np_array, affine)
    nib.save(nifti_img, filename)
    print(f"Saved NIfTI file to {filename}")


def apply_ball_convolution_and_save(input_size=(64, 64, 64), square_size=20,
                                    ball_diameter=15, gaussian=False, gaussian_std=3.0,
                                    output_filename='ball_convolution_output.nii.gz'):
    """
    Creates an input tensor with a centered cube (i.e., a 3D "square"),
    applies the ball convolution to it, prints the center coordinates of the input,
    prints the center of mass of the output, and saves the result as a NIfTI file.
    
    Args:
        input_size (tuple): Size of the 3D input (H, W, D).
        square_size (int): Size of the cube to insert in the center.
        ball_diameter (int): Diameter of the ball kernel.
        gaussian (bool): Whether to use a Gaussian weighting in the ball.
        gaussian_std (float): Standard deviation for the Gaussian.
        output_filename (str): Path for the output NIfTI file.
    """
    # Create a 5D input tensor (B, C, H, W, D) filled with zeros
    x = torch.zeros((1, 1, *input_size), dtype=torch.float32)
    
    # Determine the center of the input
    center = [dim // 2 for dim in input_size]
    
    # Insert a cube (all ones) at the center of the input volume.
    half_square = square_size // 2
    x[0, 0,
    center[0]-half_square : center[0]+half_square+1,
    center[1]-half_square : center[1]+half_square+1,
    center[2]-half_square : center[2]+half_square+1] = 1.0

    # Print the center coordinates of the input
    print(f"Input center coordinates: {center}")
    
    # Apply the ball convolution over the input
    output = ball_convolution(x, ball_diameter, gaussian, gaussian_std)
    
    # Remove batch and channel dimensions and convert to a NumPy array
    output_np = output.squeeze().numpy()
    
    # Compute the center of mass of the output
    H, W, D = output_np.shape
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    total = np.sum(output_np)
    if total == 0:
        com = (0.0, 0.0, 0.0)
    else:
        com_x = np.sum(grid_x * output_np) / total
        com_y = np.sum(grid_y * output_np) / total
        com_z = np.sum(grid_z * output_np) / total
        com = (com_x, com_y, com_z)
    
    print(f"Center of mass of output: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})")
    
    # Create an identity affine (customize voxel sizes if needed)
    affine = np.eye(4)
    
    # Save the convolved output as a NIfTI file
    nii_img = nib.Nifti1Image(output_np, affine)
    nib.save(nii_img, output_filename)
    
    print(f"Saved ball convolution output to {output_filename}")


def generate_input_and_process_volume(input_size=(64, 64, 64), square_size=20, square_location='center',
                                        diameter=15, gaussian=False, gaussian_std=3.0, tumor_volume=100,
                                        output_input_filename='input_volume.nii.gz', output_mask_filename='tumor_mask.nii.gz'):
    """
    Generates an input volume with a cube (square in 3D) composed of random values, places it either in the
    center or in the corner of the volume, applies isolate_tumor, and saves both the input volume and the
    resulting tumor mask as NIfTI files.
    
    Args:
        input_size (tuple): The size of the 3D input volume (H, W, D).
        square_size (int): The edge-length of the cube to insert.
        square_location (str): Where to place the cube. Options: "center" or "corner".
        diameter (int): Diameter of the ball kernel for isolate_tumor.
        gaussian (bool): Whether to use Gaussian weighting in the ball convolution.
        gaussian_std (float): Standard deviation of the Gaussian.
        tumor_volume (int): The number of voxels to select as the tumor volume.
        output_input_filename (str): File path to save the input volume (as .nii.gz).
        output_mask_filename (str): File path to save the tumor mask (as .nii.gz).
    
    Returns:
        None
    """

    # Create a 5D input tensor with shape (B, C, H, W, D)
    x = torch.zeros((1, 1, *input_size), dtype=torch.float32)
    
    # Insert a cube with random values
    if square_location.lower() == 'center':
        # Compute center and half-size
        center = [dim // 2 for dim in input_size]
        half_square = square_size // 2
        
        # Calculate starting indices so that the cube is centered
        start_x = center[0] - half_square
        start_y = center[1] - half_square
        start_z = center[2] - half_square
        
        # Make sure we get exactly square_size elements along each dimension
        x[0, 0, start_x:start_x+square_size, start_y:start_y+square_size, start_z:start_z+square_size] = \
            torch.rand((square_size, square_size, square_size))+0.5
    
    elif square_location.lower() == 'corner':
        # Place the cube at the (0,0,0) corner
        x[0, 0, 0:square_size, 0:square_size, 0:square_size] = torch.rand((square_size, square_size, square_size))
    
    else:
        raise ValueError("square_location must be either 'center' or 'corner'")
    
    # Save the input volume as a NIfTI file (save the spatial part: (H, W, D))
    input_np = x[0, 0].numpy()
    affine = np.eye(4)
    input_nii = nib.Nifti1Image(input_np, affine)
    nib.save(input_nii, output_input_filename)
    print(f"Saved input volume to {output_input_filename}")
    
    # Apply isolate_tumor to the input volume
    tumor_mask = isolate_tumor(x, diameter, gaussian, gaussian_std, tumor_volume)
    
    # Save the tumor mask as a NIfTI file (convert to uint8 for a binary mask)
    tumor_mask_np = tumor_mask.numpy().astype(np.uint8)
    tumor_mask_nii = nib.Nifti1Image(tumor_mask_np, affine)
    nib.save(tumor_mask_nii, output_mask_filename)
    print(f"Saved tumor mask to {output_mask_filename}")
    
    
    
