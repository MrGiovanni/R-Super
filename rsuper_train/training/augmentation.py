import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import math
import pdb
from scipy.ndimage import binary_erosion, binary_dilation
import random

# This is a PyTorch data augmentation library, that takes PyTorch Tensor as input
# Functions can be applied in the __getitem__ function to do augmentation on the fly during training.
# These functions can be easily parallelized by setting 'num_workers' in pytorch dataloader.

# tensor_img: 1, C, (D), H, W

def gaussian_noise(tensor_img, std, mean=0):
    
    return tensor_img + torch.randn(tensor_img.shape).to(tensor_img.device) * std + mean

def generate_2d_gaussian_kernel(kernel_size, sigma):
    # Generate a meshgrid for the kernel
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    y = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    x, y = torch.meshgrid(x, y)

    # Calculate the 2D Gaussian kernel
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / (2 * math.pi * sigma ** 2)
    kernel = kernel / kernel.sum()

    return kernel.unsqueeze(0).unsqueeze(0)

def generate_3d_gaussian_kernel(kernel_size, sigma):
    # Generate a meshgrid for the kernel
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    y = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    z = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    x, y, z = torch.meshgrid(x, y, z)

    # Calculate the 3D Gaussian kernel
    kernel = torch.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / (2 * math.pi * sigma ** 2) ** 1.5
    kernel = kernel / kernel.sum()

    return kernel.unsqueeze(0).unsqueeze(0)

def gaussian_blur(tensor_img, sigma_range=[0.5, 1.0]):

    sigma = torch.rand(1) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    
    if len(tensor_img.shape) == 5:
        dim = '3d'
        kernel = generate_3d_gaussian_kernel(kernel_size, sigma).to(tensor_img.device)
        padding = [kernel_size // 2 for i in range(3)]

        return F.conv3d(tensor_img, kernel, padding=padding)
    elif len(tensor_img.shape) == 4:
        dim = '2d'
        kernel = generate_2d_gaussian_kernel(kernel_size, sigma).to(tensor_img.device)
        padding = [kernel_size // 2 for i in range(2)]

        return F.conv2d(tensor_img, kernel, padding=padding)
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')


def brightness_additive(tensor_img, std, mean=0, per_channel=False):
    
    if per_channel:
        C = tensor_img.shape[1]
    else:
        C = 1

    if len(tensor_img.shape) == 5:
        rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1, 1)).to(tensor_img.device)
    elif len(tensor_img.shape) == 4:
        rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1)).to(tensor_img.device)
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    return tensor_img + rand_brightness


def brightness_multiply(tensor_img, multiply_range=[0.7, 1.3], per_channel=False):

    if per_channel:
        C = tensor_img.shape[1]
    else:
        C = 1

    assert multiply_range[1] > multiply_range[0], 'Invalid range'

    span = multiply_range[1] - multiply_range[0]
    if len(tensor_img.shape) == 5:
        rand_brightness = torch.rand(size=(1, C, 1, 1, 1)).to(tensor_img.device) * span + multiply_range[0]
    elif len(tensor_img.shape) == 4:
        rand_brightness = torch.rand(size=(1, C, 1, 1)).to(tensor_img.device) * span + multiply_range[0]
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    return tensor_img * rand_brightness


def gamma(tensor_img, gamma_range=(0.5, 2), per_channel=False, retain_stats=True):
    
    if len(tensor_img.shape) == 5:
        dim = '3d'
        _, C, D, H, W = tensor_img.shape
    elif len(tensor_img.shape) == 4:
        dim = '2d'
        _, C, H, W = tensor_img.shape
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')
    
    tmp_C = C if per_channel else 1
    tensor_img = tensor_img.reshape(tmp_C, -1)
    minm, _ = tensor_img.min(dim=1)
    maxm, _ = tensor_img.max(dim=1)
    minm, maxm = minm.unsqueeze(1), maxm.unsqueeze(1) # unsqueeze for broadcast machanism

    rng = maxm - minm

    mean = tensor_img.mean(dim=1).unsqueeze(1)
    std = tensor_img.std(dim=1).unsqueeze(1)
    gamma = torch.rand(C, 1).to(tensor_img.device) * (gamma_range[1] - gamma_range[0]) + gamma_range[0]

    tensor_img = torch.pow((tensor_img - minm) / rng, gamma) * rng + minm

    if retain_stats:
        tensor_img -= tensor_img.mean(dim=1).unsqueeze(1)
        tensor_img = tensor_img / tensor_img.std(dim=1).unsqueeze(1) * std + mean

    if dim == '3d':
        return tensor_img.reshape(1, C, D, H, W)
    else:
        return tensor_img.reshape(1, C, H, W)
        
def contrast(tensor_img, contrast_range=(0.65, 1.5), per_channel=False, preserve_range=True):

    if len(tensor_img.shape) == 5:
        dim = '3d'
        _, C, D, H, W = tensor_img.shape
    elif len(tensor_img.shape) == 4:
        dim = '2d'
        _, C, H, W = tensor_img.shape
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')

    tmp_C = C if per_channel else 1
    tensor_img = tensor_img.reshape(tmp_C, -1)
    minm, _ = tensor_img.min(dim=1)
    maxm, _ = tensor_img.max(dim=1)
    minm, maxm = minm.unsqueeze(1), maxm.unsqueeze(1) # unsqueeze for broadcast machanism


    mean = tensor_img.mean(dim=1).unsqueeze(1)
    factor = torch.rand(C, 1).to(tensor_img.device) * (contrast_range[1] - contrast_range[0]) + contrast_range[0]

    tensor_img = (tensor_img - mean) * factor + mean

    if preserve_range:
        tensor_img = torch.clamp(tensor_img, min=minm, max=maxm)

    if dim == '3d':
        return tensor_img.reshape(1, C, D, H, W)
    else:
        return tensor_img.reshape(1, C, H, W)

def mirror(tensor_img, axis=0):

    '''
    Args:
        tensor_img: an image with format of pytorch tensor
        axis: the axis for mirroring. 0 for the first image axis, 1 for the second, 2 for the third (if volume image)
    '''


    if len(tensor_img.shape) == 5:
        dim = '3d'
        assert axis in [0, 1, 2], "axis should be either 0, 1 or 2 for volume images"

    elif len(tensor_img.shape) == 4:
        dim = '2d'
        assert axis in [0, 1], "axis should be either 0 or 1 for 2D images"
    else:
        raise ValueError('Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image')


    return torch.flip(tensor_img, dims=[2+axis])


def random_scale_rotate_translate_2d(tensor_img, tensor_lab, scale, rotate, translate):

    # implemented with affine transformation

    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale] * 2
    if isinstance(translate, float) or isinstance(translate, int):
        translate = [translate] * 2
    

    scale_x = 1 - scale[0] + np.random.random() * 2*scale[0]
    scale_y = 1 - scale[1] + np.random.random() * 2*scale[1]
    shear_x = np.random.random() * 2*scale[0] - scale[0] 
    shear_y = np.random.random() * 2*scale[1] - scale[1]
    translate_x = np.random.random() * 2*translate[0] - translate[0]
    translate_y = np.random.random() * 2*translate[1] - translate[1]

    theta_scale = torch.tensor([[scale_x, shear_x, translate_x], 
                                [shear_y, scale_y, translate_y],
                                [0, 0, 1]]).float()
    angle = (float(np.random.randint(-rotate, max(rotate, 1))) / 180.) * math.pi

    theta_rotate = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                [math.sin(angle), math.cos(angle), 0],
                                [0, 0, 1]]).float()
    
    theta = torch.mm(theta_scale, theta_rotate)[0:2, :]
    grid = F.affine_grid(theta.unsqueeze(0), tensor_img.size(), align_corners=True).to(tensor_img.device)

    tensor_img = F.grid_sample(tensor_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    tensor_lab = F.grid_sample(tensor_lab.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()

    return tensor_img, tensor_lab

def random_scale_rotate_translate_3d(tensor_img, tensor_lab, scale=0.3, rotate=45, translate=0.1, shear=0.05, foreground=None):
    '''
    The axis order of SimpleITK is x,y,z
    The axis order of numpy/tensor is z,y,x
    The arguments of all transformation should use the numpy/tensor order: [z,y,x]

    '''
    
    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale] * 3
    if isinstance(translate, float) or isinstance(translate, int):
        translate = [translate] * 3
    if isinstance(rotate, float) or isinstance(rotate, int):
        rotate = [rotate] * 3
    if isinstance(shear, float) or isinstance(shear, int):
        shear = [shear] * 3

    scale_x = np.random.uniform(low=1-scale[0], high=1/(1-scale[0]))
    scale_y = np.random.uniform(low=1-scale[1], high=1/(1-scale[1]))
    scale_z = np.random.uniform(low=1-scale[2], high=1/(1-scale[2]))

    shear_xy = np.random.uniform(-shear[0], shear[0]) # contribution of y index to x axis
    shear_xz = np.random.uniform(-shear[0], shear[0]) # contribution of z index to x axis
    shear_yx = np.random.uniform(-shear[1], shear[1]) # contribution of x index to y axis
    shear_yz = np.random.uniform(-shear[1], shear[1]) # contribution of z index to y axis
    shear_zx = np.random.uniform(-shear[2], shear[2]) # contribution of x index to z axis
    shear_zy = np.random.uniform(-shear[2], shear[2]) # contribution of y index to z axis

    translate_x = np.random.uniform(-translate[0], translate[0])
    translate_y = np.random.uniform(-translate[1], translate[1])
    translate_z = np.random.uniform(-translate[2], translate[2])


    theta_scale = torch.tensor([[scale_x, shear_xy, shear_xz, translate_x],
                                [shear_yx, scale_y, shear_yz, translate_y],
                                [shear_zx, shear_zy, scale_z, translate_z], 
                                [0, 0, 0, 1]]).float()
    angle_x = (float(np.random.randint(-rotate[0], max(rotate[0], 1))) / 180.) * math.pi 
    # rotate along x axis (x index fix, rotae in yz plane)
    angle_y = (float(np.random.randint(-rotate[1], max(rotate[1], 1))) / 180.) * math.pi
    # rotate along y axis (y index fix, rotate in xz plane)
    angle_z = (float(np.random.randint(-rotate[2], max(rotate[2], 1))) / 180.) * math.pi
    # rotate along z axis (z index fix, rotate in xy plane)
    
    theta_rotate_x = torch.tensor([[1, 0, 0, 0],
                                    [0, math.cos(angle_x), -math.sin(angle_x), 0],
                                    [0, math.sin(angle_x), math.cos(angle_x), 0],
                                    [0, 0, 0, 1]]).float()
    theta_rotate_y = torch.tensor([[math.cos(angle_y), 0, -math.sin(angle_y), 0],
                                    [0, 1, 0, 0],
                                    [math.sin(angle_y), 0, math.cos(angle_y), 0],
                                    [0, 0, 0, 1]]).float()
    theta_rotate_z = torch.tensor([[math.cos(angle_z), -math.sin(angle_z), 0, 0],
                                    [math.sin(angle_z), math.cos(angle_z), 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]).float()

    theta = torch.mm(theta_rotate_x, theta_rotate_y)
    theta = torch.mm(theta, theta_rotate_z)
    
    theta = torch.mm(theta, theta_scale)[0:3, :].unsqueeze(0)
    assert len(tensor_img.size())==5
    grid = F.affine_grid(theta, tensor_img.size(), align_corners=True).to(tensor_img.device)
    tensor_img = F.grid_sample(tensor_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    tensor_lab = F.grid_sample(tensor_lab.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True).long()
    
    # --- random_scale_rotate_translate_3d ---
    if foreground is not None:
        add_chan = False
        if foreground.ndim == 3:            # (D,H,W)
            foreground = foreground.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
            add_chan = 2
        elif foreground.ndim == 4:
            foreground = foreground.unsqueeze(0)  # (1,1,D,H,W)
            add_chan = 1
        elif foreground.ndim == 5:
            add_chan = 0
        else:
            raise ValueError('Invalid dimension of foreground mask')
            
        foreground = F.grid_sample(
            foreground.float(), grid,
            mode='nearest', padding_mode='zeros',
            align_corners=True)
        if add_chan==2:
            foreground = foreground[0,0]    # back to (D,H,W)
        if add_chan==1:
            foreground = foreground[0]
        foreground = foreground.bool()
        return tensor_img, tensor_lab, foreground
    else:
        return tensor_img, tensor_lab
    



def invert_random_scale_rotate_translate_3d(
        tensor_img,
        tensor_lab,
        theta_fwd,               # (3,4)   or (1,3,4)      –– the matrix returned by the forward call
        foreground=None,
        mode_img='bilinear',     # keep same modes/padding as forward pass
        padding_mode='zeros',
        align_corners=True):
    """
    Reverse the affine transform applied by `random_scale_rotate_translate_3d`.

    Parameters
    ----------
    tensor_img : (B,C,D,H,W) torch.Tensor
        Image volume **after** augmentation.
    tensor_lab : (B,C,D,H,W) torch.Tensor
        Label/mask volume **after** augmentation.
    theta_fwd  : torch.Tensor
        The 3×4 (or 1×3×4) affine matrix used in the forward augmentation.
        Capture it during the forward call and pass it here.
    foreground : torch.Tensor or None
        Optional foreground mask in any of the shapes accepted by the forward function.
    mode_img, padding_mode, align_corners
        Passed straight to `F.grid_sample`; keep the same settings as in the forward pass.

    Returns
    -------
    tensor_img_inv, tensor_lab_inv
        Volumes mapped back to their pre-augmentation geometry.
    foreground_inv  (only if `foreground` was given)
    """
    raise ValueError('Careful, not verified yet!')
    
    # ---- 1. prepare the forward matrix ----------------------------------------------------------
    if theta_fwd.ndim == 3:      # (1,3,4)  from DataLoader batching
        theta_fwd = theta_fwd[0]
    if theta_fwd.ndim != 2 or theta_fwd.shape != (3, 4):
        raise ValueError("theta_fwd must have shape (3,4) or (1,3,4)")

    device = theta_fwd.device
    dtype  = theta_fwd.dtype

    # ---- 2. build 4×4 homogeneous matrix, invert it ---------------------------------------------
    theta4 = torch.eye(4, dtype=dtype, device=device)
    theta4[:3, :] = theta_fwd                # copy the 3×4 block

    theta_inv4 = torch.inverse(theta4)       # full affine inverse
    theta_inv  = theta_inv4[:3, :]           # back to 3×4
    theta_inv  = theta_inv.unsqueeze(0)      # (1,3,4) for affine_grid

    # ---- 3. grid + resample ---------------------------------------------------------------------
    grid_back = F.affine_grid(
        theta_inv,
        tensor_img.size(),                   # same shape as (transformed) input
        align_corners=align_corners
    )

    img_inv  = F.grid_sample(
        tensor_img, grid_back,
        mode=mode_img,
        padding_mode=padding_mode,
        align_corners=align_corners
    )

    lab_inv  = F.grid_sample(
        tensor_lab.float(), grid_back,
        mode='nearest',                      # keep labels discrete
        padding_mode=padding_mode,
        align_corners=align_corners
    ).long()

    # ---- 4. optional foreground mask ------------------------------------------------------------
    if foreground is not None:
        add_chan = 0
        if foreground.ndim == 3:            # (D,H,W)
            foreground = foreground.unsqueeze(0).unsqueeze(0)
            add_chan = 2
        elif foreground.ndim == 4:          # (C,D,H,W)
            foreground = foreground.unsqueeze(0)
            add_chan = 1
        elif foreground.ndim != 5:          # anything else is invalid
            raise ValueError('Invalid dimension of foreground mask')

        foreground_inv = F.grid_sample(
            foreground.float(), grid_back,
            mode='nearest',
            padding_mode=padding_mode,
            align_corners=align_corners
        )

        if add_chan == 2:
            foreground_inv = foreground_inv[0, 0]
        if add_chan == 1:
            foreground_inv = foreground_inv[0]
        return img_inv, lab_inv, foreground_inv.bool()

    return img_inv, lab_inv
  

def crop_2d(tensor_img, tensor_lab, crop_size, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 2

    _, _, H, W = tensor_img.shape

    diff_H = H - crop_size[0]
    diff_W = W - crop_size[1]
    
    if mode == 'random':
        rand_x = np.random.randint(0, max(diff_H, 1))
        rand_y = np.random.randint(0, max(diff_W, 1))
    else:
        rand_x = diff_H // 2
        rand_y = diff_W // 2

    cropped_img = tensor_img[:, :, rand_x:rand_x+crop_size[0], rand_y:rand_y+crop_size[1]]
    cropped_lab = tensor_lab[:, :, rand_x:rand_x+crop_size[0], rand_y:rand_y+crop_size[1]]

    return cropped_img.contiguous(), cropped_lab.contiguous()


def crop_3d(tensor_img, tensor_lab, crop_size, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    _, _, D, H, W = tensor_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    if mode == 'random':
        rand_z = np.random.randint(0, max(diff_D, 1))
        rand_y = np.random.randint(0, max(diff_H, 1))
        rand_x = np.random.randint(0, max(diff_W, 1))
    else:
        rand_z = diff_D // 2
        rand_y = diff_H // 2
        rand_x = diff_W // 2

    cropped_img = tensor_img[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = tensor_lab[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]

    return cropped_img.contiguous(), cropped_lab.contiguous()

def np_crop_3d(np_img, np_lab, crop_size, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3 

    D, H, W = np_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    if mode == 'random':
        rand_z = np.random.randint(0, max(diff_D, 1)) 
        rand_y = np.random.randint(0, max(diff_H, 1)) 
        rand_x = np.random.randint(0, max(diff_W, 1)) 
    else:
        rand_z = diff_D // 2
        rand_y = diff_H // 2
        rand_x = diff_W // 2

    cropped_img = np_img[rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = np_lab[:,rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]

    return np.ascontiguousarray(cropped_img), np.ascontiguousarray(cropped_lab)



def crop_around_coordinate_3d(tensor_img, tensor_lab, crop_size, coordinate, mode, foreground=None):
    assert mode in ['random', 'center', 'small_rnd_shift'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    z, y, x = coordinate

    _, _, D, H, W = tensor_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]
    
    
    if mode == 'random':
        min_z = max(0, z-crop_size[0])
        max_z = min(diff_D, z+crop_size[0])
        min_y = max(0, y-crop_size[1])
        max_y = min(diff_H, y+crop_size[1])
        min_x = max(0, x-crop_size[2])
        max_x = min(diff_W, x+crop_size[2])
        
        rand_z = np.random.randint(min_z, max_z)
        rand_y = np.random.randint(min_y, max_y)
        rand_x = np.random.randint(min_x, max_x)
    elif mode == 'small_rnd_shift':
        # Calculate centered crop start indices
        center_z = z - crop_size[0] // 2
        center_y = y - crop_size[1] // 2
        center_x = x - crop_size[2] // 2

        # Maximum random shift (50% of crop size)
        max_shift_z = int(crop_size[0] * 0.5)
        max_shift_y = int(crop_size[1] * 0.5)
        max_shift_x = int(crop_size[2] * 0.5)

        # Generate random offsets in the range [-max_shift, max_shift]
        offset_z = np.random.randint(-max_shift_z, max_shift_z + 1)
        offset_y = np.random.randint(-max_shift_y, max_shift_y + 1)
        offset_x = np.random.randint(-max_shift_x, max_shift_x + 1)

        # Apply offsets and clip to image boundaries
        rand_z = np.clip(center_z + offset_z, 0, D - crop_size[0])
        rand_y = np.clip(center_y + offset_y, 0, H - crop_size[1])
        rand_x = np.clip(center_x + offset_x, 0, W - crop_size[2])
    else:
        min_z = max(0, z - math.ceil(crop_size[0] / 2))
        rand_z = min(min_z, D - crop_size[0])
        min_y = max(0, y - math.ceil(crop_size[1] / 2))
        rand_y = min(min_y, H - crop_size[1])
        min_x = max(0, x - math.ceil(crop_size[2] / 2))
        rand_x = min(min_x, W - crop_size[2])

    cropped_img = tensor_img[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = tensor_lab[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    if foreground is not None:
        while len(foreground.shape) < len(tensor_img.shape):
            foreground = foreground.unsqueeze(0)
        cropped_foreground = foreground[:, :, rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
        return (cropped_img.contiguous(), cropped_lab.contiguous(), cropped_foreground.contiguous())
    else:
        return cropped_img.contiguous(), cropped_lab.contiguous()


def np_crop_around_coordinate_3d(np_img, np_lab, crop_size, coordinate, mode):
    assert mode in ['random', 'center'], "Invalid Mode, should be \'random\' or \'center\'"
    if isinstance(crop_size, int):
        crop_size = [crop_size] * 3

    z, y, x = coordinate

    D, H, W = np_img.shape

    diff_D = D - crop_size[0]
    diff_H = H - crop_size[1]
    diff_W = W - crop_size[2]


    if mode == 'random':
        min_z = max(0, z-crop_size[0])
        max_z = min(diff_D, z+crop_size[0])
        min_y = max(0, y-crop_size[1])
        max_y = min(diff_H, y+crop_size[1])
        min_x = max(0, x-crop_size[2])
        max_x = min(diff_W, x+crop_size[2])

        rand_z = np.random.randint(min_z, max_z)
        rand_y = np.random.randint(min_y, max_y)
        rand_x = np.random.randint(min_x, max_x)
    else:
        min_z = max(0, z - math.ceil(crop_size[0] / 2))
        rand_z = min(min_z, D - crop_size[0])
        min_y = max(0, y - math.ceil(crop_size[1] / 2))
        rand_y = min(min_y, H - crop_size[1])
        min_x = max(0, x - math.ceil(crop_size[2] / 2))
        rand_x = min(min_x, W - crop_size[2])

    cropped_img = np_img[rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]
    cropped_lab = np_lab[rand_z:rand_z+crop_size[0], rand_y:rand_y+crop_size[1], rand_x:rand_x+crop_size[2]]

    return np.ascontiguousarray(cropped_img), np.ascontiguousarray(cropped_lab)

def random_crop_on_tumor(tensor_img, tensor_lab, lesion_classes, d, h, w, tumor_case,
                         tumor_prob=None, foreground_prob=None, background_prob=None,
                         return_crop_organ=False, class_names=None,
                         foreground_classes=None):
    """
    Perform a cropping operation on a tensor, based on tumor presence or other criteria.

    Parameters:
        tensor_img (torch.Tensor): The input image tensor.
        tensor_lab (torch.Tensor): The label tensor.
        lesion_classes (list or torch.Tensor): Indices of lesion classes in the label tensor.
        d, h, w (int): Dimensions for cropping.

    Returns:
        tensor_img (torch.Tensor): The cropped image tensor.
        tensor_lab (torch.Tensor): The cropped label tensor.

    """
    rnd = np.random.random()
    if (tumor_prob is None) or (foreground_prob is None) or (background_prob is None):
        #standard probs
        if tumor_case:
            tumor_prob=0.9
            foreground_prob=0.05
            background_prob=0.05
            print('Tumor case')
        else:
            tumor_prob=0
            foreground_prob=0.9
            background_prob=0.1
            print('Non-tumor case')

    if rnd < tumor_prob:
        print('Attempting tumor crop')
        t = tumor_crop(tensor_img, tensor_lab, lesion_classes, d, h, w, return_crop_organ=return_crop_organ)
        if return_crop_organ:
            #print(f'Tumor crop successful, organ: {crop_organ}')
            tensor_img, tensor_lab, crop_organ = t
        else:
            #print(f'Tumor crop successful, organ: {crop_organ}')
            tensor_img, tensor_lab = t
    elif rnd < (tumor_prob + background_prob):
        print('Background crop')
        tensor_img, tensor_lab = negative_crop(tensor_img, tensor_lab, lesion_classes, d, h, w)
        crop_organ = 'random'
    else:
        print('Attempting organ crop')
        t = organ_crop(tensor_img, tensor_lab, lesion_classes, d, h, w, return_crop_organ=return_crop_organ,
                       foreground_classes=foreground_classes)
        if return_crop_organ:
            tensor_img, tensor_lab, crop_organ = t
        else:
            tensor_img, tensor_lab = t
    if return_crop_organ:
        if (crop_organ is not None) and crop_organ != 'random':
            crop_organ = class_names[crop_organ]
        print(f'Organ crop successful, organ: {crop_organ}')
        return tensor_img, tensor_lab, crop_organ
    else:
        #print(f'Organ crop successful, organ: {crop_organ}')
        return tensor_img, tensor_lab

def negative_crop (tensor_img, tensor_lab, lesion_classes, d, h, w):
    # Negative crop
    foreground_mask = tensor_lab[0].sum(0, keepdim=False)
    back_voxels = torch.nonzero(foreground_mask == 0, as_tuple=False)
    if len(back_voxels) == 0:
        # Fallback to random crop
        tensor_img, tensor_lab = crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
    else:
        # Crop around a random background voxel
        center = back_voxels[torch.randint(0, len(back_voxels), (1,))][0]
        tensor_img, tensor_lab = crop_around_coordinate_3d(tensor_img, tensor_lab, [d, h, w], coordinate=center, mode='small_rnd_shift')
    return tensor_img, tensor_lab

def organ_crop(tensor_img, tensor_lab, lesion_classes, d, h, w,
               return_crop_organ=False,foreground_classes=None):
    # Random label crop
    foreground_mask = tensor_lab[0]#.sum(0, keepdim=False)
    foreg = []
    clss = []
    print('Foreground classes:', foreground_classes)
    for c in range(tensor_lab.shape[1]): # Loop through each class in the label tensor
        if c not in lesion_classes:
            if foreground_classes is not None:
                if c not in foreground_classes:
                    continue
            # Only consider non lesion classes for foreground
            fore = tensor_lab[0][c]
            if fore.sum() > 0:
                foreg.append(fore)
                clss.append(c)
    if len(foreg) !=0:
        #randomly choose one of the foreground classes to crop around
        idx = torch.randint(0, len(foreg), (1,)).item() # randomly choose one of the foreground classes
        foreground_mask = foreg[idx] # choose the randomly selected foreground mask
        crop_organ = clss[idx]
        print('Cropping on organ (not tumor):', crop_organ)
        foreground_voxels = torch.nonzero(foreground_mask)
    else:
        print('Foreground voxels empty, falling back to random crop')
        foreground_voxels = []
        crop_organ = 'random'
    if len(foreground_voxels) == 0:
        # Fallback to random crop
        tensor_img, tensor_lab = crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
        crop_organ = 'random'
    else:
        # Crop around a random foreground voxel
        center = foreground_voxels[torch.randint(0, len(foreground_voxels), (1,))][0]
        tensor_img, tensor_lab = crop_around_coordinate_3d(tensor_img, tensor_lab, [d, h, w], coordinate=center, mode='small_rnd_shift')
    if return_crop_organ:
        return tensor_img, tensor_lab, crop_organ
    else:
        return tensor_img, tensor_lab

def tumor_crop(tensor_img, tensor_lab, lesion_classes, d, h, w, return_crop_organ=False):
    # Tumor crop
    #print('Applying Tumor crop')
    tumor_mask = tensor_lab[0][lesion_classes]#.sum(1, keepdim=False)---do not sum, this will favor larger tumors. Choose the tumor class randomly across the classes with lesion in the CT
    assert tensor_lab.shape[0] == 1
    if tumor_mask.sum() == 0:
        #print('Fell back to random crop')
        # Fallback to random crop
        tensor_img, tensor_lab = crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
        crop_organ = 'random'
    else:
        # Crop around a random tumor voxel
        #which classes have tumor? sum last 3 dimensions
        positives = tumor_mask.sum(dim=(-3,-2,-1))>0
        possibilities = [i for i in range(positives.shape[0]) if positives[i]]
        #choose one of the possibilities
        chosen_class = possibilities[ torch.randint(0, len(possibilities), (1,)).item() ]
        #print('Chosen tumor class:',chosen_class)
        tumor_mask = tumor_mask[chosen_class]
        crop_organ = lesion_classes[chosen_class]
        tumor_voxels = torch.nonzero(tumor_mask)
        center = tumor_voxels[torch.randint(0, len(tumor_voxels), (1,))][0]
        tensor_img, tensor_lab = crop_around_coordinate_3d(tensor_img, tensor_lab, [d, h, w], coordinate=center, mode='small_rnd_shift')
    if return_crop_organ:
        return tensor_img, tensor_lab, crop_organ
    else:
        return tensor_img, tensor_lab
        
from scipy.ndimage import binary_erosion, binary_dilation, label

def denoise_mask(mask_3d, iterations=2, connected_component=True):
    """
    Perform `iterations` binary erosions + `iterations` binary dilations,
    then AND with the original mask to remove small/noisy regions.
    Then keep only the largest connected component of the result.
    """
    device = mask_3d.device
    #check if mask is torch tensor
    if isinstance(mask_3d, torch.Tensor):
        np_mask = mask_3d.cpu().numpy().astype(bool)
    else:
        np_mask = mask_3d.astype(bool)

    # 1) Morphological denoise
    eroded  = binary_erosion(np_mask, iterations=iterations)
    dilated = binary_dilation(eroded,  iterations=iterations)
    final   = dilated & np_mask  # shape: (D,H,W), bool

    if connected_component:
        # 2) Label connected components in `final`
        labeled, num_components = label(final)  # labeled: int array with [1..num_components] labels

        if num_components == 0:
            # No foreground at all
            refined_mask = torch.from_numpy(final).to(device)
        elif num_components == 1:
            # Only one component, so it's already the largest
            refined_mask = torch.from_numpy(final).to(device)
        else:
            # More than one => pick largest
            # counts[i] = number of voxels with label i
            counts = np.bincount(labeled.ravel())
            # Index 0 is background, so ignore it by zeroing it out.
            counts[0] = 0  
            largest_label = np.argmax(counts)     # The label with the most voxels
            largest_mask = (labeled == largest_label)
            refined_mask = torch.from_numpy(largest_mask).to(device)
    else:
        # No connected component analysis, just return the mask
        refined_mask = torch.from_numpy(final).to(device)

    return refined_mask


def crop_foreground_3d(tensor_ct, tensor_lab, foreground, crop_size, margin=1, refine_iterations=3, rand=True):
    """
    Crops a 3D CT & binary label around the label's nonzero region, returning EXACT [d,h,w].
    
    If rand=True, the bounding box is randomly shifted within the volume if possible.
    If rand=False, it is centered if possible.

    1) If label is empty => return "zero mask"
    2) If bounding box is bigger than crop_size => morphological denoise => 
       if still doesn't fit => return "mask does not fit crop size"
    3) If bounding box <= crop_size => compute the valid range of random shifts
       for each dimension. If no valid shift is possible => "mask does not fit crop size"
    4) Otherwise, pick a random shift and return (cropped_ct, cropped_label).

    Args:
        tensor_ct (torch.Tensor): shape [D,H,W] or [1,D,H,W]
        foreground (torch.Tensor): shape [D,H,W] or [1,D,H,W], binary
        
        crop_size (tuple/list): (d,h,w)
        margin (int or tuple): extra margin
        refine_iterations (int): # of erosions/dilations

    Returns:
        (cropped_ct, cropped_label) or
        "zero mask" or
        "mask does not fit crop size"
    """

    ##### 1) Unify shapes #####
    if tensor_ct.ndim == 3:
        D, H, W = tensor_ct.shape
        ct_has_channel = False
        ct_has_batch = False
    elif tensor_ct.ndim == 4 and tensor_ct.shape[0] == 1:
        _, D, H, W = tensor_ct.shape
        ct_has_channel = True
        ct_has_batch = False
    elif tensor_ct.ndim == 5 and tensor_ct.shape[0] == 1 and tensor_ct.shape[1] == 1:
        _, _, D, H, W = tensor_ct.shape
        ct_has_channel = True
        ct_has_batch = True
    else:
        raise ValueError(f"CT must be [D,H,W] or [1,D,H,W] or [1,1,D,H,W], got {tensor_ct.shape}")

    # --- replace the old squeeze block ----------------------------------------
    if foreground.ndim == 4 and foreground.shape[0] == 1:
        # foreground is [1, D, H, W]  → drop the leading 1
        label_3d = foreground[0].clone()
    elif foreground.ndim == 3:
        label_3d = foreground.clone()
    else:
        raise ValueError(
            f"Foreground must be [D,H,W] or [1,D,H,W], got {foreground.shape}"
        )
    
    backup_foreground = foreground.clone()
        
    # Check empty
    if torch.count_nonzero(label_3d) == 0:
        return "zero mask"

    ##### 2) Get bounding box #####
    coords = torch.nonzero(label_3d, as_tuple=False)
    zmin, zmax = coords[:, 0].min().item(), coords[:, 0].max().item()
    ymin, ymax = coords[:, 1].min().item(), coords[:, 1].max().item()
    xmin, xmax = coords[:, 2].min().item(), coords[:, 2].max().item()

    if isinstance(margin, int):
        margin = (margin, margin, margin)
    mz, my, mx = margin

    # Apply margin---this is the foreground bounding box
    zmin = max(zmin - mz, 0)
    zmax = min(zmax + mz, D - 1)
    ymin = max(ymin - my, 0)
    ymax = min(ymax + my, H - 1)
    xmin = max(xmin - mx, 0)
    xmax = min(xmax + mx, W - 1)
    
    # After applying margin and clamping:
    if xmin > xmax:   xmin, xmax = xmax, xmin
    if ymin > ymax:   ymin, ymax = ymax, ymin
    if zmin > zmax:   zmin, zmax = zmax, zmin

    desired_d, desired_h, desired_w = crop_size

    def bbox_dim(z0, z1, y0, y1, x0, x1):
        return (z1 - z0 + 1), (y1 - y0 + 1), (x1 - x0 + 1)

    bbox_d, bbox_h, bbox_w = bbox_dim(zmin, zmax, ymin, ymax, xmin, xmax)

    # Check if bounding box is bigger
    if bbox_d > desired_d or bbox_h > desired_h or bbox_w > desired_w:
        # Attempt morphological denoise
        refined = denoise_mask(label_3d, iterations=refine_iterations)
        label_3d = refined.clone()
        if torch.count_nonzero(refined) == 0:
            return "zero mask"

        # Recompute bounding box
        coords = torch.nonzero(refined, as_tuple=False)
        zmin, zmax = coords[:, 0].min().item(), coords[:, 0].max().item()
        ymin, ymax = coords[:, 1].min().item(), coords[:, 1].max().item()
        xmin, xmax = coords[:, 2].min().item(), coords[:, 2].max().item()

        zmin = max(zmin - mz, 0)
        zmax = min(zmax + mz, D - 1)
        ymin = max(ymin - my, 0)
        ymax = min(ymax + my, H - 1)
        xmin = max(xmin - mx, 0)
        xmax = min(xmax + mx, W - 1)
        
        if xmin > xmax:   xmin, xmax = xmax, xmin
        if ymin > ymax:   ymin, ymax = ymax, ymin
        if zmin > zmax:   zmin, zmax = zmax, zmin

        bbox_d, bbox_h, bbox_w = bbox_dim(zmin, zmax, ymin, ymax, xmin, xmax)
        if bbox_d > desired_d or bbox_h > desired_h or bbox_w > desired_w:
            return "mask does not fit crop size"

    ##### 3) We know bounding box is <= crop_size. Let's find valid shifts. #####

    # We want subvolume [zstart : zstart+desired_d-1] to fully contain [zmin : zmax].
    # => zstart <= zmin
    # => zstart+desired_d-1 >= zmax => zstart >= zmax - (desired_d-1)
    # So zstart in [ zmax-(desired_d-1), zmin ]
    # Also zstart cannot be negative, and zstart+desired_d-1 cannot extend beyound the volume.
    # We'll define a helper:

    def valid_shifts_1D(min_bb, max_bb, vol_size, crop_size):
        """
        Returns a range (low, high) of all valid starting positions 
        such that [start : start+crop_size-1] fully contains [min_bb : max_bb]
        and stays within [0, vol_size-1].
        If there's no valid integer in [low, high], no shift is possible.
        """
        min_start = max_bb - (crop_size - 1)  # bounding box forced at the 'end'
        max_start = min_bb                    # bounding box forced at the 'start'

        # clamp to [0, vol_size - crop_size]
        lower_bound = 0
        upper_bound = vol_size - crop_size

        # intersection
        final_low = max(min_start, lower_bound)
        final_high = min(max_start, upper_bound)
        return int(final_low), int(final_high)

    # z dimension
    z_low, z_high = valid_shifts_1D(zmin, zmax, D, desired_d)
    # y dimension
    y_low, y_high = valid_shifts_1D(ymin, ymax, H, desired_h)
    # x dimension
    x_low, x_high = valid_shifts_1D(xmin, xmax, W, desired_w)

    # If any dimension has final_low > final_high, 
    # there's no integer that can satisfy bounding box constraints.
    if z_low > z_high or y_low > y_high or x_low > x_high:
        return "mask does not fit crop size"

    # Helper to pick shift in one dimension
    # If there's no valid shift (low>high), we 'crop in place' by placing bounding box at zmin
    # (clamped so we stay inside [0, vol_size - crop_size]).
    def pick_shift_1d(low, high, bb_min, vol_size, csize, rand_flag):
        if low > high:
            # No shift range => just place bounding box at bb_min (clamp to valid range)
            return max(0, min(bb_min, vol_size - csize))
        else:
            if rand_flag:
                return random.randint(int(low), int(high))
            else:
                return (low + high) // 2

    ##### 4) Pick the shift (or no shift if none is possible) #####
    z_start = pick_shift_1d(z_low, z_high, zmin, D, desired_d, rand)
    y_start = pick_shift_1d(y_low, y_high, ymin, H, desired_h, rand)
    x_start = pick_shift_1d(x_low, x_high, xmin, W, desired_w, rand)

    z_end = z_start + desired_d
    y_end = y_start + desired_h
    x_end = x_start + desired_w
    
    def dbg(dim, low, high, start, bb_min, bb_max, size):
        print(f"{dim}:  bb=({bb_min},{bb_max})  "
            f"shift_range=[{low},{high}]  chosen={start}  "
            f"crop=({start},{start+size-1})")
    #dbg('z', z_low, z_high, z_start, zmin, zmax, desired_d)
    #dbg('y', y_low, y_high, y_start, ymin, ymax, desired_h)
    #dbg('x', x_low, x_high, x_start, xmin, xmax, desired_w)

    # Now we check if indeed we are inside the volume
    if z_end > D or y_end > H or x_end > W:
        raise ValueError(f"Crop failed. Why? It should not fail here.")

    ##### 5) Final Crop #####
    if ct_has_channel and not ct_has_batch:
        cropped_ct = tensor_ct[:, z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_label = tensor_lab[:, z_start:z_end, y_start:y_end, x_start:x_end]
    elif ct_has_channel and ct_has_batch:
        cropped_ct = tensor_ct[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_label = tensor_lab[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
    else:
        cropped_ct = tensor_ct[z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_label = tensor_lab[z_start:z_end, y_start:y_end, x_start:x_end]

    if cropped_ct.shape[-3:] != (desired_d, desired_h, desired_w):
        raise ValueError(f"Crop failed, got {cropped_ct.shape[-3:]}. Why? It should not fail here.")
    
    cropped_fg = label_3d[z_start:z_end, y_start:y_end, x_start:x_end]
    if torch.count_nonzero(cropped_fg) == 0:
        #is the original foreground 0?
        print('Original foreground total:', torch.count_nonzero(label_3d))
        print('Cropped foreground total:',torch.count_nonzero(cropped_fg))
        #check for inplace changes at foreground
        print('Inplace changes in foreground:',(not torch.equal(foreground,backup_foreground)))
        #is the problem in random??
        z_start = pick_shift_1d(z_low, z_high, zmin, D, desired_d, False)
        y_start = pick_shift_1d(y_low, y_high, ymin, H, desired_h, False)
        x_start = pick_shift_1d(x_low, x_high, xmin, W, desired_w, False)

        z_end = z_start + desired_d
        y_end = y_start + desired_h
        x_end = x_start + desired_w
        
        cropped_fg_deter = label_3d[z_start:z_end, y_start:y_end, x_start:x_end]
        
        print('Deter foreground total:',torch.count_nonzero(cropped_fg_deter))
        raise ValueError("zero mask after crop")

    return (cropped_ct, cropped_label, cropped_fg)



def pad_volume_pair(input_tensor: torch.Tensor, label_tensor: torch.Tensor, 
                    desired_d: int, desired_h: int, desired_w: int):
    """
    Pads both the input_tensor and label_tensor along their last three dimensions 
    (assumed to be depth, height, width) with zeros on both sides if any of the dimensions 
    is smaller than the specified desired size. The same padding is applied to both tensors.
    
    If no padding is needed (i.e. all spatial dimensions are at least the desired sizes), 
    the original tensors are returned.
    
    Args:
        input_tensor (torch.Tensor): Tensor of shape (..., D, H, W).
        label_tensor (torch.Tensor): Tensor of shape (..., D, H, W). Must have the same shape
                                     in the last three dimensions as input_tensor.
        desired_d (int): Desired depth.
        desired_h (int): Desired height.
        desired_w (int): Desired width.
    
    Returns:
        tuple: (padded_input, padded_label), both with spatial dimensions at least 
               (desired_d, desired_h, desired_w).
    """
    # Check that the spatial dimensions of the two tensors match.
    if input_tensor.shape[-3:] != label_tensor.shape[-3:]:
        raise ValueError("The input and label tensors must have the same spatial dimensions.")
    
    current_d, current_h, current_w = input_tensor.shape[-3:]
    
    # Compute total padding required for each spatial dimension.
    pad_d = max(0, desired_d - current_d)
    pad_h = max(0, desired_h - current_h)
    pad_w = max(0, desired_w - current_w)
    
    # If no padding is needed, return the original tensors.
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return input_tensor, label_tensor
    
    # Compute symmetric padding amounts.
    pad_d_left = pad_d // 2
    pad_d_right = pad_d - pad_d_left
    pad_h_left = pad_h // 2
    pad_h_right = pad_h - pad_h_left
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    
    # F.pad expects a tuple with padding for the last dimension first.
    # For 3 dimensions, the tuple is: (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)
    padding = (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)
    
    padded_input = F.pad(input_tensor, padding, mode="constant", value=0)
    padded_label = F.pad(label_tensor, padding, mode="constant", value=0)
    
    return padded_input, padded_label