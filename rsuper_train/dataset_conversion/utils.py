import numpy as np
import SimpleITK as sitk
from skimage.measure import regionprops
import pdb
import torch
from typing import Union, Tuple, List

import os

def ResampleXYZAxis(imImage, space=(1., 1., 1.), interp=sitk.sitkLinear):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)
    sp1 = imImage.GetSpacing()
    sz1 = imImage.GetSize()

    sz2 = (int(round(sz1[0]*sp1[0]*1.0/space[0])), int(round(sz1[1]*sp1[1]*1.0/space[1])), int(round(sz1[2]*sp1[2]*1.0/space[2])))

    imRefImage = sitk.Image(sz2, imImage.GetPixelIDValue())
    imRefImage.SetSpacing(space)
    imRefImage.SetOrigin(imImage.GetOrigin())
    imRefImage.SetDirection(imImage.GetDirection())

    imOutImage = sitk.Resample(imImage, imRefImage, identity1, interp)

    return imOutImage

def ResampleLabelToRef(imLabel, imRef, interp=sitk.sitkNearestNeighbor):
    identity1 = sitk.Transform(3, sitk.sitkIdentity)

    imRefImage = sitk.Image(imRef.GetSize(), imLabel.GetPixelIDValue())
    imRefImage.SetSpacing(imRef.GetSpacing())
    imRefImage.SetOrigin(imRef.GetOrigin())
    imRefImage.SetDirection(imRef.GetDirection())
        
    ResampledLabel = sitk.Resample(imLabel, imRefImage, identity1, interp)
    
    return ResampledLabel

def reorient_image(image, desired_orientation='RAI'):
    
    current_orientation = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(image.GetDirection())

    if current_orientation != desired_orientation:
        reorient_filter = sitk.DICOMOrientImageFilter()
        reorient_filter.SetDesiredCoordinateOrientation(desired_orientation)
        image = reorient_filter.Execute(image)

    return image

def ITKReDirection(itkimg, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):
    # target direction should be orthognal, i.e. (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # permute axis
    tmp_target_direction = np.abs(np.round(np.array(target_direction))).reshape(3,3).T
    current_direction = np.abs(np.round(itkimg.GetDirection())).reshape(3,3).T
    
    permute_order = []
    if not np.array_equal(tmp_target_direction, current_direction):
        for i in range(3):
            for j in range(3):
                if np.array_equal(tmp_target_direction[i], current_direction[j]):
                    permute_order.append(j)
                    #print(i, j)
                    #print(permute_order)
                    break
        redirect_img = sitk.PermuteAxes(itkimg, permute_order)
    else:
        redirect_img = itkimg
    # flip axis
    current_direction = np.round(np.array(redirect_img.GetDirection())).reshape(3,3).T
    current_direction = np.max(current_direction, axis=1)

    tmp_target_direction = np.array(target_direction).reshape(3,3).T 
    tmp_target_direction = np.max(tmp_target_direction, axis=1)
    flip_order = ((tmp_target_direction * current_direction) != 1)
    fliped_img = sitk.Flip(redirect_img, [bool(flip_order[0]), bool(flip_order[1]), bool(flip_order[2])])
    return fliped_img



def CropForeground(imImage, imLabel, context_size=[10, 30, 30]):
    # The context_size is in numpy index order: z, y, x
    # Note that SimpleITK uses the index order of: x, y, z

    npImg = sitk.GetArrayFromImage(imImage)
    npLab = []

    # Convert labels to NumPy arrays and create the combined mask
    for key in sorted(imLabel.keys()):  # Sort keys for consistent processing
        item = sitk.GetArrayFromImage(imLabel[key])
        npLab.append(item)
    npLab = np.stack(npLab, axis=0)

    mask = (npLab.sum(0) > 0).astype(np.uint8)  # Foreground mask
    
    regions = regionprops(mask)
    assert len(regions) == 1

    zz, yy, xx = npImg.shape

    z, y, x = regions[0].centroid

    z_min, y_min, x_min, z_max, y_max, x_max = regions[0].bbox
    print('forground size:', z_max-z_min, y_max-y_min, x_max-x_min)

    z, y, x = int(z), int(y), int(x)

    z_min = max(0, z_min-context_size[0])
    z_max = min(zz, z_max+context_size[0])
    y_min = max(0, y_min-context_size[2])
    y_max = min(yy, y_max+context_size[2])
    x_min = max(0, x_min-context_size[1])
    x_max = min(xx, x_max+context_size[1])

    img = npImg[z_min:z_max, y_min:y_max, x_min:x_max]

    # Crop the labels
    lab = {}
    for key in sorted(imLabel.keys()):  # Ensure consistent order
        lab[key] = npLab[sorted(imLabel.keys()).index(key)][z_min:z_max, y_min:y_max, x_min:x_max]

    # Convert cropped NumPy arrays back to SimpleITK images
    croppedImage = sitk.GetImageFromArray(img)
    croppedImage.SetSpacing(imImage.GetSpacing())
    croppedImage.SetDirection(imImage.GetDirection())

    croppedLabel = {}
    for key in lab.keys():
        item = sitk.GetImageFromArray(lab[key])
        item.SetSpacing(imLabel[key].GetSpacing())  # Preserve label-specific spacing
        item.SetDirection(imLabel[key].GetDirection())  # Preserve label-specific direction
        croppedLabel[key] = item

    return croppedImage, croppedLabel

def resample_torch_simple(
        data: Union[torch.Tensor, np.ndarray],
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        is_seg: bool = False,
        num_threads: int = 4,
        device: torch.device = torch.device('cpu'),
        memefficient_seg_resampling: bool = False,
        mode='linear'
):
    torch_mode=mode

    if isinstance(new_shape, np.ndarray):
        new_shape = [int(i) for i in new_shape]

    if all([i == j for i, j in zip(new_shape, data.shape[1:])]):
        return data
    else:
        n_threads = torch.get_num_threads()
        torch.set_num_threads(num_threads)
        new_shape = tuple(new_shape)
        with torch.no_grad():

            input_was_numpy = isinstance(data, np.ndarray)
            if input_was_numpy:
                data = torch.from_numpy(data).to(device)
            else:
                orig_device = deepcopy(data.device)
                data = data.to(device)

            if is_seg:
                result = F.interpolate(data[None].float(), new_shape, mode=torch_mode, antialias=False)[0]
                #binarize
                result = (result > 0.5).type(torch.uint8)
            else:
                result = F.interpolate(data[None].float(), new_shape, mode=torch_mode, antialias=False)[0]
            if input_was_numpy:
                result = result.cpu().numpy()
            else:
                result = result.to(orig_device)
        torch.set_num_threads(n_threads)
        return result


def resample_torch_fornnunet(
        data: Union[torch.Tensor, np.ndarray],
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        is_seg: bool = False,
        num_threads: int = 4,
        device: torch.device = torch.device('cpu'),
        memefficient_seg_resampling: bool = False,
        force_separate_z: Union[bool, None] = None,
        mode='trilinear',
        aniso_axis_mode='nearest-exact'
):
    """
    data must be c, x, y, z
    """
    assert data.ndim == 4, "data must be c, x, y, z"
    new_shape = [int(i) for i in new_shape]
    orig_shape = data.shape

    do_separate_z, axis = True,len(orig_shape)-1
    # print('shape', data.shape, 'current_spacing', current_spacing, 'new_spacing', new_spacing, 'do_separate_z', do_separate_z, 'axis', axis)

    if do_separate_z:
        was_numpy = isinstance(data, np.ndarray)
        if was_numpy:
            data = torch.from_numpy(data)

        assert len(axis) == 1
        axis = axis[0]
        tmp = "xyz"
        axis_letter = tmp[axis]
        others_int = [i for i in range(3) if i != axis]
        others = [tmp[i] for i in others_int]

        # reshape by overloading c channel
        data = rearrange(data, f"c x y z -> (c {axis_letter}) {others[0]} {others[1]}")

        # reshape in-plane
        tmp_new_shape = [new_shape[i] for i in others_int]
        data = resample_torch_simple(data, tmp_new_shape, is_seg=is_seg, num_threads=num_threads, device=device,
                                     memefficient_seg_resampling=memefficient_seg_resampling, mode=mode)
        data = rearrange(data, f"(c {axis_letter}) {others[0]} {others[1]} -> c x y z",
                         **{
                             axis_letter: orig_shape[axis + 1],
                             others[0]: tmp_new_shape[0],
                             others[1]: tmp_new_shape[1]
                         }
                         )
        # reshape out of plane w/ nearest
        data = resample_torch_simple(data, new_shape, is_seg=is_seg, num_threads=num_threads, device=device,
                                     memefficient_seg_resampling=memefficient_seg_resampling, mode=aniso_axis_mode)
        if was_numpy:
            data = data.numpy()
        return data
    else:
        return resample_torch_simple(data, new_shape, is_seg, num_threads, device, memefficient_seg_resampling)