import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import split_idx
import pdb


def inference_whole_image(net, img, args=None):
    '''
    img: torch tensor, B, C, D, H, W
    return: prob (after softmax), B, classes, D, H, W

    Use this function to inference if whole image can be put into GPU without memory issue
    Better to be consistent with the training window size
    '''

    net.eval()

    with torch.no_grad():
        pred = net(img)

        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = pred[0]

    return F.sigmoid(pred)


def inference_sliding_window(net, img, args, pancreas=None):
    '''
    img: torch tensor, B, C, D, H, W
    return: prob (after softmax), B, classes, D, H, W
    pancreas: pancreas mask, used in pancreas_only_inference

    The overlap of two windows will be half the window size

    Use this function to inference if out-of-memory occurs when whole image inferencing
    Better to be consistent with the training window size
    '''
    net.eval()
    
    if pancreas is not None:
        while len(pancreas.shape) < len(img.shape):
            pancreas = pancreas.unsqueeze(0)
        assert pancreas.shape == img.shape, f"Pancreas mask shape must match image shape, got {pancreas.shape} and {img.shape}"

    B, C, D, H, W = img.shape

    win_d, win_h, win_w = args.window_size

    flag = False
    if D < win_d or H < win_h or W < win_w:
        flag = True
        diff_D = max(0, win_d-D)
        diff_H = max(0, win_h-H)
        diff_W = max(0, win_w-W)

        img = F.pad(img, (0, diff_W, 0, diff_H, 0, diff_D))
        
        origin_D, origin_H, origin_W = D, H, W
        B, C, D, H, W = img.shape


    half_win_d = win_d // 2
    half_win_h = win_h // 2
    half_win_w = win_w // 2

    pred_output = torch.zeros((B, args.classes, D, H, W)).cpu()#.to(img.device)

    counter = torch.zeros((B, 1, D, H, W)).cpu()#.to(img.device)
    one_count = torch.ones((B, 1, win_d, win_h, win_w)).cpu()#.to(img.device)

    with torch.no_grad():
        for i in range(D // half_win_d):
            for j in range(H // half_win_h):
                for k in range(W // half_win_w):
                    
                    d_start_idx, d_end_idx = split_idx(half_win_d, D, i)
                    h_start_idx, h_end_idx = split_idx(half_win_h, H, j)
                    w_start_idx, w_end_idx = split_idx(half_win_w, W, k)

                    input_tensor = img[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx]
                    
                    if pancreas is None or pancreas[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx].sum() > 0:
                        pred = net(input_tensor)
                        if isinstance(pred, dict):
                            pred = pred['segmentation']
                        if isinstance(pred, tuple) or isinstance(pred, list):
                            pred = pred[0]
                        if isinstance(pred, tuple) or isinstance(pred, list):
                            pred = pred[0]
                        
                        pred = F.sigmoid(pred)
                    else:
                        #print('Skipped ')
                        pred = torch.zeros((B, args.classes, win_d, win_h, win_w))

                    

                    pred_output[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred.cpu()

                    counter[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += one_count.cpu()

    pred_output /= counter
    if flag:
        pred_output = pred_output[:, :, :origin_D, :origin_H, :origin_W]

    return pred_output

                    





