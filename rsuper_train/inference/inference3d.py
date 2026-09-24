import os as _os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import split_idx
import pdb
import numpy as np
#from ..training.augmentation import crop_foreground_3d


def _lesion_like_name(segment_name):
    """Map an organ entry of `organs_with_tumor` to its lesion channel name.

    Mirrors `lesion_like_name` in RT-Super/foundational/MedFormer/tumor_info_builder.py
    (which itself mirrors the dataset's canonicalisation at lines 2887-2895).

    Examples:
        'bladder'            -> 'bladder_lesion'
        'gall_bladder'       -> 'gallbladder_lesion'
        'adrenal_gland_left' -> 'adrenal_lesion'
        'pancreas'           -> 'pancreatic_lesion'
    """
    s = (segment_name
         .replace(' ', '_')
         .replace('_right', '')
         .replace('_left', '')
         .replace('_gland', '')
         .replace('gall_bladder', 'gallbladder'))
    if ('liver' in s) or ('segment' in s):
        s = 'liver'
    elif ('pancrea' in s) or ('head' in s) or ('body' in s) or ('tail' in s):
        s = 'pancreatic'
    return s + '_lesion'


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

    return torch.sigmoid(pred)

def classification_to_3D(pred,D,H,W):
    if 'classification on segmentation' in list(pred.keys()):
        pred_cls = pred['classification on segmentation']
    elif 'classification on output' in list(pred.keys()):
        pred_cls = pred['classification on output']
    elif 'classification' in list(pred.keys()):
        pred_cls = pred['classification']
    else:
        pred_cls = None
    if pred_cls is not None and (isinstance(pred_cls, tuple) or isinstance(pred_cls, list)):
        pred_cls = pred_cls[-1]
    if pred_cls is not None:
        #make it match the shape of pred by adding the 3 spatial dimensions (B,C) -> (B,C,D,H,W)
        pred_cls = torch.sigmoid(pred_cls)
        pred_cls = pred_cls.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, D,H,W)
    #if pred_cls is None:
    #    raise ValueError(f"No classification output found in the model output dictionary. Available keys: {list(pred.keys())}")
    return pred_cls


# ---------------- TRAINING-FAITHFUL TTA (proved equivalent to the loader) ----------------
import math as _math, numpy as _np, torch.nn.functional as _F
from training import augmentation as _aug
_SCALE=[0,0,0]; _ROT=[30,30,30]; _TRANS=[0,0,0]; _SHEAR=[0.05]*3   # released model's config.txt

def _build_theta():
    """verbatim sampling order of augmentation.random_scale_rotate_translate_3d"""
    sx=_np.random.uniform(1-_SCALE[0],1/(1-_SCALE[0])); sy=_np.random.uniform(1-_SCALE[1],1/(1-_SCALE[1])); sz=_np.random.uniform(1-_SCALE[2],1/(1-_SCALE[2]))
    hxy=_np.random.uniform(-_SHEAR[0],_SHEAR[0]); hxz=_np.random.uniform(-_SHEAR[0],_SHEAR[0])
    hyx=_np.random.uniform(-_SHEAR[1],_SHEAR[1]); hyz=_np.random.uniform(-_SHEAR[1],_SHEAR[1])
    hzx=_np.random.uniform(-_SHEAR[2],_SHEAR[2]); hzy=_np.random.uniform(-_SHEAR[2],_SHEAR[2])
    tx=_np.random.uniform(-_TRANS[0],_TRANS[0]); ty=_np.random.uniform(-_TRANS[1],_TRANS[1]); tz=_np.random.uniform(-_TRANS[2],_TRANS[2])
    th_s=torch.tensor([[sx,hxy,hxz,tx],[hyx,sy,hyz,ty],[hzx,hzy,sz,tz],[0,0,0,1]]).float()
    ax=(float(_np.random.randint(-_ROT[0],max(_ROT[0],1)))/180.)*_math.pi
    ay=(float(_np.random.randint(-_ROT[1],max(_ROT[1],1)))/180.)*_math.pi
    az=(float(_np.random.randint(-_ROT[2],max(_ROT[2],1)))/180.)*_math.pi
    Rx=torch.tensor([[1,0,0,0],[0,_math.cos(ax),-_math.sin(ax),0],[0,_math.sin(ax),_math.cos(ax),0],[0,0,0,1]]).float()
    Ry=torch.tensor([[_math.cos(ay),0,-_math.sin(ay),0],[0,1,0,0],[_math.sin(ay),0,_math.cos(ay),0],[0,0,0,1]]).float()
    Rz=torch.tensor([[_math.cos(az),-_math.sin(az),0,0],[_math.sin(az),_math.cos(az),0,0],[0,0,1,0],[0,0,0,1]]).float()
    th=torch.mm(Rx,Ry); th=torch.mm(th,Rz); return torch.mm(th,th_s)

def _apply_theta(x, th4, mode='bilinear'):
    th=th4[0:3,:].unsqueeze(0).to(x.device)
    grid=_F.affine_grid(th, x.size(), align_corners=True).to(x.device)
    return _F.grid_sample(x, grid, mode=mode, padding_mode='zeros', align_corners=True)


def _centre_crop(x, size):
    _,_,D,H,W = x.shape; d,h,w = size
    sd=(D-d)//2; sh=(H-h)//2; sw=(W-w)//2
    return x[:,:,sd:sd+d, sh:sh+h, sw:sw+w]

def _train_tta_view_ctx(img, d0,d1,h0,h1,w0,w1, seed, effective_only=False):
    """Training-faithful view: take a LARGER region (+20,+40,+40 as in the loader),
       apply the spatial transform, then centre-crop to the window -- so no zero
       wedges ever enter the model, exactly as crop_3d(...,mode='center') ensures
       in dataset_abdomenatlas_UFO.py line 617-618."""
    _np.random.seed(seed); torch.manual_seed(seed)
    B,C,D,H,W = img.shape
    wd,wh,ww = d1-d0, h1-h0, w1-w0
    md,mh,mw = 10,20,20                      # half of +20,+40,+40
    ed0,ed1 = max(d0-md,0), min(d1+md,D)
    eh0,eh1 = max(h0-mh,0), min(h1+mh,H)
    ew0,ew1 = max(w0-mw,0), min(w1+mw,W)
    big = img[:,:,ed0:ed1, eh0:eh1, ew0:ew1]
    # replicate-pad if the window sits at the volume border (training never sees zeros here)
    pad=(mw-(w0-ew0), mw-(ew1-w1), mh-(h0-eh0), mh-(eh1-h1), md-(d0-ed0), md-(ed1-d1))
    if any(v>0 for v in pad):
        big = _F.pad(big, tuple(max(v,0) for v in pad), mode='replicate')
    th=_build_theta()
    big = _apply_theta(big, th)
    x = _centre_crop(big, (wd,wh,ww))
    if effective_only:
        x=_aug.gamma(x, gamma_range=[0.7,1.5])
        x=_aug.gaussian_blur(x, sigma_range=[0.5,1.5])
        x=_aug.gaussian_noise(x, std=_np.random.random()*0.2)
    else:
        if _noaug: pass
        elif _np.random.random() < 0.3: x=_aug.brightness_multiply(x, multiply_range=[0.7,1.3])
        if (not _noaug) and _np.random.random() < 0.3: x=_aug.brightness_additive(x, std=0.1)
        if (not _noaug) and _np.random.random() < 0.3: x=_aug.gamma(x, gamma_range=[0.7,1.5])
        if (not _noaug) and _np.random.random() < 0.3: x=_aug.contrast(x, contrast_range=[0.7,1.3])
        if (not _noaug) and _np.random.random() < 0.3: x=_aug.gaussian_blur(x, sigma_range=[0.5,1.5])
        if (not _noaug) and _np.random.random() < 0.3: x=_aug.gaussian_noise(x, std=_np.random.random()*0.2)
    return x, th, (wd,wh,ww)

def _invert_pred(p, th, win):
    """Embed the window-sized prediction back into the enlarged frame, undo the
       transform, and centre-crop back to the window."""
    wd,wh,ww = win
    big = _F.pad(p, (20,20,20,20,10,10))      # back to (+20,+40,+40)
    big = _apply_theta(big, torch.inverse(th))
    return _centre_crop(big, (wd,wh,ww))


# ================= REAL-METHOD TTA (uses the TRAINING dataset code) =================
_TRAIN_DS=[None]

def _intensity_views(x, seed, n):
    """N views of the SAME patch using the training intensity block verbatim
       (dataset_abdomenatlas_UFO.py:1049-1062).  View 0 is the unaugmented patch.
       No geometry change: caller keeps the sliding window's own accumulation."""
    out=[x]
    for v in range(1,n):
        _np.random.seed(seed+v); torch.manual_seed(seed+v)
        y=x.clone()
        if _np.random.random() < 0.3: y=_aug.brightness_multiply(y, multiply_range=[0.7,1.3])
        if _np.random.random() < 0.3: y=_aug.brightness_additive(y, std=0.1)
        if _np.random.random() < 0.3: y=_aug.gamma(y, gamma_range=[0.7,1.5])
        if _np.random.random() < 0.3: y=_aug.contrast(y, contrast_range=[0.7,1.3])
        if _np.random.random() < 0.3: y=_aug.gaussian_blur(y, sigma_range=[0.5,1.5])
        if _np.random.random() < 0.3: y=_aug.gaussian_noise(y, std=_np.random.random()*0.2)
        out.append(y)
    return out



def inference_boxcentre_tta(net, img, args, n_views=4, class_list=None):
    """EXPERIMENT 2: crop around the CENTRE of each sliding-window box that contains
       pancreas, then apply the TRAINING augmentations (spatial + intensity) for TTA.
       Uses the training functions: crop_around_coordinate_3d for the enlarged crop,
       random_scale_rotate_translate_3d for the spatial transform, crop_3d for the
       centre-crop, and the loader's intensity block.  Coordinate tracers ride in the
       image so the prediction can be mapped back exactly."""
    import types as _t
    p1,_cls = inference_sliding_window_one_pass(net, img, args, pancreas=None, gaussian=False)
    p1=p1.float()
    cl=list(class_list) if class_list is not None else list(args.class_list)
    les=[i for i,c in enumerate(cl) if c=='pancreatic_lesion']
    pan=[i for i,c in enumerate(cl) if c=='pancreas']
    if not les: return p1,_cls
    LES=les[0]; PAN=pan[0] if pan else None
    # Gate: box-centre TTA gains on medium/large lesions but collapses when pass-1's
    # lesion is tiny or already failing (the crop is then mostly non-lesion context).
    # Fall back to pass-1 in that regime.  THRESHOLD MUST BE SET ON A VALIDATION SET
    # (PanTS / PANORAMA positives), not on the test data.
    _minles=float(_os.environ.get('RSUPER_BOXMINLES','0'))
    _lesvox=float((p1[0,LES]>0.5).sum())
    if _minles>0 and _lesvox<_minles:
        print(f'[box-tta] pass-1 lesion {_lesvox:.0f} vox < {_minles:.0f}: fall back to pass-1', flush=True)
        return p1,_cls
    B,C,D,H,W=img.shape
    wd,wh,ww=args.window_size
    hd,hh,hw=wd//2,wh//2,ww//2
    organ=(p1[0,PAN]>0.5).cpu() if PAN is not None else None
    gw=make_gaussian_kernel(wd,wh,ww,sigma_scale=0.25)[0,0].float().cpu()
    ct=img.float().cpu()
    zz,yy,xx=torch.meshgrid(torch.arange(D),torch.arange(H),torch.arange(W),indexing='ij')
    img5=torch.cat([ct[:, :1],(zz+1).float()[None,None],(yy+1).float()[None,None],
                    (xx+1).float()[None,None],torch.ones(1,1,D,H,W)],dim=1)
    dummy=torch.zeros(1,1,D,H,W)
    acc=p1[0,LES].cpu().clone(); cnt=torch.ones(D,H,W)
    nbox=0
    with torch.no_grad():
        for i in range(D//hd):
            for j in range(H//hh):
                for k in range(W//hw):
                    d0,d1=split_idx(hd,D,i); h0,h1=split_idx(hh,H,j); w0,w1=split_idx(hw,W,k)
                    if organ is not None and organ[d0:d1,h0:h1,w0:w1].sum()==0: continue
                    nbox+=1
                    ctr=torch.tensor([(d0+d1)//2,(h0+h1)//2,(w0+w1)//2])
                    for v in range(n_views):
                        _np.random.seed(7919*nbox+v); torch.manual_seed(7919*nbox+v)
                        big,_=_aug.crop_around_coordinate_3d(img5, dummy, [wd+20,wh+40,ww+40],
                                                             ctr, 'small_rnd_shift')
                        if v>0:
                            big,_=_aug.random_scale_rotate_translate_3d(big, torch.zeros_like(big[:, :1]),
                                                                        args.scale,args.rotate,args.translate)
                        cr,_=_aug.crop_3d(big, torch.zeros_like(big[:, :1]), [wd,wh,ww], mode='center')
                        x=cr[:, :1].clone()
                        if v>0:
                            if _np.random.random()<0.3: x=_aug.brightness_multiply(x, multiply_range=[0.7,1.3])
                            if _np.random.random()<0.3: x=_aug.brightness_additive(x, std=0.1)
                            if _np.random.random()<0.3: x=_aug.gamma(x, gamma_range=[0.7,1.5])
                            if _np.random.random()<0.3: x=_aug.contrast(x, contrast_range=[0.7,1.3])
                            if _np.random.random()<0.3: x=_aug.gaussian_blur(x, sigma_range=[0.5,1.5])
                            if _np.random.random()<0.3: x=_aug.gaussian_noise(x, std=_np.random.random()*0.2)
                        # A box near a volume border comes back smaller than the window, and
                        # MedFormer's patch_merging cannot split an odd spatial dim (27 -> 14+13),
                        # which crashed 31/250 PANORAMA cases (thick-slice scans). Pad up to the
                        # window for the forward pass, then crop the output back so it still
                        # aligns with the tracer channels in cr. No-op when the box is full size.
                        _xs=tuple(x.shape[2:])
                        if _xs!=(wd,wh,ww):
                            print(f'[box-tta] padding box {_xs} -> {(wd,wh,ww)}', flush=True)
                            x=_F.pad(x,(0,ww-_xs[2],0,wh-_xs[1],0,wd-_xs[0]))
                        o=net(x.cuda().float())
                        pr=o['segmentation'] if isinstance(o,dict) else o
                        while isinstance(pr,(tuple,list)): pr=pr[0]
                        pr=torch.sigmoid(pr)[0,LES].float().cpu()
                        if _xs!=(wd,wh,ww):
                            pr=pr[:_xs[0],:_xs[1],:_xs[2]]
                        fz=cr[0,1]-1.0; fy=cr[0,2]-1.0; fx=cr[0,3]-1.0
                        m=(cr[0,4]>0.999)&(fz>=0)&(fy>=0)&(fx>=0)&(fz<=D-1)&(fy<=H-1)&(fx<=W-1)
                        if m.sum()==0: continue
                        # gw is full-window; for a clipped border box use the matching sub-block
                        # (padding above is applied at the end of each dim, so the valid data is
                        # the leading sub-block).
                        _gwv = gw if _xs==(wd,wh,ww) else gw[:_xs[0],:_xs[1],:_xs[2]]
                        _w=_gwv[m]; _p=pr[m]
                        z0=torch.floor(fz[m]); y0=torch.floor(fy[m]); x0=torch.floor(fx[m])
                        dz=fz[m]-z0; dy=fy[m]-y0; dx=fx[m]-x0
                        z0=z0.long(); y0=y0.long(); x0=x0.long()
                        for _iz in (0,1):
                            for _iy in (0,1):
                                for _ix in (0,1):
                                    wz=dz if _iz else (1-dz); wy=dy if _iy else (1-dy); wx=dx if _ix else (1-dx)
                                    ww_=_w*wz*wy*wx
                                    acc.index_put_(((z0+_iz).clamp(0,D-1),(y0+_iy).clamp(0,H-1),
                                                    (x0+_ix).clamp(0,W-1)), _p*ww_, accumulate=True)
                                    cnt.index_put_(((z0+_iz).clamp(0,D-1),(y0+_iy).clamp(0,H-1),
                                                    (x0+_ix).clamp(0,W-1)), ww_, accumulate=True)
    p1[0,LES]=(acc/cnt).to(p1.device)
    print(f'[box-tta] {nbox} pancreas boxes x {n_views} views', flush=True)
    return p1,_cls

def inference_sliding_window(net, img, args, pancreas=None,gaussian=False):
    _bx=int(_os.environ.get('RSUPER_BOXTTA','0'))
    if _bx:
        pred_output, cls_output = inference_boxcentre_tta(net, img, args, n_views=_bx,
                                                          class_list=getattr(args,'class_list',None))
        if cls_output is not None and (cls_output.sum() > 0).item():
            cls_output = torch.amax(cls_output, dim=(-3,-2,-1))
            return pred_output.float(), cls_output.float()
        return pred_output.float()
    if args.inference_2_stages:
        pred_output, cls_output =  inference_2_stages(net, img, args, organs_with_tumors=args.organs_with_tumor, class_list=args.class_list)
    else:
        pred_output, cls_output = inference_sliding_window_one_pass(net, img, args, pancreas=pancreas, gaussian=gaussian)
    
    if cls_output is not None and (cls_output.sum() > 0).item():
        cls_output = torch.amax(cls_output, dim=(-3, -2, -1))
        return pred_output.float(), cls_output.float()
    else:
        return pred_output.float()
    
    
def inference_sliding_window_one_pass(net, img, args, pancreas=None,gaussian=False):
    gaussian = gaussian or _os.environ.get('RSUPER_GAUSS','0')=='1'
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
    
    if gaussian:
        gauss_w = make_gaussian_kernel(win_d, win_h, win_w, sigma_scale=float(_os.environ.get('RSUPER_SIGMA','0.25'))).to(torch.bfloat16).cpu()

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

    pred_output = torch.zeros((B, args.classes, D, H, W),dtype=torch.bfloat16).cpu()#.to(img.device)
    cls_output = None
    cls_counter = torch.zeros((B, 1, D, H, W),dtype=torch.bfloat16).cpu()#.to(img.device)

    counter = torch.zeros((B, 1, D, H, W),dtype=torch.bfloat16).cpu()#.to(img.device)
    one_count = torch.ones((B, 1, win_d, win_h, win_w),dtype=torch.bfloat16).cpu()#.to(img.device)

    with torch.no_grad():
        for i in range(D // half_win_d):
            for j in range(H // half_win_h):
                for k in range(W // half_win_w):
                    
                    d_start_idx, d_end_idx = split_idx(half_win_d, D, i)
                    h_start_idx, h_end_idx = split_idx(half_win_h, H, j)
                    w_start_idx, w_end_idx = split_idx(half_win_w, W, k)

                    input_tensor = img[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx]
                    pred_cls = None # for MTL-like approaches, which can provide the tumor probability
                    if pancreas is None or pancreas[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx].sum() > 0:
                        _ttrain = int(_os.environ.get('RSUPER_TRAINTTA','0'))
                        _teff   = int(_os.environ.get('RSUPER_EFFTTA','0'))
                        if _ttrain or _teff:
                            _n = _ttrain or _teff
                            _eff = bool(_teff)
                            _acc=None
                            for _v in range(_n):
                                if _v==0:
                                    _inp=input_tensor; _th=None; _win=None
                                else:
                                    _inp,_th,_win=_train_tta_view_ctx(img, d_start_idx,d_end_idx,
                                        h_start_idx,h_end_idx, w_start_idx,w_end_idx, _v, effective_only=_eff)
                                _o=net(_inp)
                                _p=_o['segmentation'] if isinstance(_o,dict) else _o
                                while isinstance(_p,(tuple,list)): _p=_p[0]
                                _p=torch.sigmoid(_p)
                                if _th is not None:
                                    _p=_invert_pred(_p, _th, _win)
                                _acc=_p if _acc is None else _acc+_p
                            model_output={'segmentation':_acc/_n}
                            _itta=0; _ftta=0
                            pred=model_output['segmentation']; pred_cls=None
                            _TRAIN_TTA_DONE=True
                        else:
                            _TRAIN_TTA_DONE=False
                        _int = int(_os.environ.get('RSUPER_INTTTA','0'))
                        if _int>1:
                            # EXPERIMENT 1: intensity-only TTA, identical geometry.
                            _sd=1000003*(i+1)+1009*(j+1)+13*(k+1)
                            _acc=None
                            for _x in _intensity_views(input_tensor,_sd,_int):
                                _o=net(_x)
                                _pp=_o['segmentation'] if isinstance(_o,dict) else _o
                                while isinstance(_pp,(tuple,list)): _pp=_pp[0]
                                _pp=torch.sigmoid(_pp)
                                _acc=_pp if _acc is None else _acc+_pp
                            model_output={'segmentation':_acc/_int}
                            _INT_DONE=True
                        else:
                            _INT_DONE=False
                        _itta = int(_os.environ.get('RSUPER_ITTA','0')) if not _INT_DONE else 0 if not _TRAIN_TTA_DONE else 0
                        _ftta = int(_os.environ.get('RSUPER_TTA','0'))
                        if (not _TRAIN_TTA_DONE) and (_itta or _ftta or float(_os.environ.get('RSUPER_ROT','0')) or int(_os.environ.get('RSUPER_COMBO','0'))):
                            if _itta:
                                _views=[('id',None),('mul',0.9),('mul',1.1),('con',0.9),
                                        ('con',1.1),('add',0.05),('add',-0.05),('mul',0.8)][:_itta]
                            else:
                                _views=[('flip',a) for a in
                                        [[],[-1],[-2],[-3],[-1,-2],[-1,-3],[-2,-3],[-1,-2,-3]][:_ftta]]
                            _combo=int(_os.environ.get('RSUPER_COMBO','0'))
                            _rot=float(_os.environ.get('RSUPER_ROT','0'))
                            if _rot:
                                # rotation TTA about the axial (z) axis, within the +-30 deg
                                # range used in training (rotate:[30,30,30])
                                _views=[('rot',0.0),('rot',_rot),('rot',-_rot),
                                        ('rot',2*_rot),('rot',-2*_rot)][:int(_os.environ.get('RSUPER_NROT','3'))]
                            if _combo:
                                # rotation x intensity, both inside the trained ranges
                                _r=float(_os.environ.get('RSUPER_ROT','15'))
                                _views=[('ri',(0.0,1.0)),('ri',(_r,1.0)),('ri',(-_r,1.0)),
                                        ('ri',(0.0,0.9)),('ri',(0.0,1.1)),
                                        ('ri',(_r,0.9)),('ri',(-_r,1.1))][:_combo]
                            _acc=None
                            for _k,_v in _views:
                                if _k=='id' or (_k=='flip' and not _v): _inp=input_tensor
                                elif _k=='mul': _inp=input_tensor*_v
                                elif _k=='add': _inp=input_tensor+_v
                                elif _k=='con':
                                    _m=input_tensor.mean(); _inp=(input_tensor-_m)*_v+_m
                                elif _k=='ri':
                                    _ang,_mul=_v
                                    _inp=input_tensor*_mul
                                    if abs(_ang)>1e-6:
                                        import torchvision.transforms.functional as _TF
                                        _b,_c,_d,_h,_w=_inp.shape
                                        _x=_inp.permute(0,2,1,3,4).reshape(_b*_d,_c,_h,_w)
                                        _x=_TF.rotate(_x,_ang)
                                        _inp=_x.reshape(_b,_d,_c,_h,_w).permute(0,2,1,3,4)
                                elif _k=='rot':
                                    if abs(_v)<1e-6: _inp=input_tensor
                                    else:
                                        import torchvision.transforms.functional as _TF
                                        _b,_c,_d,_h,_w=input_tensor.shape
                                        _x=input_tensor.permute(0,2,1,3,4).reshape(_b*_d,_c,_h,_w)
                                        _x=_TF.rotate(_x,_v)
                                        _inp=_x.reshape(_b,_d,_c,_h,_w).permute(0,2,1,3,4)
                                else: _inp=torch.flip(input_tensor,_v)
                                _o=net(_inp)
                                _p=_o['segmentation'] if isinstance(_o,dict) else _o
                                while isinstance(_p,(tuple,list)): _p=_p[0]
                                _p=torch.sigmoid(_p)
                                if _k=='flip' and _v: _p=torch.flip(_p,_v)
                                if _k=='ri' and abs(_v[0])>1e-6:
                                    import torchvision.transforms.functional as _TF
                                    _b,_c,_d,_h,_w=_p.shape
                                    _y=_p.permute(0,2,1,3,4).reshape(_b*_d,_c,_h,_w)
                                    _y=_TF.rotate(_y,-_v[0])
                                    _p=_y.reshape(_b,_d,_c,_h,_w).permute(0,2,1,3,4)
                                if _k=='rot' and abs(_v)>1e-6:
                                    import torchvision.transforms.functional as _TF
                                    _b,_c,_d,_h,_w=_p.shape
                                    _y=_p.permute(0,2,1,3,4).reshape(_b*_d,_c,_h,_w)
                                    _y=_TF.rotate(_y,-_v)
                                    _p=_y.reshape(_b,_d,_c,_h,_w).permute(0,2,1,3,4)
                                _acc=_p if _acc is None else _acc+_p
                            model_output={'segmentation':_acc/len(_views)}
                        elif not (_TRAIN_TTA_DONE or _INT_DONE):
                            model_output = net(input_tensor)
                        
                        if isinstance(model_output, dict):
                            pred = model_output['segmentation']
                            
                        if isinstance(pred, tuple) or isinstance(pred, list):
                            pred = pred[0]
                        if isinstance(pred, tuple) or isinstance(pred, list):
                            pred = pred[0]
                            
                        if isinstance(model_output, dict):
                            pred_cls = classification_to_3D(model_output, pred.shape[-3], pred.shape[-2], pred.shape[-1])
                        
                        if (_INT_DONE or _TRAIN_TTA_DONE or _itta or _ftta or float(_os.environ.get('RSUPER_ROT','0')) or int(_os.environ.get('RSUPER_COMBO','0'))):
                            pass
                        elif not args.epai_stage_2:
                            pred = torch.sigmoid(pred)
                            #print('using sigmoid')
                        else:
                            pred = F.softmax(pred, dim=1)
                        
                        #pred_cls is trained with sigmoid loss (BCE)
                    else:
                        #print('Skipped ')
                        pred = torch.zeros((B, args.classes, win_d, win_h, win_w),dtype=torch.bfloat16)

                    if pred_cls is not None:
                        if cls_output is None:
                            cls_output = torch.zeros((B, pred_cls.shape[1], D, H, W),dtype=torch.bfloat16).cpu()
                        cls_output[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred_cls.to(torch.bfloat16).cpu()
                        cls_counter[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += one_count
                    
                    
                    if not gaussian:
                        pred_output[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred.to(torch.bfloat16).cpu()
                        counter[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += one_count
                    else:
                        # Slice the Gaussian kernel to the actual patch size
                        w_patch = gauss_w[
                            :,
                            :,
                            : d_end_idx - d_start_idx,
                            : h_end_idx - h_start_idx,
                            : w_end_idx - w_start_idx,
                        ]

                        # Broadcast to match classes and batch dims
                        w_patch_cls = w_patch.expand(B, args.classes, *w_patch.shape[2:])
                        w_patch_cnt = w_patch.expand(B, 1,           *w_patch.shape[2:])

                        # Accumulate weighted logits (or probabilities)
                        pred_output[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += (pred.to(torch.bfloat16).cpu() * w_patch_cls)
                        # Accumulate the weights themselves
                        counter[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += w_patch_cnt

    pred_output /= (counter + 1e-6)
    if flag:
        pred_output = pred_output[:, :, :origin_D, :origin_H, :origin_W]
        
    if cls_output is not None and (cls_output.sum() > 0).item():
        cls_output /= (cls_counter + 1e-6)
        return pred_output, cls_output

    return pred_output, None

                    
def make_gaussian_kernel(win_d, win_h, win_w, sigma_scale=0.25):
    """
    Return a tensor of shape (1, 1, win_d, win_h, win_w) whose values peak at 1
    in the centre and taper off with a 3-D Gaussian.
    sigma_scale : fraction of the window size -– 0.125 → σ ≈ win_dim / 8
    """
    z = torch.linspace(-1, 1, steps=win_d)
    y = torch.linspace(-1, 1, steps=win_h)
    x = torch.linspace(-1, 1, steps=win_w)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    # choose separate σ for each axis
    sigma_d = sigma_scale * 2         # range (-1,1) ⇒ length 2
    sigma_h = sigma_scale * 2
    sigma_w = sigma_scale * 2

    g = torch.exp(
        -((zz / sigma_d) ** 2 + (yy / sigma_h) ** 2 + (xx / sigma_w) ** 2) / 2
    )
    g /= g.max()            # centre = 1.0
    return g.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)


def inference_2_stages(net, img, args, organs_with_tumors=None, class_list=None):
    #stage 1: coarse, run over all the CT
    pred_output, cls_output = inference_sliding_window_one_pass(net, img, args, pancreas=None,gaussian=True)
    pred_output_stage_1_binary = (pred_output > 0.5).float()
    #stage 2: for each organ in organs_with_tumors, crop and run stage-2 inference,
    # but ONLY paste back the lesion channel for that organ (and the matching cls
    # channel). Sparse per-channel buffers replace the (1, classes, D, H, W) and
    # (1, n_cls, D, H, W) bfloat16 buffers — ~74/9 ≈ 8× cut on the seg buffer.
    # Mirrors inference3d_teacher.inference_2_stages_teacher (commit 05b5f4d).
    B, C, D, H, W = img.shape
    win_d, win_h, win_w = args.window_size

    # cls channel ordering = alphabetically-sorted lesion classes (matches the
    # ordering produced by predict_abdomenatlas.init_model for the cls head).
    cls_class_list = sorted([c for c in class_list if 'lesion' in c])

    for batch in range(B):
        pred_per_ch = {}         # seg lesion_ch -> (D, H, W) bfloat16 sum
        counter_per_ch = {}      # seg lesion_ch -> (D, H, W) bfloat16 counter
        cls_per_ch = {}          # cls_ch -> (D, H, W) bfloat16 sum
        cls_counter_per_ch = {}  # cls_ch -> (D, H, W) bfloat16 counter
        one_count = torch.ones((win_d, win_h, win_w), dtype=torch.bfloat16).cpu()

        for org in organs_with_tumors:
            org_idx = class_list.index(org)
            #check if the organ is present
            if pred_output_stage_1_binary[batch, org_idx, :, :, :].sum() > 0:
                #get the mask of the organ
                organ_mask = pred_output_stage_1_binary[batch, org_idx, :, :, :]
                x=img[batch,0]
                out = crop_foreground_3d(tensor_ct=x, tensor_lab=pred_output_stage_1_binary[batch], foreground=organ_mask,
                                         crop_size=[win_d, win_h, win_w],rand=False,return_coordinate=True)
                if not isinstance(out, tuple):
                    #failed crop on organ
                    print(f"Failed to crop on organ {org}: {out}")
                    continue
                cropped_ct, _, cropped_organ, coord = out
                d_start_idx,d_end_idx, h_start_idx,h_end_idx, w_start_idx,w_end_idx = coord

                # Resolve the seg lesion channel for this organ before running
                # stage 2 — if there's no matching channel in class_list, skip
                # (no place to paste).
                lesion_name = _lesion_like_name(org)
                if lesion_name not in class_list:
                    print(f"[stage2] no lesion channel for {org} (lesion_name={lesion_name}) — skip")
                    continue
                lesion_ch = class_list.index(lesion_name)

                #run inference on the cropped ct
                _s2 = int(_os.environ.get('RSUPER_S2TTA','0'))
                if _s2:
                    # --- training-faithful TTA on the ORGAN-CENTRED crop (pass 2) ---
                    # training: random_crop_on_tumor(d+20,h+40,w+40) -> spatial -> centre-crop
                    # here:     organ-centred crop (+20,+40,+40)     -> spatial -> centre-crop
                    _eff = _os.environ.get('RSUPER_S2EFF','0')=='1'
                    _outb = crop_foreground_3d(tensor_ct=x, tensor_lab=pred_output_stage_1_binary[batch],
                                               foreground=organ_mask,
                                               crop_size=[win_d+20, win_h+40, win_w+40],
                                               rand=False, return_coordinate=True)
                    _big = _outb[0].unsqueeze(0).unsqueeze(0) if isinstance(_outb, tuple) else None
                    _acc=None
                    with torch.no_grad():
                        for _v in range(_s2):
                            if _v==0 or _big is None:
                                _inp=cropped_ct.unsqueeze(0).unsqueeze(0); _th=None
                            else:
                                _np.random.seed(_v); torch.manual_seed(_v)
                                _th=_build_theta()
                                _r=_apply_theta(_big, _th)
                                _inp=_centre_crop(_r, (win_d, win_h, win_w))
                                if _eff:
                                    _inp=_aug.gamma(_inp, gamma_range=[0.7,1.5])
                                    _inp=_aug.gaussian_blur(_inp, sigma_range=[0.5,1.5])
                                    _inp=_aug.gaussian_noise(_inp, std=_np.random.random()*0.2)
                                else:
                                    if _np.random.random()<0.3: _inp=_aug.brightness_multiply(_inp, multiply_range=[0.7,1.3])
                                    if _np.random.random()<0.3: _inp=_aug.brightness_additive(_inp, std=0.1)
                                    if _np.random.random()<0.3: _inp=_aug.gamma(_inp, gamma_range=[0.7,1.5])
                                    if _np.random.random()<0.3: _inp=_aug.contrast(_inp, contrast_range=[0.7,1.3])
                                    if _np.random.random()<0.3: _inp=_aug.gaussian_blur(_inp, sigma_range=[0.5,1.5])
                                    if _np.random.random()<0.3: _inp=_aug.gaussian_noise(_inp, std=_np.random.random()*0.2)
                            _o=net(_inp)
                            _p=_o['segmentation'] if isinstance(_o,dict) else _o
                            while isinstance(_p,(tuple,list)): _p=_p[0]
                            _p=torch.sigmoid(_p)
                            if _th is not None:
                                _p=_invert_pred(_p, _th, (win_d, win_h, win_w))
                            _acc=_p if _acc is None else _acc+_p
                    model_output={'segmentation': _acc/_s2}
                    _S2_TTA=True
                else:
                    _S2_TTA=False
                with torch.no_grad():
                    if not _S2_TTA:
                        model_output = net(cropped_ct.unsqueeze(0).unsqueeze(0))
                    pred_cls = None
                    if isinstance(model_output, dict):
                        pred = model_output['segmentation']
                    if isinstance(pred, tuple) or isinstance(pred, list):
                        pred = pred[0]
                    if isinstance(pred, tuple) or isinstance(pred, list):
                        pred = pred[0]
                    if isinstance(model_output, dict):
                        pred_cls = classification_to_3D(model_output, pred.shape[-3], pred.shape[-2], pred.shape[-1])

                    if _S2_TTA:
                        pass
                    elif not args.epai_stage_2:
                        pred = torch.sigmoid(pred)
                    else:
                        pred = F.softmax(pred, dim=1)

                # Paste only the seg lesion channel for this organ.
                if lesion_ch not in pred_per_ch:
                    pred_per_ch[lesion_ch] = torch.zeros((D, H, W), dtype=torch.bfloat16).cpu()
                    counter_per_ch[lesion_ch] = torch.zeros((D, H, W), dtype=torch.bfloat16).cpu()
                pred_per_ch[lesion_ch][d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred[0, lesion_ch].to(torch.bfloat16).cpu()
                counter_per_ch[lesion_ch][d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += one_count

                # Paste only the cls channel for this lesion (cls head ordering
                # matches sorted(lesion_classes) per predict_abdomenatlas init).
                if pred_cls is not None and lesion_name in cls_class_list:
                    cls_ch = cls_class_list.index(lesion_name)
                    if cls_ch not in cls_per_ch:
                        cls_per_ch[cls_ch] = torch.zeros((D, H, W), dtype=torch.bfloat16).cpu()
                        cls_counter_per_ch[cls_ch] = torch.zeros((D, H, W), dtype=torch.bfloat16).cpu()
                    cls_per_ch[cls_ch][d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred_cls[0, cls_ch].to(torch.bfloat16).cpu()
                    cls_counter_per_ch[cls_ch][d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += one_count

        # Blend per-channel: average the stage-2 paste-backs over the counter,
        # then replace stage-1 voxels where counter>0 with the average.
        # Stage-1 values on other channels and on untouched voxels are kept.
        for ch, pred_t in pred_per_ch.items():
            counter_t = counter_per_ch[ch]
            ch_mask = (counter_t > 0).float()
            counter_safe = counter_t * ch_mask + 1e-6 * (1 - ch_mask)
            avg = pred_t / counter_safe
            pred_output[batch, ch] = (1 - ch_mask) * pred_output[batch, ch] + ch_mask * avg

        if cls_output is not None:
            for ch, pred_t in cls_per_ch.items():
                counter_t = cls_counter_per_ch[ch]
                ch_mask = (counter_t > 0).float()
                counter_safe = counter_t * ch_mask + 1e-6 * (1 - ch_mask)
                avg = pred_t / counter_safe
                cls_output[batch, ch] = (1 - ch_mask) * cls_output[batch, ch] + ch_mask * avg

    return pred_output, cls_output
                
                




def crop_foreground_3d(tensor_ct, tensor_lab, foreground, crop_size, margin=1, refine_iterations=3, rand=True, return_coordinate=False):
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
    
    assert foreground.shape[-3:]==tensor_ct.shape[-3:], f"Foreground shape must match CT shape, got {foreground.shape} and {tensor_ct.shape}"
    
    backup_foreground = label_3d.clone()
        
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
    if desired_d > D or desired_h > H or desired_w > W:
        return "requesting crop larger than the CT!!"

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
    if torch.count_nonzero(cropped_fg) == 0 or \
        (torch.count_nonzero(cropped_fg) >= torch.count_nonzero(label_3d)*1.5) or \
        (torch.count_nonzero(cropped_fg) <= torch.count_nonzero(label_3d)*0.5):
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
        
        cropped_fg_deter = backup_foreground[z_start:z_end, y_start:y_end, x_start:x_end]
        
        print('Deter foreground total:',torch.count_nonzero(cropped_fg_deter))
        raise ValueError("zero mask after crop")
    
    if return_coordinate:
        coord=[z_start,z_end, y_start,y_end, x_start,x_end]
        return (cropped_ct, cropped_label, cropped_fg, coord)
    else:
        return (cropped_ct, cropped_label, cropped_fg)
    

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
