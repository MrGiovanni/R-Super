import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_block, get_norm, get_act
from .medformer_utils import down_block, up_block, inconv, SemanticMapFusion
import pdb
import numpy as np

from .trans_layers import TransformerBlock

class ClassificationBranch(nn.Module):
    def __init__(self, in_dim=160, reduced_dim=64, heads=4, dim_head=16, mlp_dim=320, 
                 num_classes=3, extra_layer=None,
                 reducer=True):
        """
        For multi-tumor classification, the voxel_choice input indicates which tumor we want to classify. It is a binary tensor,
        with the same shape as the input, and one voxel is set to 1. The rest are 0. To classify multiple tumors, just run this module
        multiple times, each time with a different voxel_choice input. At inference, you can take the centers of all tumors predicted in 
        segmentation.
        """
        
        super().__init__()
        
        # Add a reducer to lower the channel dimension
        if reducer:
            self.reducer = nn.Conv3d(in_dim, reduced_dim, kernel_size=1)
        else:
            self.reducer = nn.Identity()
        # Optionally, add an extra layer if needed
        self.extra_layer = extra_layer
        # Use a transformer block with the reduced dimension
        self.transformer = TransformerBlock(
            dim=reduced_dim,         # embedding dimension is now reduced_dim
            depth=1,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim
        )
        # Classification head from the reduced dimension to num_classes
        self.head = nn.Linear(reduced_dim, num_classes)

    def forward(self, x, segmentation_out=None, voxel_choice=None):
        """
        x: features of the segmenter
        segmentation_out: segmentation output from the segmenter
        voxel_choice: binary tensor indicating which tumor to classify
        """
        
        if segmentation_out is not None:
            #concatenate the segmentation output with the features
            x = torch.cat((x, segmentation_out), dim=1)
        if voxel_choice is not None:
            #concatenate the voxel_choice with the features
            x = torch.cat((x, voxel_choice), dim=1)
        
        # x is [B, in_dim, D, H, W]
        #print('Shape of x in classification branch:', x.shape)
        if self.extra_layer is not None:
            x, tmp_map = self.extra_layer(x)
        else:
            tmp_map = torch.zeros(1, device=x.device)  # dummy value so gradient flows if needed


        x = self.reducer(x)  # now x becomes [B, reduced_dim, D, H, W]

        # Flatten and rearrange to [B, L, reduced_dim]
        B, C, D, H, W = x.shape
        x = x.flatten(start_dim=2).permute(0, 2, 1).contiguous()
        # Pass through the transformer block
        x = self.transformer(x)  # remains [B, L, reduced_dim]
        # Global average pooling
        x = x.mean(dim=1)  # [B, reduced_dim]
        # Classification head produces output [B, num_classes]
        x = self.head(x)
        # Ensure gradient flows through tmp_map (if needed for DDP)
        x = x + 0 * tmp_map.sum()
        return x
    

class MedFormer(nn.Module):
    
    def __init__(self, 
        in_chan, 
        num_classes, 
        base_chan=32, 
        map_size=[4,8,8], 
        conv_block='BasicBlock', 
        conv_num=[2,1,0,0, 0,1,2,2], 
        trans_num=[0,1,2,2, 2,1,0,0], 
        chan_num=[64,128,256,320,256,128,64,32], 
        num_heads=[1,4,8,16, 8,4,1,1], 
        fusion_depth=2, 
        fusion_dim=320, 
        fusion_heads=4, 
        expansion=4, attn_drop=0., 
        proj_drop=0., 
        proj_type='depthwise', 
        norm='in', 
        act='gelu', 
        kernel_size=[3,3,3,3], 
        scale=[2,2,2,2], 
        aux_loss=False,
        classification_branch=False,
        class_list_seg=None,
        class_list_cls=None,
        clip_branch=False,
        clip_feats=768,
        ):
        super().__init__()

        if conv_block == 'BasicBlock':
            dim_head = [chan_num[i]//num_heads[i] for i in range(8)]

        
        conv_block = get_block(conv_block)
        norm = get_norm(norm)
        act = get_act(act)
        
        # self.inc and self.down1 forms the conv stem
        self.inc = inconv(in_chan, base_chan, block=conv_block, kernel_size=kernel_size[0], norm=norm, act=act)
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block=conv_block, kernel_size=kernel_size[1], down_scale=scale[0], norm=norm, act=act, map_generate=False)
        
        # down2 down3 down4 apply the B-MHA blocks
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block=conv_block, kernel_size=kernel_size[2], down_scale=scale[1], heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block=conv_block, kernel_size=kernel_size[3], down_scale=scale[2], heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)


        self.map_fusion = SemanticMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)

        self.up1 = up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block=conv_block, kernel_size=kernel_size[3], up_scale=scale[3], heads=num_heads[4], dim_head=dim_head[4], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)

        self.up2 = up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block=conv_block, kernel_size=kernel_size[2], up_scale=scale[2], heads=num_heads[5], dim_head=dim_head[5], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True, no_map_out=True)

        self.up3 = up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6], conv_block=conv_block, kernel_size=kernel_size[1], up_scale=scale[1], norm=norm, act=act, map_shortcut=False)

        self.up4 = up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7], conv_block=conv_block, kernel_size=kernel_size[0], up_scale=scale[0], norm=norm, act=act, map_shortcut=False)

        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_out = nn.Conv3d(chan_num[5], num_classes, kernel_size=1)

        self.outc = nn.Conv3d(chan_num[7], num_classes, kernel_size=1)

        if classification_branch:
            self.classification_branch = ClassificationBranch(num_classes=len(class_list_cls),
                                                              extra_layer=down_block(chan_num[3], chan_num[3]//2, 0, 1, conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=4, dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True))
        else:
            self.classification_branch = None
            
        if clip_branch:
            self.clip_branch = ClassificationBranch(num_classes=clip_feats,
                                                    extra_layer=down_block(chan_num[3], chan_num[3]//2, 0, 1, conv_block=conv_block, kernel_size=kernel_size[4], down_scale=scale[3], heads=4, dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True))
        else:
            self.clip_branch = None

        
        if class_list_seg is not None:
            patterns = ('lesion', 'pdac', 'pnet', 'cyst')
            tumor_cls = [c for c in class_list_seg if any(p in c for p in patterns)]
        

    def forward(self, x):
        
        x0 = self.inc(x)
        x1, _ = self.down1(x0)
        x2, map2 = self.down2(x1)
        x3, map3 = self.down3(x2)
        x4, map4 = self.down4(x3)

        if self.classification_branch:
            y_class = self.classification_branch(x4)
        else:
            y_class = None
            
        if self.clip_branch:
            y_clip = self.clip_branch(x4)
        else:
            y_clip = None
        
        map_list = [map2, map3, map4]
        map_list = self.map_fusion(map_list)
        

        out, semantic_map = self.up1(x4, x3, map_list[2], map_list[1])
        out, semantic_map = self.up2(out, x2, semantic_map, map_list[0])

        if self.aux_loss:
            aux_out = self.aux_out(out)
            aux_out = F.interpolate(aux_out, size=x.shape[-3:], mode='trilinear', align_corners=True)
        else:
            aux_out = None

        out, semantic_map = self.up3(out, x1, semantic_map, None)
        out, semantic_map = self.up4(out, x0, semantic_map, None)
        
    
        out = self.outc(out)
        
        return self.prepare_return(out, aux_out=aux_out, y_class=y_class, y_clip=y_clip)
        
    def prepare_return(
        self,
        out,
        aux_out=None,
        y_class=None,
        y_clip=None,
    ):
        # 1) Build the primary output exactly as before (segmentation only, [final output, deep supervision])
        primary = [out, aux_out] if self.aux_loss else out
        
        retur = {'segmentation': primary}
        
        if self.classification_branch:#optional, not needed for R-Super, only for baseline (MTL)
            retur['classification'] = y_class
        if self.clip_branch:#optional, not needed for R-Super, only for baseline (CLIP)
            retur['clip'] = y_clip

        return retur

def update_output_layer_onk(model, original_classes, new_classes, copy_pancreas=False):
    """
    Update the model's final output layers so that they produce outputs for the new set of classes.
    For segmentation layers (model.outc and model.aux_out), we update them to have len(new_classes) outputs.
    For the classification branch (model.classification_branch.head) we update it only for lesion classes,
    that is, only classes whose name contains 'lesion'. Similarly, for the Gate module, we set:
        - class_list_seg = new_classes  (all segmentation classes)
        - class_list_cls = new_classes filtered to those containing 'lesion'
    
    Args:
        model (nn.Module): The pretrained model instance that has attributes outc, and possibly aux_out, classification_branch, gate_cls.
        original_classes (list of str): The original full list of class names (e.g., segmentation channels) used in the checkpoint.
        new_classes (list of str): The new full list of class names (e.g., segmentation channels).
        copy_pancreas (bool): If True, copy the weights of the pancreas class from the original model to all classes in the new model.
    Returns:
        model: The updated model.
    """
    # For classification, consider only classes with the word "lesion".
    new_class_cls = [cls for cls in new_classes if (("background" in cls) or ("lesion" in cls) or ('pdac' in cls) or ('pnet' in cls) or ('cyst' in cls))]
    old_class_cls = [cls for cls in original_classes if ("lesion" in cls)]

    # Helper: update a Conv3d layer given an old layer and a desired new number of output channels.
    def update_conv(old_conv, new_out_channels, full_class_list):
        in_channels = old_conv.in_channels
        new_conv = nn.Conv3d(
            in_channels,
            new_out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None)
        )
        # For each new class in full_class_list, if it exists in original_classes, copy the corresponding weight.
        for new_idx, new_cls in enumerate(full_class_list):
            if (new_cls not in original_classes) and copy_pancreas:
                # Copy the pancreas class weights to all new classes.
                orig_idx = original_classes.index('pancreatic_lesion')
                new_conv.weight.data[new_idx] = old_conv.weight.data[orig_idx].clone()
                if old_conv.bias is not None:
                    new_conv.bias.data[new_idx] = old_conv.bias.data[orig_idx].clone()
                print('Cloned the weights for class {} from index {} to new index {}'.format(new_cls, orig_idx, new_idx))
        
            if new_cls in original_classes:
                orig_idx = original_classes.index(new_cls)
                new_conv.weight.data[new_idx] = old_conv.weight.data[orig_idx].clone()
                if old_conv.bias is not None:
                    new_conv.bias.data[new_idx] = old_conv.bias.data[orig_idx].clone()
                print('Cloned the weights for class {} from index {} to new index {}'.format(new_cls, orig_idx, new_idx))
        return new_conv

    # Update model.outc (segmentation layer) using the full new_classes.
    old_outc = model.outc
    if old_outc.out_channels != len(new_classes):
        print("Updating model.outc from {} to {} outputs".format(old_outc.out_channels, len(new_classes)))
        model.outc = update_conv(old_outc, len(new_classes), new_classes)
    else:
        print("model.outc already has {} outputs.".format(len(new_classes)))

    # Update model.aux_out if present.
    if hasattr(model, 'aux_out') and model.aux_loss:
        old_aux = model.aux_out
        if old_aux.out_channels != len(new_classes):
            print("Updating model.aux_out from {} to {} outputs".format(old_aux.out_channels, len(new_classes)))
            model.aux_out = update_conv(old_aux, len(new_classes), new_classes)
        else:
            print("model.aux_out already has {} outputs.".format(len(new_classes)))
    
    # Update classification branch head.
    if hasattr(model, 'classification_branch') and (model.classification_branch is not None):
        old_head = model.classification_branch.head
        if old_head.out_features != len(new_class_cls):
            print("Updating classification branch head from {} to {} outputs".format(old_head.out_features, len(new_class_cls)))
            in_features = old_head.in_features
            new_head = nn.Linear(in_features, len(new_class_cls))
            # Copy weights for overlapping lesion classes.
            for new_idx, new_cls in enumerate(new_class_cls):
                if (new_cls not in original_classes) and copy_pancreas:
                    orig_idx = old_class_cls.index('pancreatic_lesion')
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
                if new_cls in old_class_cls:
                    orig_idx = old_class_cls.index(new_cls)
                    new_head.weight.data[new_idx] = old_head.weight.data[orig_idx].clone()
                    if old_head.bias is not None:
                        new_head.bias.data[new_idx] = old_head.bias.data[orig_idx].clone()
            model.classification_branch.head = new_head
        else:
            print("Classification branch head already has {} outputs.".format(len(new_class_cls)))
        assert model.classification_branch.head.out_features == len(new_class_cls)
    
    


    return model