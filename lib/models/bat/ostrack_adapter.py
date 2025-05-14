"""
Basic BAT model.
"""
import math
import os
from typing import List
from timm.models.layers import to_2tuple
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
#from lib.models.bat.vit_adapter import vit_base_patch16_224_adapter
from lib.models.bat.vit_ce_adapter import vit_base_patch16_224_ce_adapter
from lib.utils.box_ops import box_xyxy_to_cxcywh
import torch.nn.functional as F

class BATrack(nn.Module):
    """ This is the base class for BATrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.re_box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                img_blocks = None,
                ts_images = None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                Test=None,
                localization=None,
                second_phase=None, #是否启用第二阶段
                ):
        
        if second_phase:
            
            xcorr,loss = self.backbone(z=template, x=search,img_blocks=img_blocks,
                                        ts_images = ts_images,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn,Test=Test,localization=localization,second_phase=second_phase)
            out_up = self.re_forward_head(xcorr, None)
            # out_up = self.forward_head(xcorr, None)
            # out_up.update(aux_dict)
            out_up['ccl'] = loss            
            return out_up

            
        else:
            x, aux_dict = self.backbone(z=template, x=search,img_blocks=img_blocks,
                                                ts_images = ts_images,
                                                ce_template_mask=ce_template_mask,
                                                ce_keep_rate=ce_keep_rate,
                                                return_last_attn=return_last_attn,Test=Test,localization=localization,second_phase=second_phase)
        # Forward head
        if Test:
            if type(x) is tuple:
                x_up,x_down = x 
                x = x_up + x_down
                out_up = self.forward_head(x, None)
                out_up.update(aux_dict)
                out_up['backbone_feat'] = x            
                return out_up
            else:
                feat_last = x
                if isinstance(x, list):
                    feat_last = x[-1]
                out = self.forward_head(feat_last, None)
                out.update(aux_dict)
                out['backbone_feat'] = x
                return out
        else:


            out = self.forward_head(x, None)
            out.update(aux_dict)
            out['backbone_feat'] = x        
            return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        #print("cat_feature",cat_feature.shape)
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        #print("opt_feat", opt_feat.shape)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            #print("outputs_coord", outputs_coord.shape)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def re_forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        #print("cat_feature",cat_feature.shape)
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        #print("opt_feat", opt_feat.shape)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.re_box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.re_box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            #print("outputs_coord", outputs_coord.shape)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError



def build_batrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')  # use pretrained OSTrack as initialization
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE and 'DropTrack' not in cfg.MODEL.PRETRAIN_FILE and 'best' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_adapter':
        backbone = vit_base_patch16_224_adapter(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                               search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                               template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                               new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                               adapter_type=cfg.TRAIN.PROMPT.TYPE
                                               )
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_adapter':
        backbone = vit_base_patch16_224_ce_adapter(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                           template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                           new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                           adapter_type=cfg.TRAIN.PROMPT.TYPE
                                           )
        hidden_dim = backbone.embed_dim

    else:
        raise NotImplementedError
    """For adapter no need, because we have OSTrack as initialization"""
    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)
    # re_box_head = build_box_head(cfg, hidden_dim)

    model = BATrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if training and ('OSTrack' in cfg.MODEL.PRETRAIN_FILE or 'DropTrack' in cfg.MODEL.PRETRAIN_FILE or 'best' in cfg.MODEL.PRETRAIN_FILE):
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        param_dict_rgbt = dict()
        if 'DropTrack' in cfg.MODEL.PRETRAIN_FILE:
            for k,v in checkpoint["net"].items():
                if k in ['box_head.conv1_ctr.0.weight','box_head.conv1_offset.0.weight','box_head.conv1_size.0.weight']:
                    # v = torch.cat([v,v],1)
                    v = v
                elif 'pos_embed_x' in k:
                    v = resize_pos_embed(v, 16, 16) + checkpoint["net"]['backbone.temporal_pos_embed_x']
                elif 'pos_embed_z' in k:
                    v = resize_pos_embed(v, 8, 8) + checkpoint["net"]['backbone.temporal_pos_embed_z']
                else:
                    v = v
                param_dict_rgbt[k] = v
            missing_keys, unexpected_keys = model.load_state_dict(param_dict_rgbt, strict=False)

        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)


    return model

def resize_pos_embed(posemb, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_grid = posemb[0, :]
    
    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to new token with height:{} width: {}'.format(posemb_grid.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    # posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb_grid