"""
Basic mamda_fetrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.mamba_fetrack.models_mamba import create_block
from timm.models import create_model
import torch.nn.functional as F
from lib.models.mamba_fetrack.mamba_cross import CrossMamba
from thop import profile


class Mamba_FEtrack(nn.Module):
    """ This is the base class for mamda_fetrack """

    def __init__(self, visionmamba, cross_mamba, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = visionmamba
        self.cross_mamba= cross_mamba
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                event_template: torch.Tensor,       
                event_search: torch.Tensor,          
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
      
        rgb_feature = self.backbone.forward_features( z=template, x=search,                                                                     #[B, 320, 384]
                                                inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False)
        event_feature = self.backbone.forward_features(z=event_template, x=event_search,                                                        #[B, 320, 384]
                                                inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False)

        residual_event_f = 0
        residual_rgb_f = 0
        event_f = self.cross_mamba(event_feature,residual_event_f,rgb_feature) + event_feature
        rgb_f = self.cross_mamba(rgb_feature,residual_rgb_f,event_feature) + rgb_feature
        
        event_searh = event_f[:, -self.feat_len_s:]
        rgb_search = rgb_f[:, -self.feat_len_s:]
        x = torch.cat((event_searh,rgb_search),dim=-1)
        
        
        # Forward head
        feat_last = x             
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)
       
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        search_feature = cat_feature
        opt = (search_feature.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()           # opt.shape = torch.Size([B, 1, 384, 256])
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)                       # opt_feat.shape = torch.Size([B, 384, 16, 16])

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
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_mamba_fetrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('Mamba_FETrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
    
    backbone = create_model( model_name= cfg.MODEL.BACKBONE.TYPE, pretrained= pretrained, num_classes=1000,
            drop_rate=0.0, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, drop_block_rate=None, img_size=256
            )
    hidden_dim = 384
    cross_mamba = CrossMamba(hidden_dim)
    box_head = build_box_head(cfg, hidden_dim*2)
    model = Mamba_FEtrack(
        backbone,
        cross_mamba,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )
   
    if 'Mamba_FETrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

   
