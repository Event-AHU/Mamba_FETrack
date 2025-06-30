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


class Mamba_FEtrack(nn.Module):
    """ This is the base class for mamda_fetrack """

    def __init__(self, visionmamba, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = visionmamba
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
                ):
      
        x = self.backbone(rgb_z=template, rgb_x=search,  event_z=event_template, event_x=event_search,     #[B, 640, 384], 
                        inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False)

        rgb_feat, event_feat = torch.chunk(x, 2, dim=1)
        
        # Forward head
        out = self.forward_head(rgb_feat, event_feat, None)
        out['backbone_feat'] = x
        
        return out

    def forward_head(self, rgb_feat, event_feat, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt1 = rgb_feat[:, -self.feat_len_s:]
        enc_opt2 = event_feat[:, -self.feat_len_s:]
        enc_opt = enc_opt1 + enc_opt2

        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

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
    
    backbone = create_model(model_name=cfg.MODEL.BACKBONE.TYPE, pretrained=pretrained, num_classes=1000,
            drop_rate=0.0, drop_path_rate=0.05, drop_block_rate=None, img_size=256
            )
    
    hidden_dim = 384
    box_head = build_box_head(cfg, hidden_dim)
    
    model = Mamba_FEtrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )
   
    if 'Mamba_FETrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

   
