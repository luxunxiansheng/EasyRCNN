# #### BEGIN LICENSE BLOCK #####
# MIT License
#    
# Copyright (c) 2021 Bin.Li (ornot2008@yahoo.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# #### END LICENSE BLOCK #####
# /

import torch
from torch import nn
from torch.nn import functional as F
from torch.types import Device
from torchvision.ops import PSRoIPool
from yacs.config import CfgNode

from common import CNNBlock
from feature_extractor import FeatureExtractorFactory
from location_utility import LocationUtility
from rpn.anchor_creator import AnchorCreator
from rpn.proposal_creator import ProposalCreator
from rpn.region_proposal_network import RPN

class PositionSensitiveNetwork(nn.module):

    def __init__(self,
                config:CfgNode,
                device:Device='cpu') -> None:
        super().__init__()
        self.config = config
        self.device = device
        
        self.double_channel_conv = CNNBlock(config.R_FCN.IN_CHANNELS,2*config.R_FCN.IN_CHANNELS,1)
    
        self.score_map_conv =CNNBlock(2*config.R_FCN.IN_CHANNELS,config.R_FCN.POOL_SIZE**2*(config.R_FCN.NUM_CLASSES+1),1)
        self.ps_roi_pool_class = PSRoIPool(output_size=config.R_FCN.POOL_SIZE,
                                    spatial_scale=1.0/config.R_FCN.FEATURE_STRIDE)
        self.class_avg_pool = nn.AvgPool2d(kernel_size=config.R_FCN.POOL_SIZE,stride=config.R_FCN.POOL_SIZE)

        self.bbox_map_conv = CNNBlock(2*config.R_FCN.IN_CHANNELS,config.R_FCN.POOL_SIZE**2*4,1)
        self.ps_roi_pool_bbox = PSRoIPool(output_size=config.R_FCN.POOL_SIZE,spatial_scale=1.0/config.R_FCN.FEATURE_STRIDE)
        self.bbox_avg_pool = nn.AvgPool2d(kernel_size=config.R_FCN.POOL_SIZE,stride=config.R_FCN.POOL_SIZE)

        
    def predict(self,
                feature:torch.Tensor,
                proposed_roi_bboxes:torch.Tensor):
        return self.forward(feature=feature,proposed_roi_bboxes=proposed_roi_bboxes)

    def forward(self, 
                feature:torch.Tensor, 
                proposed_roi_bboxes:torch.Tensor,
                ):
        """
        Args:
            feature (torch.Tensor): [C,H,W]
            proposed_roi_bboxes (torch.Tensor): [n_bboxes,4]
        
        Returns:
            bboxes (torch.Tensor): [n_bboxes,4]
            labels (torch.Tensor): [n_bboxes,]
            scores (torch.Tensor): [n_bboxes,]
        """
        double_channel_feature = self.double_channel_conv(feature)
        
        #* 1. compute the probability of the class scores
        position_sensitive_score_maps = self.score_map_conv(double_channel_feature)
        class_vote_array = self.ps_roi_pool_class(position_sensitive_score_maps, proposed_roi_bboxes)
        predicted_roi_score = self.class_avg_pool(class_vote_array)
        
        #* 2. compute the bbox offsets
        position_sensitive_bbox_maps = self.bbox_map_conv(double_channel_feature)
        bbox_vote_array = self.ps_roi_pool_bbox(position_sensitive_bbox_maps, proposed_roi_bboxes)
        predicted_roi_offset = self.bbox_avg_pool(bbox_vote_array)

        return predicted_roi_score,predicted_roi_offset, 

