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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from torch.types import Device
from yacs.config import CfgNode

from rpn.anchor_target_creator import AnchorTargetCreator

class RPNLoss(nn.Module):
    """calculate the loss for a single image"""
    def __init__(self,config:CfgNode,device:Device='cpu'):
        super().__init__()
        self.sigma = config.RPN.RPN_SIGMA
        self.device = device

        self.anchor_target_creator = AnchorTargetCreator(config,device=device)

    def forward(self,
                anchors_of_img: torch.Tensor,
                predicted_scores: torch.Tensor,
                predicted_offsets: torch.Tensor,
                target_bboxs: torch.Tensor,
                img_height: int,
                img_width: int):
        """
                Compute the loss for a single image.

                Args:
                    anchors_of_img: (N, 4) tensor. 
                    predicted_scores: (N, 1) tensor.
                    predicted_offsets: (N, 4) tensor.
                    target_bboxs: (M, 4) tensor.
                    img_height: int.
                    img_width: int.
                
                Returns:
                    classification_loss: float.
                    regression_loss: float.

        """
        target_labels,target_offsets = self.anchor_target_creator.create(anchors_of_img,target_bboxs,img_height,img_width)

        if target_labels is None:
            return torch.tensor(0.0,device=self.device),torch.tensor(0.0,device=self.device)

        # we only concern those anchors which have positive labels and negative labels
        target_keep =  target_labels.ne(-1).nonzero().squeeze()

        #----------------------- classfication loss -----------------------#
        predicted_scores = predicted_scores.permute(1,2,0).contiguous().view(-1,2)
        predicted_scores_keep = torch.index_select(predicted_scores,0,target_keep)
        target_labels_keep = torch.index_select(target_labels,0,target_keep) 
        classification_loss = F.cross_entropy(predicted_scores_keep,target_labels_keep.long(),ignore_index=-1)

        #----------------------- regression loss --------------------------#
        inside_weight = torch.zeros(target_offsets.shape,device=self.device)

        # get the positive locs predicted by the network
        inside_weight[(target_labels > 0).view(-1,1).expand_as(inside_weight)] = 1
        predicted_offsets = predicted_offsets.permute(1,2,0).contiguous().view(-1,4)

        predicted_offsets = inside_weight * predicted_offsets
        target_offsets = inside_weight * target_offsets

        # loc_loss is the sum of the smooth L1 loss of the four coordinates of the positive anchors
        loc_loss = self._soomth_l1_loss(predicted_offsets,target_offsets,self.sigma)

        # Normalize by the number of the  positives
        regression_loss = loc_loss/((target_labels>=0).sum().float())  

        return classification_loss,regression_loss

    def compute(self,anchors_of_img,predicted_scores,predicted_offsets,target_bboxs,img_height,img_width):
        """
            A explict interface for computing the loss by calling the forward function.
        """
        return self.forward(anchors_of_img,predicted_scores,predicted_offsets,target_bboxs,img_height,img_width)
    
    def _soomth_l1_loss(self, predicted_offsets, target_offsets,sigma):
        """
        calculate smooth L1 loss 

        Args:
            predicted_offsets: (B, N, 4)
            target_offsets: (B, N, 4)
            sigma: float
        
        Returns:
            loss: (B,)
        """

        sigma2 = sigma**2
        diff = predicted_offsets - target_offsets
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1.0 / sigma2)).float()
        loss = flag * (sigma2 / 2.) * (diff ** 2) +(1 - flag) * (abs_diff - 0.5 / sigma2)
        return loss.sum() 

