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
from torch.types import Device
from yacs.config import CfgNode


class RFCNLoss(nn.Module):
    """ 
    Detection Losses : classfication loss and regression loss  
    """
    def __init__(self,config:CfgNode,device:Device='cpu'):
        super().__init__()
        self.roi_sigma = config.R_FCN.ROI_SIGMMA
        self.device = device
    
    def forward(self,
                predicted_scores:torch.Tensor,
                predicted_offsets:torch.Tensor,
                target_labels:torch.Tensor,
                target_offsets:torch.Tensor):
        """
        Args:
            predicted_scores: (B, N, C)
            predicted_offsets: (B, N, 4)
            target_labels: (B, N)
            target_offsets: (B, N, 4)

        Returns:
            classification_loss: (B,), regression_loss: (B,)
        """

        classfication_loss = F.cross_entropy(predicted_scores,target_labels)

        n_sample = target_offsets.shape[0]

        positive_wieight = torch.zeros(target_offsets.shape).to(self.device)
        positive_wieight[(target_labels > 0).view(-1,1).expand_as(positive_wieight)] = 1

        predicted_offsets = predicted_offsets.contiguous().view(n_sample,-1,4)
        predicted_offsets = predicted_offsets[torch.arange(0,n_sample).long(),target_labels.long()]
    
        predicted_offsets = positive_wieight * predicted_offsets
        target_offsets    = positive_wieight * target_offsets

        regression_loss = self._soomth_l1_loss(predicted_offsets,target_offsets,self.roi_sigma)
        regression_loss = regression_loss /((target_labels>0).sum().float())

        return classfication_loss,regression_loss

    def compute(self,
                predicted_scores:torch.Tensor,
                predicted_offsets:torch.Tensor,
                target_labels:torch.Tensor,
                target_offsets:torch.Tensor):
        """
            A explicit function to compute the loss by calling forward function
        """
        
        return self.forward(predicted_scores,predicted_offsets,target_labels,target_offsets)
    
    def _soomth_l1_loss(self,
                        predicted_offsets:torch.Tensor, 
                        target_offsets:torch.Tensor,
                        sigma:float):
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