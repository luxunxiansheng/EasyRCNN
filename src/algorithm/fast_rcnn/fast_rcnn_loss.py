# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
from yacs.config import CfgNode


class FastRCNNLoss(nn.Module):
    """ 
    Detection Losses : classfication loss and regression loss  
    """
    def __init__(self,config:CfgNode,device:Device='cpu'):
        super().__init__()
        self.roi_sigma = config.FAST_RCNN.ROI_SIGMMA
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