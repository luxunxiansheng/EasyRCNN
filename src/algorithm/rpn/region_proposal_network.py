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
from yacs.config import CfgNode
from common import CNNBlock, weights_normal_init


class RPN(nn.Module):
    def __init__(self,config:CfgNode):
        super().__init__()
    
        self.num_base_anchors = len(config.RPN.ANCHOR_CREATOR.ANCHOR_SCALES)**2
        self.conv1 = CNNBlock(config.RPN.FEATURE_CHANNELS, config.RPN.MID_CHANNELS, 3,  relu=False,bn=False, same_padding=True)
        self.score_conv = CNNBlock(config.RPN.MID_CHANNELS, self.num_base_anchors*2, 1, relu=False,bn=False)
        self.bbox_conv = CNNBlock(config.RPN.MID_CHANNELS,  self.num_base_anchors*4, 1, relu=False,bn=False)

        weights_normal_init(self.conv1, dev=0.01)
        weights_normal_init(self.score_conv, dev=0.01)
        weights_normal_init(self.bbox_conv, dev=0.001)

    def forward(self,features:torch.Tensor):
        """
        Args:
            features: (N, C, H, W)

        Returns:
            scores: (N, num_base_anchors*2, H, W)
            bboxs: (N, num_base_anchors*4, H, W)
        """
        # [batch_size, middle_channels, feature_height, feature_width]
        hidden = self.conv1(features)  

        #-------------------- predict cls score ----------------------#
        #[batch_size, num_base_anchors * 2, feature_height,feature_width]
        predicted_scores = self.score_conv(hidden) 
        
        #-------------------- predict locs ----------------------#
        # [batch_size, num_base_anchors * 4, feature_height,feature_width]
        predicted_offsets = self.bbox_conv(hidden) 

        return predicted_scores,predicted_offsets

    def predict(self,features):
        return self.forward(features)
