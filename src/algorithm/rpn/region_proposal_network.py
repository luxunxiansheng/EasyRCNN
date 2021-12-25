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
from yacs.config import CfgNode
from common import CNNBlock, weights_normal_init


class RPN(nn.Module):
    def __init__(self,config:CfgNode):
        super().__init__()
    
        self.num_base_anchors = len(config.RPN.ANCHOR_CREATOR.ANCHOR_SCALES)**2
        self.conv1 = CNNBlock(config.RPN.FEATURE_CHANNELS, config.RPN.MID_CHANNELS, 3, same_padding=True)
        self.score_conv = CNNBlock(config.RPN.MID_CHANNELS, self.num_base_anchors*2, 1, relu=False, same_padding=False)
        self.bbox_conv = CNNBlock(config.RPN.MID_CHANNELS,  self.num_base_anchors*4, 1, relu=False, same_padding=False)

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
