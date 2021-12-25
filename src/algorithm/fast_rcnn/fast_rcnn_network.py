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
from torchvision.ops import RoIPool

from common import FCBlock, weights_normal_init

class FastRCNN(nn.Module):
    def __init__(self, config,device='cpu'):
        super().__init__()
    
        self.n_classes = config.FAST_RCNN.NUM_CLASSES

        self.roi_pool = RoIPool((config.FAST_RCNN.ROI_SIZE,config.FAST_RCNN.ROI_SIZE),config.FAST_RCNN.SPATIAL_SCALE)

        self.fc6 = FCBlock(config.FAST_RCNN.IN_CHANNELS*config.FAST_RCNN.ROI_SIZE*config.FAST_RCNN.ROI_SIZE,config.FAST_RCNN.FC7_CHANNELS)
        self.fc7 = FCBlock(config.FAST_RCNN.FC7_CHANNELS, config.FAST_RCNN.FC7_CHANNELS)
        
        self.offset = nn.Linear(config.FAST_RCNN.FC7_CHANNELS,  (self.n_classes+1) * 4)
        self.score = nn.Linear(config.FAST_RCNN.FC7_CHANNELS,self.n_classes+1)

        weights_normal_init(self.offset, 0.001)
        weights_normal_init(self.score,0.01)

    def forward(self,feature,rois):
        roi_indices = torch.zeros(len(rois),device=feature.device)
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = indices_and_rois.contiguous()

        pool = self.roi_pool(feature.unsqueeze(0) , indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc6 = self.fc6(pool)
        fc7 = self.fc7(fc6)
        roi_scores = self.score(fc7)
        roi_offsets = self.offset(fc7)

        return roi_scores,roi_offsets 
    
    def predict(self,feature,rois):
        return  self.forward(feature,rois)
    