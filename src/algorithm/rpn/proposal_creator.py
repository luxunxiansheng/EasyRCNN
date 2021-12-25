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
from torchvision.ops import nms
from location_utility import LocationUtility

class ProposalCreator:
    def __init__(self,config):
        self.n_pre_nms = config.RPN.PROPOSAL_CREATOR.N_PRE_NMS
        self.n_post_nms= config.RPN.PROPOSAL_CREATOR.N_POST_NMS
                
        self.nms_thresh = config.RPN.PROPOSAL_CREATOR.NMS_THRESHOLD
        self.min_size =   config.RPN.PROPOSAL_CREATOR.MIN_SIZE

    def create(self, 
                anchors_of_image: torch.Tensor,  
                predicted_scores: torch.Tensor,  
                predicted_offsets: torch.Tensor,
                img_height: int,
                img_width: int,
                feature_height: int,
                feature_width: int,
                scale:float=1.):
        """
            Generate proposals from anchors and predicted scores and offsets.

            Args:
                anchors_of_image: (N, 4) tensor.
                predicted_scores: (N, 1) tensor.
                predicted_offsets: (N, 4) tensor.
                img_height: int.
                img_width: int.
                feature_height: int.
                feature_width: int.
                scale: float.
            
            Returns:
                proposals: (n_proposals, 4) tensor.
        """
                
        #------------------------Locs---------------------------------#
        # [feature_height,feature_width, num_base_anchors * 4]
        predicted_offsets = predicted_offsets.permute(1,2,0).contiguous()
        
        # [Num_anchors,4]
        predicted_offsets = predicted_offsets.view(-1,4) 

        #------------------------Scores---------------------------------#
        # [feature_height,feature_width, num_base_anchors * 2]
        predicted_scores = predicted_scores.permute(1, 2, 0).contiguous() 
        
        # [feature_height,feature_width, Num_base_anchors,2]
        predicted_scores = predicted_scores.view(feature_height,feature_width,-1,2) 

        #------------------------Objectness_scores---------------------------------#
        #[Num_anchors]
        predicted_objectness_scores= predicted_scores[:,:,:,1].contiguous().view(-1)


        #------------------------Proposed ROI bboxs---------------------------------#
        # Convert anchors into proposal via bbox transformations.
        proposed_roi_bboxs = LocationUtility.offset2bbox(anchors_of_image, predicted_offsets)
        
        # Clip predicted boxes to image.
        proposed_roi_bboxs[:, slice(0, 4, 2)] = torch.clip(proposed_roi_bboxs[:, slice(0, 4, 2)], 0, img_height)
        proposed_roi_bboxs[:, slice(1, 4, 2)] = torch.clip(proposed_roi_bboxs[:, slice(1, 4, 2)], 0, img_width)

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale
        hs = proposed_roi_bboxs[:, 2] - proposed_roi_bboxs[:, 0]
        ws = proposed_roi_bboxs[:, 3] - proposed_roi_bboxs[:, 1]
        keep = torch.where((hs >= min_size) & (ws >= min_size))[0]

        proposed_roi_bboxs = proposed_roi_bboxs[keep, :]
        proposed_objectness_scores = predicted_objectness_scores[keep]

        
        # Sort all (proposal, score) pairs by score from highest to lowest.
        order = proposed_objectness_scores.argsort(descending=True)

        # Take top pre_nms_topN (e.g. 6000).
        if self.n_pre_nms > 0:
            order = order[:self.n_pre_nms]
        proposed_roi_bboxs = proposed_roi_bboxs[order,:]
        proposed_objectness_scores = proposed_objectness_scores[order]

        # Apply nms (e.g. threshold = 0.7)
        proposed_roi_bboxs_xyxy=proposed_roi_bboxs.index_select(dim=1,
                                                                index=torch.tensor([1,0,3,2],
                                                                device=proposed_roi_bboxs.device))

        keep = nms(proposed_roi_bboxs_xyxy,proposed_objectness_scores,
                    self.nms_thresh)
        
        # Take after_nms_topN (e.g. 300)
        if self.n_post_nms > 0:
            keep = keep[:self.n_post_nms]

        return proposed_roi_bboxs[keep,:]

