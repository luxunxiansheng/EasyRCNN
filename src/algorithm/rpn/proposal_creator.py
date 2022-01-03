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
            Generate proposals from anchors with predicted scores and offsets.

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
        #
        #  1. get all of the bboxes based on the anchors and predicted offsets
        #

        # [feature_height,feature_width, num_base_anchors * 4]
        predicted_offsets = predicted_offsets.permute(1,2,0).contiguous()
        
        # [Num_anchors,4]
        predicted_offsets = predicted_offsets.view(-1,4) 

        # Convert anchors into proposals via bbox transformations.
        predicted_roi_bboxs = LocationUtility.offset2bbox(anchors_of_image, predicted_offsets)

        #
        # 2. Clip predicted boxes to image.
        #
        predicted_roi_bboxs[:, slice(0, 4, 2)] = torch.clip(predicted_roi_bboxs[:, slice(0, 4, 2)], 0, img_height)
        predicted_roi_bboxs[:, slice(1, 4, 2)] = torch.clip(predicted_roi_bboxs[:, slice(1, 4, 2)], 0, img_width)

        # 
        # 3. Remove those predicted boxes with either height or width > threshold.
        #
        min_size = self.min_size * scale
        hs = predicted_roi_bboxs[:, 2] - predicted_roi_bboxs[:, 0]
        ws = predicted_roi_bboxs[:, 3] - predicted_roi_bboxs[:, 1]
        index_to_keep_with_specified_size = torch.where((hs >= min_size) & (ws >= min_size))[0]
        predicted_roi_bboxs = predicted_roi_bboxs[index_to_keep_with_specified_size, :]

        #
        #  4. Take n_pre_nms top objectness score  proposals before NMS.
        #

        # [feature_height,feature_width, num_base_anchors * 2]
        predicted_scores = predicted_scores.permute(1, 2, 0).contiguous() 
        
        # [feature_height,feature_width, Num_base_anchors,2]
        predicted_scores = predicted_scores.view(feature_height,feature_width,-1,2) 
        
        predicted_softmax_scores = torch.softmax(predicted_scores,dim=3)

        #[Num_anchors]
        predicted_objectness_scores= predicted_softmax_scores[:,:,:,1].contiguous().view(-1)
        proposed_objectness_scores = predicted_objectness_scores[index_to_keep_with_specified_size]

        # Sort all proposed_objectness_scores by score from highest to lowest.
        proposed_objectness_scores,order = proposed_objectness_scores.sort(descending=True)

        # Take top pre_nms_topN (e.g. 6000) boxes before NMS.
        if self.n_pre_nms > 0:
            index_top_pre_nms = order[:self.n_pre_nms]
        predicted_roi_bboxs = predicted_roi_bboxs[index_top_pre_nms,:]
        proposed_objectness_scores = proposed_objectness_scores[index_top_pre_nms]

        #
        #  5. Run NMS on the top proposals.
        #

        # Apply nms (e.g. threshold = 0.7)
        keep = nms(predicted_roi_bboxs[:,[1,0,3,2]],proposed_objectness_scores,
                    self.nms_thresh)
        
        #
        # 6. Take post_nms_topN (e.g. 300) bboxes after NMS.
        #      
        if self.n_post_nms > 0:
            index_top_after_nms = keep[:self.n_post_nms]

        return predicted_roi_bboxs[index_top_after_nms]

