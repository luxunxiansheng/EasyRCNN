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
from torchvision.ops import box_iou

from location_utility import LocationUtility

class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs."""
    def __init__(self,config):
        self.n_sample = config.RPN.PROPOSAL_TARGET_CREATOR.N_SAMPLES
        self.positive_ratio = config.RPN.PROPOSAL_TARGET_CREATOR.POSITIVE_RATIO
        self.positive_iou_thresh = config.RPN.PROPOSAL_TARGET_CREATOR.POSITIVE_IOU_THRESHOLD
        self.negative_iou_thresh_hi = config.RPN.PROPOSAL_TARGET_CREATOR.NEGATIVE_IOU_THRESHOLD_HI
        self.negative_iou_thresh_lo = config.RPN.PROPOSAL_TARGET_CREATOR.NEGATIVE_IOU_THRESHOLD_LO  
        self.loc_normalize_mean = torch.tensor(config.RPN.PROPOSAL_TARGET_CREATOR.OFFSET_NORM_MEAN)
        self.loc_normalize_std  = torch.tensor(config.RPN.PROPOSAL_TARGET_CREATOR.OFFSET_NORM_STD)

    def create(self, 
                    proposed_roi_bboxs: torch.Tensor,
                    gt_bboxs: torch.Tensor,
                    gt_labels: torch.Tensor,
                ):               
        """Assigns ground truth to sampled proposals.
            Args:
                proposed_roi_bboxs: (n_proposals, 4) tensor.
                gt_bboxs: (n_gt, 4) tensor.
                gt_labels: (n_gt,) tensor.

            Returns:
                sampled_proposals: (n_sampled, 4) tensor.
        """                
        n_positive_roi = int(self.n_sample * self.positive_ratio)

        proposed_roi_bboxs = torch.concat((proposed_roi_bboxs, gt_bboxs),dim=0)

        ious = box_iou(proposed_roi_bboxs[:,[1,0,3,2]], gt_bboxs[:,[1,0,3,2]])
        gt_max_ious_for_proposed_roi_bboxs, gt_index_with_max_ious_for_proposed_roi_bboxs= ious.max(dim=1)
        
        # for each proposed roi bbox, assign the label as the gt label with max iou
        gt_roi_label = gt_labels[gt_index_with_max_ious_for_proposed_roi_bboxs] 

        # select proposed roi bbox as positive with >= pos_iou_thresh IoU.
        positive_index = torch.where(gt_max_ious_for_proposed_roi_bboxs >= self.positive_iou_thresh)[0]
        n_positive_roi = int(min(n_positive_roi, len(positive_index)))
        if len(positive_index) > 1:
            selected_positive_index = torch.multinomial(positive_index.float(),num_samples=n_positive_roi, replacement=False)
            positive_index = positive_index[selected_positive_index]        

        # Select background RoIs as those within [neg_iou_thresh_lo, neg_iou_thresh_hi).
        negative_index = torch.where((gt_max_ious_for_proposed_roi_bboxs < self.negative_iou_thresh_hi) &(gt_max_ious_for_proposed_roi_bboxs >= self.negative_iou_thresh_lo))[0]
        n_negative_rois = self.n_sample - n_positive_roi
        n_negative_rois = int(min(n_negative_rois,len(negative_index)))
        if len(negative_index) > 1:
            selected_negative_index = torch.multinomial(negative_index.float(), num_samples=n_negative_rois, replacement=False)
            negative_index = negative_index[selected_negative_index]
        
        # The indices that we're selecting (both positive and negative).
        keep_index = torch.concat((positive_index,negative_index),dim=0)
        gt_label_for_sampled_roi = gt_roi_label[keep_index]
        gt_label_for_sampled_roi[n_positive_roi:] = 0  # negative labels --> 0
        sampled_roi = proposed_roi_bboxs[keep_index]
        gt_bboxes_for_sampled_roi = gt_bboxs[gt_index_with_max_ious_for_proposed_roi_bboxs[keep_index]]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_offsets_for_sampled_roi = LocationUtility.bbox2offset(sampled_roi, gt_bboxes_for_sampled_roi)
        gt_offsets_for_sampled_roi = (gt_offsets_for_sampled_roi - self.loc_normalize_mean.to(gt_offsets_for_sampled_roi.device)) / self.loc_normalize_std.to(gt_offsets_for_sampled_roi.device)
        return sampled_roi,gt_label_for_sampled_roi,gt_offsets_for_sampled_roi


