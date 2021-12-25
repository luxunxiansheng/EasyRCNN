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

        ious = box_iou(proposed_roi_bboxs, gt_bboxs)
        max_ious_for_proposed_roi_bboxs, argmax_ious_for_proposed_roi_bboxs= ious.max(dim=1)
        
        gt_roi_label = gt_labels[argmax_ious_for_proposed_roi_bboxs] 

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        positive_index = torch.where(max_ious_for_proposed_roi_bboxs >= self.positive_iou_thresh)[0]
        n_positive_roi = int(min(n_positive_roi, len(positive_index)))
        if len(positive_index) > 1:
            selected_positive_index = torch.multinomial(positive_index.float(),num_samples=n_positive_roi, replacement=False)
            positive_index = positive_index[selected_positive_index]        

        # Select background RoIs as those within [neg_iou_thresh_lo, neg_iou_thresh_hi).
        negative_index = torch.where((max_ious_for_proposed_roi_bboxs < self.negative_iou_thresh_hi) &(max_ious_for_proposed_roi_bboxs >= self.negative_iou_thresh_lo))[0]
        n_negative_rois = self.n_sample - n_positive_roi
        n_negative_rois = int(min(n_negative_rois,len(negative_index)))
        if len(negative_index) > 1:
            selected_negative_index = torch.multinomial(negative_index.float(), num_samples=n_negative_rois, replacement=False)
            negative_index = negative_index[selected_negative_index]
        
        # The indices that we're selecting (both positive and negative).
        keep_index = torch.concat((positive_index,negative_index),dim=0)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[n_positive_roi:] = 0  # negative labels --> 0
        sampled_roi = proposed_roi_bboxs[keep_index]
        gt_bboxes = gt_bboxs[argmax_ious_for_proposed_roi_bboxs[keep_index]]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_offsets = LocationUtility.bbox2offset(sampled_roi, gt_bboxes)
        gt_roi_offsets = (gt_roi_offsets - self.loc_normalize_mean.to(gt_roi_offsets.device)) / self.loc_normalize_std.to(gt_roi_offsets.device)
        return sampled_roi,gt_roi_label,gt_roi_offsets


