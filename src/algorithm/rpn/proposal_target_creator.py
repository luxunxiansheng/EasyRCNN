import torch
from torchvision.ops import nms
from torchvision.ops import box_iou

from utility import Utility

class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs."""
    def __init__(self,config):
        self.n_sample = config.RPN.PROPOSAL_TARGET_CREATOR.N_SAMPLES
        self.positive_ratio = config.RPN.PROPOSAL_TARGET_CREATOR.POSITIVE_RATIO
        self.positive_iou_thresh = config.RPN.PROPOSAL_TARGET_CREATOR.POSITIVE_IOU_THRESHOLD
        self.negative_iou_thresh_hi = config.RPN.PROPOSAL_TARGET_CREATOR.NEGATIVE_IOU_THRESHOLD_HI
        self.negative_iou_thresh_lo = config.RPN.PROPOSAL_TARGET_CREATOR.NEGATIVE_IOU_THRESHOLD_LO  # NOTE:default 0.1 in py-faster-rcnn
        

    def create(self, 
                    proposed_roi_bboxs, 
                    gt_bboxs, 
                    gt_labels,
                    
                ):
        """Assigns ground truth to sampled proposals."""
        
        
        n_positive_roi_per_image = int(self.n_sample * self.positive_ratio)

        proposed_roi_bboxs = torch.concat((proposed_roi_bboxs, gt_bboxs),dim=0)

        ious = box_iou(proposed_roi_bboxs, gt_bboxs)
        max_ious_for_proposed_roi_bboxs, argmax_ious_for_proposed_roi_bboxs= ious.max(dim=1)
        
        gt_roi_label = gt_labels[argmax_ious_for_proposed_roi_bboxs] 

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        positive_index = torch.where(max_ious_for_proposed_roi_bboxs >= self.positive_iou_thresh)[0]
        n_positive_roi_per_image = int(min(n_positive_roi_per_image, len(positive_index)))
        if len(positive_index) > 0:
            selected_positive_index = torch.multinomial(positive_index.float(),num_samples=n_positive_roi_per_image, replacement=False)
            positive_index = positive_index[selected_positive_index]

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        negative_index = torch.where((max_ious_for_proposed_roi_bboxs < self.negative_iou_thresh_hi) &(max_ious_for_proposed_roi_bboxs >= self.negative_iou_thresh_lo))[0]
        n_negative_rois_per_image = self.n_sample - n_positive_roi_per_image
        n_negative_rois_per_image = int(min(n_negative_rois_per_image,len(negative_index)))
        if len(negative_index) > 0:
            selected_negative_index = torch.multinomial(negative_index.float(), num_samples=n_negative_rois_per_image, replacement=False)
            negative_index = negative_index[selected_negative_index]

        # The indices that we're selecting (both positive and negative).
        keep_index = torch.concat((positive_index,negative_index),dim=0)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[n_positive_roi_per_image:] = 0  # negative labels --> 0
        sampled_roi = proposed_roi_bboxs[keep_index]
        gt_bboxes = gt_bboxs[argmax_ious_for_proposed_roi_bboxs[keep_index]]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = Utility.bbox2loc(sampled_roi, gt_bboxes)
        return sampled_roi,gt_roi_label,gt_roi_loc


