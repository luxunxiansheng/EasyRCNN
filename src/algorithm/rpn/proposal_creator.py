import sys
sys.path.append('..')

import torch
from torchvision.ops import nms
from utility import Utility

class ProposalCreator:
    def __init__(self,config,n_pre_nms=12000,n_post_nms=2000):
        self.n_pre_nms = n_pre_nms
        self.n_post_nms = n_post_nms
                
        self.nms_thresh = config.RPN.PROPOSAL_CREATOR.NMS_THRESHOLD
        self.min_size =   config.RPN.PROPOSAL_CREATOR.MIN_SIZE

    
    
    # Note we generate proposal for each image independently
    def create(self, 
                anchors_of_image,  # [Num_anchors]
                predicted_scores,  # [Num_base_anchors*2,feature_h,feature_w]
                predicted_locs,    # [Num_base_anchors*4,feature_h,feature_w]
                img_height,
                img_width,
                feature_height,
                feature_width,
                scale=1.):
        
                
        #------------------------Locs---------------------------------#
        # [feature_height,feature_width, num_base_anchors * 4]
        predicted_locs = predicted_locs.permute(1,2,0).contiguous()
        
        # [Num_anchors,4]
        predicted_locs = predicted_locs.view(-1,4) 

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
        proposed_roi_bboxs = Utility.loc2bbox(anchors_of_image, predicted_locs)
        
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

        # Apply nms (e.g. threshold = 0.7).
        
        
        
        proposed_roi_bboxs_xyxy=proposed_roi_bboxs.index_select(dim=1,
                                                                index=torch.tensor([1,0,3,2],
                                                                device=proposed_roi_bboxs.device))

        keep = nms(proposed_roi_bboxs_xyxy,proposed_objectness_scores,
                    self.nms_thresh)
        
        # Take after_nms_topN (e.g. 300)
        if self.n_post_nms > 0:
            keep = keep[:self.n_post_nms]

        return proposed_roi_bboxs[keep,:]

