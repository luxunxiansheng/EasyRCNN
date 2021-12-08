import torch    
from torchvision.ops import box_iou
from utility import Utility

class AnchorTargetCreator:
    """Assign the ground truth bounding boxes to anchors."""
    def __init__(self,config,device='cpu'):
        self.n_samples =      config.RPN.ANCHOR_TARGET_CREATOR.N_SAMPLES
        self.pos_iou_thresh = config.RPN.ANCHOR_TARGET_CREATOR.POSITIVE_IOU_THRESHOLD
        self.neg_iou_thresh = config.RPN.ANCHOR_TARGET_CREATOR.NEGATIVE_IOU_THRESHOLD
        self.pos_ratio =      config.RPN.ANCHOR_TARGET_CREATOR.POSITIVE_RATIO
        self.device =         device

    def generate(self, anchors_of_image,gt_bboxs,img_H,img_W):
        """Generate the labels and the target regression values.

        Args:
            anchors_of_image (Tensor): all the anchors of the shape (n_anchors, 4)
            gt_bboxs (Tensor): Ground truth bounding boxes of the shape (n_gt_boxes, 4)
            img_H (int): Image height
            img_W (int): Image width

        Returns:
            labels(Tensor): The labels of the shape (n_anchors, ), 
                            each element is either -1, 0 or 1. -1 
                            means ignore, 0 means negative and 1 
                            means positive.

                            Note that the final labels have been 
                            sampled to keep the balance of positive
                            and negative.

            locs(Tensor):   The regression values of the shape (n_anchors,4)
                            All the valid anchors have been computed.
        """
        num_anchors_of_img = len(anchors_of_image)

        # get the index of anchors  inside the image
        valid_indices = self._get_inside_indices(anchors_of_image, img_H, img_W)

        if len(valid_indices) == 0:
            return None,None

        # get the anchors inside the image
        valid_anchors = anchors_of_image[valid_indices]

        # create labels for those valid anchors (inside the image). For tranning efficence, 
        # we only sample n_samples*pos_ratio positive anchors and n_smaples*(1-pos_ratio)
        # negative anchors.
    
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels_for_valid_anchor = torch.empty((len(valid_indices),), dtype=torch.int32,device=self.device)
        labels_for_valid_anchor.fill_(-1)

        
        argmax_ious_for_valid_anchor, max_ious_for_valid_anchor, argmax_ious_for_gt_box = self._calc_ious(valid_anchors,valid_indices,gt_bboxs)
        
        # Assign negitive label (0) to all the anchor boxes which have max_iou less than negitive threshold 
        labels_for_valid_anchor[max_ious_for_valid_anchor < self.neg_iou_thresh] = 0

        # Assign positive label (1) to all the anchor boxes which have highest IoU overlap with a ground-truth box
        labels_for_valid_anchor[argmax_ious_for_gt_box] = 1

        # Assign positive label (1) to all the anchor boxes which have max_iou greater than positive threshold [b]
        labels_for_valid_anchor[max_ious_for_valid_anchor >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_positive = int(self.pos_ratio * self.n_samples)
        positive_index = torch.where(labels_for_valid_anchor == 1)[0]

        if len(positive_index) > n_positive:
            disable_index = torch.multinomial(positive_index.float(),num_samples=(len(positive_index) - n_positive), replacement=False)
            disabled_positive_index = positive_index[disable_index]
            labels_for_valid_anchor[disabled_positive_index] = -1

        # subsample negative labels if we have too many
        n_negative = self.n_samples - torch.sum(labels_for_valid_anchor == 1)
        negative_index = torch.where(labels_for_valid_anchor == 0)[0]
        if len(negative_index) > n_negative:
            disable_index = torch.multinomial(negative_index.float(),num_samples=(len(negative_index) - n_negative), replacement=False)
            disabled_negative_index = negative_index[disable_index]
            labels_for_valid_anchor[disabled_negative_index] = -1

        # compute bounding box regression targets
        # Note, we will compute the regression targets for all the anchors inside the image 
        # irrespective of its label. 
        valid_locs = Utility.bbox2loc(valid_anchors, gt_bboxs[argmax_ious_for_valid_anchor])

        # map up to original set of anchors
        labels = self._unmap(labels_for_valid_anchor, num_anchors_of_img, valid_indices, fill=-1)
        locs = self._unmap(valid_locs, num_anchors_of_img, valid_indices, fill=0)
        
        return labels,locs 

        
    def _calc_ious(self, anchors,anchor_indices,gt_bboxs):
        # ious between the anchors and the gt boxes
        ious = box_iou(anchors, gt_bboxs)

        # for each gt box, find the anchor with the highest iou
        #argmax_ious_for_gt_box = ious.argmax(dim=0)
        #max_ious_for_gt_box = ious[argmax_ious_for_gt_box,torch.arange(ious.shape[1])]

        max_ious_for_gt_box,argmax_ious_for_gt_box = ious.max(dim=0)

        # for each gt box, there mihgt be multiple anchors with the same highest iou
        argmax_ious_for_gt_box = torch.where(ious == max_ious_for_gt_box)[0]

        # for each anchor, find the gt box with the highest iou
        #argmax_ious_for_anchor = ious.argmax(dim=1)
        #max_ious_for_anchor = ious[torch.arange(len(anchor_indices)), argmax_ious_for_anchor]

        max_ious_for_anchor,argmax_ious_for_anchor = ious.max(dim=1)
        
        return argmax_ious_for_anchor, max_ious_for_anchor, argmax_ious_for_gt_box

    @staticmethod
    def _get_inside_indices(anchors, H, W):
        # Calc indicies of anchors which are located completely inside of the image
        # whose size is speficied.
        indices_inside = torch.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= H) &
            (anchors[:, 3] <= W)
        )[0]
        return indices_inside


    def _unmap(self,data, count, index, fill=0):
        # Unmap a subset of item (data) back to the original set of items (of
        # size count)

        if len(data.shape) == 1:
            ret = torch.empty((count,), dtype=data.dtype,device=self.device)
            ret.fill_(fill)
            ret[index] = data
        else:
            ret = torch.empty((count,) + data.shape[1:], dtype=data.dtype,device=self.device)
            ret.fill_(fill)
            ret[index, :] = data
        return ret