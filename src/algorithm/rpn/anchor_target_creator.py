# #### BEGIN LICENSE BLOCK #####
# MIT License
#    
# Copyright (c) 2021 Bin.Li Bin.Li (ornot2008@yahoo.com)
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
from torch.types import Device    
from torchvision.ops import box_iou
from yacs.config import CfgNode
from location_utility import LocationUtility

class AnchorTargetCreator:
    """Assign the ground truth bounding boxes to anchors."""
    def __init__(self,config:CfgNode,device:Device='cpu'):
        
        self.n_samples =      config.RPN.ANCHOR_TARGET_CREATOR.N_SAMPLES
        self.pos_iou_thresh = config.RPN.ANCHOR_TARGET_CREATOR.POSITIVE_IOU_THRESHOLD
        self.neg_iou_thresh = config.RPN.ANCHOR_TARGET_CREATOR.NEGATIVE_IOU_THRESHOLD
        self.pos_ratio =      config.RPN.ANCHOR_TARGET_CREATOR.POSITIVE_RATIO
        self.device =         device

    def create(self, 
            anchors_of_image:torch.Tensor,
            gt_bboxs:torch.Tensor,
            img_H:int,
            img_W:int)->torch.Tensor:
        """Generate the target labels and regression values.
        
        Args:
            anchors_of_image: (N,4) tensor, the anchors of the image.
            gt_bboxs: (M,4) tensor, the ground truth bounding boxes.
            img_H: int the height of the image.
            img_W: int the width of the image.

        Returns:
            labels: (N,), the target labels of the anchors.
            offsets: (N,4) tensor,the target offsets of the anchors.
        """
        num_anchors_of_img = len(anchors_of_image)

        # get the index of anchors inside the image
        valid_indices = self._get_inside_indices(anchors_of_image, img_H, img_W)

        if len(valid_indices) == 0:
            return None,None

        # get the anchors inside the image
        valid_anchors = anchors_of_image[valid_indices]

        # 
        # create labels for those valid anchors (inside the image).
        #
    
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels_for_valid_anchor = torch.empty((len(valid_indices),), dtype=torch.int32,device=self.device)
        labels_for_valid_anchor.fill_(-1)
        
        
        #  argmax_ious_for_valid_anchor: (N,) tensor, the index of the ground truth box with highest IoU overlap with each anchor.
        #  max_ious_for_valid_anchor: (N,) tensor, the IoU of the ground truth box with highest IoU overlap with each anchor.
        #  argmax_ious_for_gt_box: (M,) tensor, the index of the anchor with highest IoU overlap with each ground truth box.
        argmax_ious_for_valid_anchor, max_ious_for_valid_anchor, argmax_ious_for_gt_box = self._calc_ious(valid_anchors,gt_bboxs)
        
        # Assign negitive label (0) to all the anchor boxes which have max_iou less than negitive threshold 
        labels_for_valid_anchor[max_ious_for_valid_anchor < self.neg_iou_thresh] = 0

        # Assign positive label (1) to all the anchor boxes which have highest IoU overlap with each ground-truth box
        labels_for_valid_anchor[argmax_ious_for_gt_box] = 1

        # Assign positive label (1) to all the anchor boxes which have max_iou greater than positive threshold [b]
        labels_for_valid_anchor[max_ious_for_valid_anchor >= self.pos_iou_thresh] = 1

        
        # 
        # For tranning efficence, we only sample n_samples*pos_ratio positive anchors 
        # and n_smaples*(1-pos_ratio) negative anchors.
        #
        n_positive = int(self.pos_ratio * self.n_samples)

        # subsample positive labels if we have too many
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

        #
        # compute bounding box regression targets.
        # Note, we will compute the regression targets for all the anchors inside the image 
        # irrespective of its label. 
        #

        offsets_for_valid_anchor = LocationUtility.bbox2offset(valid_anchors, gt_bboxs[argmax_ious_for_valid_anchor])

        # map up to original set of anchors
        labels = self._unmap(labels_for_valid_anchor, num_anchors_of_img, valid_indices, fill=-1)
        offsets = self._unmap(offsets_for_valid_anchor, num_anchors_of_img, valid_indices, fill=0)
        
        return labels,offsets 
    
    def _calc_ious(self, 
                anchors:torch.Tensor,
                gt_bboxs:torch.Tensor)->torch.Tensor:

        """Calculate the IoU of anchors with ground truth boxes.
        
        Args:
            anchors: (N,4) tensor, the anchors of the image.
            gt_bboxs: (M,4) tensor, the ground truth bounding boxes.
        
        Returns:
            argmax_ious_for_anchor: (N,) tensor, the index of the ground truth box with highest IoU overlap with the anchor. 
            max_ious_for_anchor: (N,) tensor, the IoU of the anchor with the ground truth box with highest IoU overlap.
            argmax_ious_for_gt_box: (M,) tensor, the index of the anchor with highest IoU overlap with the ground truth box.              
        """
        # ious between the anchors and the gt boxes
        ious = box_iou(anchors, gt_bboxs)

        # for each anchor, find the gt box with the highest iou
        max_ious_for_anchor,argmax_ious_for_anchor = ious.max(dim=1)

        # for each gt box, find the anchor with the highest iou
        max_ious_for_gt_box,argmax_ious_for_gt_box = ious.max(dim=0)

        # for each gt box, there mihgt be multiple anchors with the same highest iou
        argmax_ious_for_gt_box = torch.where(ious == max_ious_for_gt_box)[0]
        
        return argmax_ious_for_anchor, max_ious_for_anchor, argmax_ious_for_gt_box

    @staticmethod
    def _get_inside_indices(anchors:torch.Tensor, img_H:int, img_W:int)->torch.Tensor:
                            
        """Calc indicies of anchors which are located completely inside of the image
        whose size is speficied.
        
        Args:
            anchors: (N,4) tensor, all the anchors of the image.
            img_H: int the height of the image.
            img_W: int the width of the image.
        
        Returns:
            indices: (N,) tensor, the indices of the anchors which are located completely inside of the image.

        """
        indices_inside = torch.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= img_H) &
            (anchors[:, 3] <= img_W)
        )[0]
        
        return indices_inside


    def _unmap(self,
            data:torch.Tensor, 
            count:int, 
            index:torch.Tensor,
            fill:int=0):

        """Unmap a subset of item (data) back to the original set of items (of size count)
        
        Args:
            data: (N,) tensor, the subset of data to unmap.
            count: int, the size of the original set of items.
            index: (N,) tensor, the indices of the subset of data to unmap.
            fill: the value to fill the unmapped item with.
        
        Returns:
            ret: (count,) or (count,4) tensor, the original set of items.
        """

        if len(data.shape) == 1:
            ret = torch.empty((count,), dtype=data.dtype,device=self.device)
            ret.fill_(fill)
            ret[index] = data
        else:
            ret = torch.empty((count,) + data.shape[1:], dtype=data.dtype,device=self.device)
            ret.fill_(fill)
            ret[index, :] = data
        return ret