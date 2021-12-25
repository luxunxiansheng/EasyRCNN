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
from torch import nn

from torch.nn import functional as F
from torch.types import Device
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.ops import nms
from yacs.config import CfgNode

from fast_rcnn.fast_rcnn_network import FastRCNN
from feature_extractor import FeatureExtractorFactory
from rpn.anchor_creator import AnchorCreator
from rpn.proposal_creator import ProposalCreator
from rpn.region_proposal_network import RPN
from location_utility import LocationUtility

class FasterRCNN(nn.Module):
    def __init__(self,
                config:CfgNode,
                writer:SummaryWriter,
                device:Device='cpu'):

        super().__init__()
        self.config = config
        self.writer = writer
        self.device = device
        self.feature_extractor = FeatureExtractorFactory.create_feature_extractor(config.FASTER_RCNN.FEATRUE_EXTRACTOR).to(device)
        self.rpn = RPN(config).to(device)
        self.fast_rcnn = FastRCNN(config).to(device)
        self.n_class = self.fast_rcnn.n_classes
        self.anchor_creator = AnchorCreator(config,device=device)
        self.proposal_creator = ProposalCreator(config)
        self.offset_norm_mean = torch.tensor(config.FASTER_RCNN.OFFSET_NORM_MEAN).to(device)
        self.offset_norm_std =  torch.tensor(config.FASTER_RCNN.OFFSET_NORM_STD).to(device)

    def predict(self,image_batch:torch.Tensor):
        """A explict interface for predict rahter than forward

        Args:
            image_batch (torch.Tensor): [batch_size,3,height,width]

        Returns:
            return forward result
        """
        return self.forward(image_batch)

        
    def forward(self,image_batch):
        """ A forward interface for faster rcnn
        
        Args:
            image_batch (torch.Tensor): [batch_size,3,height,width]
        
        returns:
            return a dict contains:
                'bboxes': [batch_size,n_bboxes,4]
                'labels': [batch_size,n_bboxes]
                'scores': [batch_size,n_bboxes]

        """

        feature_batch= self.feature_extractor.predict(image_batch)
        rpn_predicted_score_batch ,rpn_predicted_offset_batch = self.rpn.predict(feature_batch)
        
        bboxes_batch = list()
        labels_batch = list()
        scores_batch = list()

        for image_index in range(len(image_batch)):
            img_height, img_width = image_batch[image_index].shape[1:]
            feature = feature_batch[image_index]
            feature_height,feature_width = feature.shape[1:]
            rpn_predicted_scores = rpn_predicted_score_batch[image_index]
            rpn_predicted_offsets = rpn_predicted_offset_batch[image_index]

            anchors_of_img = self.anchor_creator.create(feature_height,feature_width)
            proposed_roi_bboxes =self.proposal_creator.create(anchors_of_img,
                                                                rpn_predicted_scores,
                                                                rpn_predicted_offsets,
                                                                img_height,
                                                                img_width,
                                                                feature_height,
                                                                feature_width)

            
            bboxes, labels, scores = self.detect(feature, 
                                                proposed_roi_bboxes, 
                                                img_height, 
                                                img_width,
                                                self.config.FASTER_RCNN.VISUAL_SCORE_THRESHOLD,
                                                self.config.FASTER_RCNN.VISUAL_NMS_THRESHOLD)
            
            bboxes_batch.append(bboxes)
            labels_batch.append(labels)
            scores_batch.append(scores)
            
        return bboxes_batch,labels_batch,scores_batch

    def detect(self, 
                feature:torch.Tensor, 
                proposed_roi_bboxes:torch.Tensor,
                img_height:int,
                img_width:int,
                score_threshold:float,
                nms_threshold:float):
        """
        Args:
            feature (torch.Tensor): [C,H,W]
            proposed_roi_bboxes (torch.Tensor): [n_bboxes,4]
            img_height (int): height of image
            img_width (int): width of image
            score_threshold (float): threshold for score
            nms_threshold (float): threshold for nms

        Returns:
            bboxes (torch.Tensor): [n_bboxes,4]
            labels (torch.Tensor): [n_bboxes,]
            scores (torch.Tensor): [n_bboxes,]
        """

        predicted_roi_score,predicted_roi_loc= self.fast_rcnn.predict(feature,proposed_roi_bboxes)

        mean = self.offset_norm_mean.repeat(self.n_class+1)[None]
        std  = self.offset_norm_std.repeat(self.n_class+1)[None]

        predicted_roi_loc = predicted_roi_loc * std + mean
        
        # post processing 
        predicted_roi_bboxes = LocationUtility.offset2bbox(proposed_roi_bboxes,predicted_roi_loc)
            
        predicted_roi_bboxes[:,0::2] =(predicted_roi_bboxes[:,0::2]).clamp(min=0,max=img_height)
        predicted_roi_bboxes[:,1::2] =(predicted_roi_bboxes[:,1::2]).clamp(min=0,max=img_width)

        prob = F.softmax(predicted_roi_score,dim=1)

        bboxes, labels, scores = self._suppress(predicted_roi_bboxes, 
                                                    prob,
                                                    score_threshold,
                                                    nms_threshold)
                                                
        return bboxes,labels,scores

    def _suppress(self, predicted_roi_bboxes, predicted_prob,score_threshold,nms_threshold):
        bboxes = list()
        labels = list()
        scores = list()

        for class_index in range(1, self.n_class+1):
            cls_bbox = predicted_roi_bboxes.reshape((-1, self.n_class+1, 4))[:, class_index, :]
            class_prob = predicted_prob[:, class_index]
            
            # for current class, keep the top-K bboxes with highest scores
            mask = class_prob > score_threshold
            cls_bbox = cls_bbox[mask]
            class_prob = class_prob[mask]
            
            cls_bboxs_xyxy=cls_bbox.index_select(dim=1,
                                                index=torch.tensor([1,0,3,2],
                                                device=cls_bbox.device))

            keep = nms(cls_bboxs_xyxy,class_prob,nms_threshold)
            
            # keep top-K bboxes only if there is at least one bbox left for current class
            if keep.shape[0] > 0:
                bboxes.append(cls_bbox[keep])
                labels.append(((class_index) * torch.ones((len(keep),),dtype=torch.int32)).to(self.device))
                scores.append(class_prob[keep])
        
        #  concatenate all bboxes and scores only if there is at least one bbox left for 
        #  any class, elsewise return empty tensors
        if len(bboxes) > 0:
            bboxes = torch.cat(bboxes, dim=0).to(self.device)
            labels = torch.cat(labels, dim=0).to(self.device)
            scores = torch.cat(scores, dim=0).to(self.device)
        else:
            bboxes = torch.empty((0, 4),device=self.device)
            labels = torch.empty((0,), device=self.device)
            scores = torch.empty((0,), device=self.device)

        return labels, scores, bboxes  