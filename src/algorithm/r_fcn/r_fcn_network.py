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
from torch import nn
from torch.nn import functional as F
from torch.types import Device
from torchvision.ops import PSRoIPool
from yacs.config import CfgNode

from common import CNNBlock
from feature_extractor import FeatureExtractorFactory
from location_utility import LocationUtility
from position_sensitive_fcn.position_senstive_network_loss import PositionSensitiveNetworkLoss
from rpn.anchor_creator import AnchorCreator
from rpn.proposal_creator import ProposalCreator
from rpn.region_proposal_network import RPN

class RFCN(nn.module):
    """
    R-FCN: Object Detection via Region-based Fully Convolutional Networks
    By Jifeng Dai, Yi Li, Kaiming He, Jian Sun

    Pseudo code:
    _________________________________________________________
    feature_maps = process(image)
    ROIs = region_proposal(feature_maps)         
    score_maps = compute_score_map(feature_maps)
    for ROI in ROIs
        V = region_roi_pool(score_maps, ROI)     
        class_scores, box = average(V)                  
        class_probabilities = softmax(class_scores) 
    _________________________________________________________
    """

    def __init__(self,
                config:CfgNode,
                device:Device='cpu') -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.feature_extractor = FeatureExtractorFactory.create_feature_extractor(config.R_FCN.FEATRUE_EXTRACTOR).to(device)
        self.rpn = RPN(config).to(device)
    
        self.anchor_creator = AnchorCreator(config,device=device)
        self.proposal_creator = ProposalCreator(config)

        self.ps_net = PositionSensitiveNetworkLoss(config)

        self.n_class = config.R_FCN.NUM_CLASSES

    
    def forward(self,image_batch):
        #* 1. feature extraction        
        feature_batch= self.feature_extractor.predict(image_batch)

        #* 2.  predicted offsets and class scores by rpn network
        rpn_predicted_score_batch,rpn_predicted_offset_batch = self.rpn.predict(feature_batch)
        
        bboxes_batch = list()
        labels_batch = list()
        scores_batch = list()

        for image_index in range(len(image_batch)):
            img_height, img_width = image_batch[image_index].shape[1:]
            feature = feature_batch[image_index]
            feature_height,feature_width = feature.shape[1:]
            rpn_predicted_scores = rpn_predicted_score_batch[image_index]
            rpn_predicted_offsets = rpn_predicted_offset_batch[image_index]

            #* 3. generate the proposed roi bboxes
            anchors_of_img = self.anchor_creator.create(feature_height,feature_width)
            proposed_roi_bboxes =self.proposal_creator.create(anchors_of_img,
                                                                rpn_predicted_scores,
                                                                rpn_predicted_offsets,
                                                                img_height,
                                                                img_width,
                                                                feature_height,
                                                                feature_width)
            #* 4. get the bboxes ,labels and scores based on the proposed roi bboxes
            bboxes, labels, scores = self.detect(feature, 
                                                proposed_roi_bboxes, 
                                                img_height, 
                                                img_width,
                                                self.config.FASTER_RCNN.SCORE_THRESHOLD,
                                                self.config.FASTER_RCNN.NMS_THRESHOLD)
            
            bboxes_batch.append(bboxes)
            labels_batch.append(labels)
            scores_batch.append(scores)
            
        return bboxes_batch,labels_batch,scores_batch

    def predict(self, x):
        return self.forward(x)

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

        predicted_roi_score,predicted_roi_offset = self.ps_net.predict(feature, proposed_roi_bboxes)


        mean = self.offset_norm_mean.repeat(self.n_class+1)[None]
        std  = self.offset_norm_std.repeat(self.n_class+1)[None]

        predicted_roi_offset = predicted_roi_offset * std + mean
        
        # post processing 
        predicted_roi_bboxes = LocationUtility.offset2bbox(proposed_roi_bboxes,predicted_roi_offset)
            
        predicted_roi_bboxes[:,0::2] =(predicted_roi_bboxes[:,0::2]).clamp(min=0,max=img_height)
        predicted_roi_bboxes[:,1::2] =(predicted_roi_bboxes[:,1::2]).clamp(min=0,max=img_width)

        prob = F.softmax(predicted_roi_score,dim=1)

        bboxes, labels, scores = self._suppress(predicted_roi_bboxes, 
                                                    prob,
                                                    score_threshold,
                                                    nms_threshold)
                                                
        return bboxes,labels,scores

