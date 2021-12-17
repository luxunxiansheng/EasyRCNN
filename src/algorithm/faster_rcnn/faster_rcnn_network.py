import torch
from torch import nn


from torch.nn import functional as F
from torchvision.ops import nms

from fast_rcnn.fast_rcnn_network import FastRCNN
from feature_extractor import FeatureExtractorFactory
from rpn.anchor_creator import AnchorCreator
from rpn.proposal_creator import ProposalCreator
from rpn.region_proposal_network import RPN
from utility import Utility

class FasterRCNN(nn.Module):
    def __init__(self,config,writer,device='cpu'):
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
        
    def forward(self,image_batch):
        feature_batch= self.feature_extractor(image_batch)
        rpn_predicted_score_batch ,rpn_predicted_loc_batch = self.rpn(feature_batch)
        
        bboxes_batch = list()
        labels_batch = list()
        scores_batch = list()

        for image_index in range(len(image_batch)):
            img_height, img_width = image_batch[image_index].shape[1:]
            feature = feature_batch[image_index]
            feature_height,feature_width = feature.shape[1:]
            rpn_predicted_scores = rpn_predicted_score_batch[image_index]
            rpn_predicted_locs = rpn_predicted_loc_batch[image_index]

            anchors_of_img = self.anchor_creator.generate(feature_height,feature_width)
            proposed_roi_bboxes =self.proposal_creator.generate(anchors_of_img,
                                                                rpn_predicted_scores,
                                                                rpn_predicted_locs,
                                                                img_height,
                                                                img_width,
                                                                feature_height,
                                                                feature_width)

            
            bboxes, labels, scores = self.detect(  feature, 
                                                    proposed_roi_bboxes, 
                                                    img_height, 
                                                    img_width,
                                                    self.config.FASTER_RCNN.VISUAL_SCORE_THRESHOLD,
                                                    self.config.FASTER_RCNN.VISUAL_NMS_THRESHOLD)
            
            bboxes_batch.append(bboxes)
            labels_batch.append(labels)
            scores_batch.append(scores)
            
        return bboxes_batch,labels_batch,scores_batch

    def detect(self, feature, proposed_roi_bboxes,img_height,img_width,score_threshold=0.05,nms_threshold=0.3):
        predicted_roi_score,predicted_roi_loc= self.fast_rcnn(feature,proposed_roi_bboxes)
        
        # post processing 
        predicted_roi_bboxes = Utility.loc2bbox(proposed_roi_bboxes,predicted_roi_loc)
            
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
                labels.append((class_index) * torch.ones((len(keep),)))
                scores.append(class_prob[keep])
        
        #  concatenate all bboxes and scores only if there is at least one bbox left for 
        #  any class, elsewise return empty tensors
        if len(bboxes) > 0:
            bboxes = torch.cat(bboxes, dim=0).to(self.device)
            labels = torch.cat(labels, dim=0).to(self.device)
            scores = torch.cat(scores, dim=0).to(self.device)
        else:
            bboxes = torch.empty((0, 4),device=self.device)
            labels = torch.empty((0,),device=self.device)
            scores = torch.empty((0,),device=self.device)

        return labels, scores, bboxes  