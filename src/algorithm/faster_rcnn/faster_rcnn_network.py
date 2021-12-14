import torch
from torch import nn

from torch.nn import functional as F
from torchvision.ops import boxes

from fast_rcnn.fast_rcnn_network import FastRCNN
from feature_extractor import FeatureExtractorFactory
from rpn.anchor_creator import AnchorCreator
from rpn.proposal_creator import ProposalCreator
from rpn.region_proposal_network import RPN
from utility import Utility

class FasterRCNN(nn.Module):
    def __init__(self,config,):
        super().__init__()
        self.config = config
        self.feature_extractor = FeatureExtractorFactory.create_feature_extractor(config.FASTER_RCNN.FEATRUE_EXTRACTOR)
        self.rpn = RPN(config)
        self.fast_rcnn = FastRCNN(config)
        self.n_class = self.fast_rcnn.n_classes
        self.anchor_creator = AnchorCreator(config)
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
            proposed_roi_bboxes =self.proposal_creator.generate(anchors_of_img,rpn_predicted_scores,rpn_predicted_locs,img_height,img_width,feature_height,feature_width)
            proposed_roi_bbox_indices = torch.zeros(len(proposed_roi_bboxes))
            predicted_roi_score,predicted_roi_loc= self.fast_rcnn(feature,proposed_roi_bboxes,proposed_roi_bbox_indices)
            predicted_roi_bboxes = Utility.loc2bbox(proposed_roi_bboxes,predicted_roi_loc)
            
            predicted_roi_bboxes[:,0::2] =(predicted_roi_bboxes[:,0::2]).clamp(min=0,max=img_height)
            predicted_roi_bboxes[:,1::2] =(predicted_roi_bboxes[:,1::2]).clamp(min=0,max=img_width)

            prob = F.softmax(predicted_roi_score,dim=1)

            bboxes, labels, scores = self._suppress(predicted_roi_bboxes, prob)

            bboxes_batch.append(bboxes)
            labels_batch.append(labels)
            scores_batch.append(scores)

        return bboxes_batch,labels_batch,scores_batch

    def _suppress(self, predicted_roi_bboxes, predicted_prob):
        bboxes = list()
        labels = list()
        scores = list()
        for label_index in range(1, self.n_class+1):
            cls_bbox_with_label_index = predicted_roi_bboxes.reshape((-1, self.n_class+1, 4))[:, label_index, :]
            prob_with_label_index = predicted_prob[:, label_index]
            mask = prob_with_label_index > self.config.FASTER_RCNN.EVALUATE_SCORE_THRESHOLD
            cls_bbox_with_label_index = cls_bbox_with_label_index[mask]
            prob_with_label_index = prob_with_label_index[mask]
            keep = boxes.nms(cls_bbox_with_label_index, prob_with_label_index,self.config.FASTER_RCNN.EVALUATE_NMS_THRESHOLD)
            bboxes.append(cls_bbox_with_label_index[keep])
            labels.append((label_index-1) * torch.ones((len(keep),)))
            scores.append(prob_with_label_index[keep])
        bboxes = torch.concat(bboxes, dim=0).to(float)
        labels = torch.concat(labels,dim=0).to(int)
        scores = torch.concat(scores,dim=0).to(float)
        return bboxes, labels, scores