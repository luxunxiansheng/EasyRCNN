from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from fast_rcnn.fast_rcnn_network import FastRCNN
from faster_rcnn.faster_rcnn_network import FasterRCNN
from feature_extractor import FeatureExtractorFactory
from rpn.anchor_creator import AnchorCreator
from rpn.anchor_target_creator import AnchorTargetCreator
from rpn.proposal_creator import ProposalCreator
from rpn.proposal_target_creator import ProposalTargetCreator
from rpn.region_proposal_network import RPN
from rpn.region_proposal_network_loss import RPNLoss
from fast_rcnn.fast_rcnn_loss import FastRCNNLoss
from visual_tool import draw_img_bboxes_labels

class FasterRCNNTrainer:
    def __init__(self,config,dataset,writer,device) -> None:
        self.config = config
        self.writer = writer
        self.device = device
        self.epoches = config.FASTER_RCNN.TRAIN.EPOCHS

        self.dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=config.RPN.TRAIN.NUM_WORKERS)    
        self.faster_rcnn = FasterRCNN(config,writer,device)
        self.feature_extractor = self.faster_rcnn.feature_extractor
        self.rpn = self.faster_rcnn.rpn
        self.fast_rcnn = self.faster_rcnn.fast_rcnn
        self.anchor_creator = self.faster_rcnn.anchor_creator
        self.proposal_creator = self.faster_rcnn.proposal_creator


        self.proposal_target_creator = ProposalTargetCreator(config)
        self.rpn_loss  = RPNLoss(config,device)   
        self.fast_rcnn_loss = FastRCNNLoss(config,device)

        params = list(self.feature_extractor.parameters()) + list(self.rpn.parameters()) + list(self.fast_rcnn.parameters())
        self.optimizer = optim.SGD(params=params,lr=float(config.FASTER_RCNN.TRAIN.LEARNING_RATE),
                                    momentum=float(config.FASTER_RCNN.TRAIN.MOMENTUM),
                                    weight_decay=config.FASTER_RCNN.TRAIN.WEIGHT_DECAY)

    def train(self):
        steps = 0 
        total_loss = torch.tensor(0.0,requires_grad=True,device=self.device)

        for epcho in tqdm(range(self.epoches)):
            for _,(images_batch,bboxes_batch,labels_batch,_,img_file) in tqdm(enumerate(self.dataloader)):

                images_batch,bboxes_batch,labels_batch = images_batch.to(self.device),bboxes_batch.to(self.device),labels_batch.to(self.device)
                
                features_batch = self.feature_extractor(images_batch.float())

                rpn_predicted_scores_batch, rpn_predicted_locs_batch = self.rpn(features_batch)
                
                total_rpn_cls_loss = torch.tensor(0.0,requires_grad=True,device=self.device)
                total_rpn_reg_loss = torch.tensor(0.0,requires_grad=True,device=self.device)
                total_roi_cls_loss = torch.tensor(0.0,requires_grad=True,device=self.device)
                total_roi_reg_loss = torch.tensor(0.0,requires_grad=True,device=self.device)
                
                for image_index in range(images_batch.shape[0]):
                    feature = features_batch[image_index]
                    image = images_batch[image_index]
                    feature_height,feature_width = feature.shape[1:]
                    
                    img_height,img_width = image.shape[1:]
                    gt_bboxes = bboxes_batch[image_index]
                    gt_labels = labels_batch[image_index]

                    rpn_predicted_scores = rpn_predicted_scores_batch[image_index]
                    rpn_predicted_locs = rpn_predicted_locs_batch[image_index]

                    anchors_of_img = self.anchor_creator.generate(feature_height,feature_width)
                    
                    rpn_cls_loss,rpn_reg_los=self.rpn_loss( anchors_of_img,
                                                            rpn_predicted_scores,
                                                            rpn_predicted_locs,
                                                            gt_bboxes,
                                                            img_height,
                                                            img_width,
                                                        )
                    
                    total_rpn_cls_loss = total_rpn_cls_loss + rpn_cls_loss
                    total_rpn_reg_loss = total_rpn_reg_loss + rpn_reg_los

                    proposed_roi_bboxes =self.proposal_creator.generate(anchors_of_img,
                                                                        rpn_predicted_scores.detach(),
                                                                        rpn_predicted_locs.detach(),
                                                                        img_height,
                                                                        img_width,
                                                                        feature_height,
                                                                        feature_width)

                    sampled_roi,gt_roi_label,gt_roi_loc = self.proposal_target_creator.generate(proposed_roi_bboxes,
                                                                                                gt_bboxes,
                                                                                                gt_labels,
                                                                                                img_height,
                                                                                                img_width)

                    sampled_roi_bbox_indices = torch.zeros(len(sampled_roi),device=self.device)
                    predicted_roi_cls_score,predicted_roi_loc = self.fast_rcnn(feature,sampled_roi,sampled_roi_bbox_indices)
                    roi_cls_loss,roi_reg_loss = self.fast_rcnn_loss(predicted_roi_cls_score,
                                                                    predicted_roi_loc,
                                                                    gt_roi_label,
                                                                    gt_roi_loc)                                                                    
                    
                    total_roi_cls_loss = total_roi_cls_loss + roi_cls_loss
                    total_roi_reg_loss = total_roi_reg_loss + roi_reg_loss

                with torch.autograd.set_detect_anomaly(True): 
                    total_loss = total_rpn_cls_loss + total_rpn_reg_loss+total_roi_cls_loss+total_roi_reg_loss
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                steps += 1

                if steps%self.config.FASTER_RCNN.TRAIN.CHECK_FREQUENCY==0:
                    self.writer.add_scalar('rpn/cls_loss',total_rpn_cls_loss.item(),steps)
                    self.writer.add_scalar('rpn/reg_loss',total_rpn_reg_loss.item(),steps)
                    self.writer.add_scalar('roi/cls_loss',total_roi_cls_loss.item(),steps)
                    self.writer.add_scalar('roi/reg_loss',total_roi_reg_loss.item(),steps)
                    self.writer.add_scalar('total_loss',total_loss.item(),steps)
                    self.writer.add_histogram('rpn/conv1',self.rpn.conv1.conv.weight,steps)
                    self.writer.add_histogram('roi/fc7',self.fast_rcnn.fc7.fc.weight,steps)

                    img_and_gt_bboxes = draw_img_bboxes_labels(images_batch,bboxes_batch,labels_batch)
                    self.writer.add_images('gt_boxes',img_and_gt_bboxes)

                    predicted_labels, predicted_scores,predicted_bboxes, = self.faster_rcnn(images_batch)
                    img_and_predicted_bboxes = draw_img_bboxes_labels(images_batch,predicted_bboxes,predicted_labels)
                    self.writer.add_images('predicted_boxes',img_and_predicted_bboxes)


