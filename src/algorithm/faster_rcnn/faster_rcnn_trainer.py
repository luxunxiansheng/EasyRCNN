from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from fast_rcnn.fast_rcnn_network import FastRCNN
from feature_extractor import FeatureExtractorFactory
from rpn.anchor_creator import AnchorCreator
from rpn.anchor_target_creator import AnchorTargetCreator
from rpn.proposal_creator import ProposalCreator
from rpn.proposal_target_creator import ProposalTargetCreator
from rpn.region_proposal_network import RPN
from rpn.region_proposal_network_loss import RPNLoss
from fast_rcnn.fast_rcnn_loss import FastRCNNLoss



class FasterRCNNTrainer:
    def __init__(self,config,dataset,writer,device) -> None:
        self.config = config
        self.writer = writer
        self.device = device
        self.epoches = config.FASTER_RCNN.TRAIN.EPOCHS

        self.dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=config.RPN.TRAIN.NUM_WORKERS)    

        self.feature_extractor = FeatureExtractorFactory.create_feature_extractor(config.RPN.BACKBONE).to(device)
        
        self.anchor_creator = AnchorCreator(config,device)
    

        self.proposal_creator = ProposalCreator(config)
        self.proposal_target_creator = ProposalTargetCreator(config)
        
        self.rpn = RPN(config).to(device)
        self.fast_rcnn = FastRCNN(config).to(device)
        
        self.rpn_loss  = RPNLoss(config,device)   
        self.fast_rcnn_loss = FastRCNNLoss(config)

        params = list(self.feature_extractor.parameters()) + list(self.rpn.parameters()) + list(self.fast_rcnn.parameters())
    
        self.optimizer = optim.SGD(params=params,lr=float(config.FASTER_RCNN.TRAIN.LEARNING_RATE),
                                    momentum=float(config.FASTER_RCNN.TRAIN.MOMENTUM),
                                    weight_decay=config.FASTER_RCNN.TRAIN.WEIGHT_DECAY)


    def train(self):
        steps = 0 

        for epcho in tqdm(range(self.epoches)):
            for _,(images,bboxes,labels,_,img_file) in tqdm(enumerate(self.dataloader)):

                images,bboxes,labels = images.to(self.device),bboxes.to(self.device),labels.to(self.device)
                
                features = self.feature_extractor(images.float())

                rpn_predicted_scores, rpn_predicted_locs = self.rpn(features)
                
                total_cls_loss = torch.tensor(0.0,requires_grad=True).to(self.device)
                total_reg_loss = torch.tensor(0.0,requires_grad=True).to(self.device)
                for image_index in range(images.shape[0]):
                    feature = features[image_index]
                    image = images[image_index]
                    feature_height,feature_width = feature.shape[1:]
                    
                    img_height,img_width = image.shape[1:]
                    gt_bboxes = bboxes[image_index]
                    gt_labels = labels[image_index]

                    rpn_predicted_scores = rpn_predicted_scores[image_index]
                    rpn_predicted_locs = rpn_predicted_locs[image_index]

                    anchors_of_img = self.anchor_creator.generate(feature_height,feature_width)
                    rpn_cls_loss,rpn_reg_los=self.rpn_loss( anchors_of_img,
                                                            rpn_predicted_scores,
                                                            rpn_predicted_locs,
                                                            gt_bboxes,
                                                            img_height,
                                                            img_width,
                                                        )
                    
                    total_cls_loss+= rpn_cls_loss
                    total_reg_loss+= rpn_reg_los

                    proposed_roi_bboxes =self.proposal_creator.generate(anchors_of_img,
                                                                        rpn_predicted_scores,
                                                                        rpn_predicted_locs,
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
                    total_cls_loss+= roi_cls_loss
                    total_reg_loss+= roi_reg_loss

                with torch.autograd.set_detect_anomaly(True):
                    total_loss = total_cls_loss + total_reg_loss
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                steps += 1


