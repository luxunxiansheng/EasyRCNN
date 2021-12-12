import logging

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from rpn.anchor_creator import AnchorCreator
from rpn.anchor_target_creator import AnchorTargetCreator
from feature_extractor import FeatureExtractorFactory
from rpn.region_proposal_network import RPN
from rpn.region_proposal_network_loss import RPNLoss
from visual_tool import draw_img_bboxes_labels

#from tool.checkpoint_tool import load_checkpoint, save_checkpoint

class RPNTrainer:
    logger = logging.getLogger('RPNTrainer')
    level = logging.getLevelName('DEBUG')
    logger.setLevel(level)

    def __init__(self,config,dataset,writer,device):
        
        self.config = config
        self.writer = writer
        self.device = device
        self.epoches = config.RPN.TRAIN.EPOCHS 
        self.dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=config.RPN.TRAIN.NUM_WORKERS)    

        self.feature_extractor = FeatureExtractorFactory.create_feature_extractor(config.RPN.BACKBONE).to(device)
        self.rpn = RPN(config).to(device)

        params = list(self.feature_extractor.parameters()) + list(self.rpn.parameters())
    
        self.optimizer = optim.SGD(params=params,lr=float(config.RPN.TRAIN.LEARNING_RATE),
                                    momentum=float(config.RPN.TRAIN.MOMENTUM),
                                    weight_decay=config.RPN.TRAIN.WEIGHT_DECAY)

        self.loss  = RPNLoss(config,device)
                
    def train(self):
        
        steps = 0
        
        for epoch in tqdm(range(self.epoches)):
            for _,(images,bboxes,labels,_,img_file)  in tqdm(enumerate(self.dataloader)):

                images,bboxes,labels = images.to(self.device),bboxes.to(self.device),labels.to(self.device)
                
                features = self.feature_extractor(images.float())
                predicted_scores, predicted_locs = self.rpn(features)
                
                total_cls_loss = torch.tensor(0.0,requires_grad=True).to(self.device)
                total_reg_loss = torch.tensor(0.0,requires_grad=True).to(self.device)
                for image_index in range(images.shape[0]):
                    img_height,img_width = images[image_index].shape[1:]
                    feature_height,feature_width = features[image_index].shape[1:]
                    cls_loss,reg_los=self.loss(predicted_scores[image_index],predicted_locs[image_index],bboxes[image_index],img_height,img_width,feature_height,feature_width)
                    total_cls_loss+= cls_loss
                    total_reg_loss+= reg_los
                    
                total_loss = total_cls_loss + total_reg_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                steps += 1

                if steps%self.config.RPN.TRAIN.CHECK_FREQUENCY==0:
                    self.writer.add_scalar('rpn/cls_loss',total_cls_loss.item(),steps)
                    self.writer.add_scalar('rpn/reg_loss',total_reg_loss.item(),steps)
                    self.writer.add_histogram('rpn/conv1',self.rpn.conv1.conv.weight,steps)

                    # img_and_gt_bboxes = draw_img_bboxes_labels(images,bboxes,labels)
                    # self.writer.add_images('gt_boxes',img_and_gt_bboxes)

                    # predicted_bboxes = Utility.loc2bbox(anchors_of_img,predicted_locs)

                    # img_and_predicted_bboxes = draw_img_bboxes_labels(images,predicted_bboxes)
                    # self.writer.add_images('predicted_boxes',img_and_predicted_bboxes)

                    















