from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchmetrics.detection.map import MAP 

from faster_rcnn.faster_rcnn_network import FasterRCNN
from rpn.proposal_target_creator import ProposalTargetCreator
from rpn.region_proposal_network_loss import RPNLoss
from fast_rcnn.fast_rcnn_loss import FastRCNNLoss
from visual_tool import draw_img_bboxes_labels
from checkpoint_tool import  load_checkpoint, save_checkpoint


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
        
        self.resume = config.FASTER_RCNN.TRAIN.RESUME
        self.checkpoint_path = config.CHECKPOINT.CHECKPOINT_PATH
        
        self.metric = MAP()

    def train(self):
        steps = 0 
        start_epoch= 0

        if self.resume:
            steps, start_epoch = self._resume()

        total_loss = torch.tensor(0.0,requires_grad=True,device=self.device)
        for epoch in tqdm(range(start_epoch,self.epoches)):
            for _,(images_batch,bboxes_batch,labels_batch,_,img_file) in tqdm(enumerate(self.dataloader)):

                images_batch,bboxes_batch,labels_batch = images_batch.to(self.device),bboxes_batch.to(self.device),labels_batch.to(self.device)
                
                features_batch = self.feature_extractor.predict(images_batch.float())

                rpn_predicted_scores_batch, rpn_predicted_locs_batch = self.rpn.predict(features_batch)
                
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

                    anchors_of_img = self.anchor_creator.create(feature_height,feature_width)
                    
                    rpn_cls_loss,rpn_reg_los=self.rpn_loss.compute( anchors_of_img,
                                                            rpn_predicted_scores,
                                                            rpn_predicted_locs,
                                                            gt_bboxes,
                                                            img_height,
                                                            img_width,
                                                        )
                    
                    total_rpn_cls_loss = total_rpn_cls_loss + rpn_cls_loss
                    total_rpn_reg_loss = total_rpn_reg_loss + rpn_reg_los

                    proposed_roi_bboxes =self.proposal_creator.create(anchors_of_img,
                                                                        rpn_predicted_scores.detach(),
                                                                        rpn_predicted_locs.detach(),
                                                                        img_height,
                                                                        img_width,
                                                                        feature_height,
                                                                        feature_width)

                    sampled_roi,gt_roi_label,gt_roi_loc = self.proposal_target_creator.create(proposed_roi_bboxes,
                                                                                                gt_bboxes,
                                                                                                gt_labels,
                                                                                                img_height,
                                                                                                img_width)
                    
                    predicted_roi_cls_score,predicted_roi_loc = self.fast_rcnn.predict(feature,sampled_roi)
                    roi_cls_loss,roi_reg_loss = self.fast_rcnn_loss.compute(predicted_roi_cls_score,
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

                if steps%self.config.FASTER_RCNN.TRAIN.CHECK_FREQUENCY==0:
                    self.writer.add_scalar('rpn/cls_loss',total_rpn_cls_loss.item(),steps)
                    self.writer.add_scalar('rpn/reg_loss',total_rpn_reg_loss.item(),steps)
                    self.writer.add_scalar('roi/cls_loss',total_roi_cls_loss.item(),steps)
                    self.writer.add_scalar('roi/reg_loss',total_roi_reg_loss.item(),steps)
                    self.writer.add_scalar('total_loss',total_loss.item(),steps)
                    self.writer.add_histogram('rpn/conv1',self.rpn.conv1.conv.weight,steps)
                    self.writer.add_histogram('roi/fc7',self.fast_rcnn.fc7.fc.weight,steps)

                    with torch.no_grad():
                        predicted_labels_batch, predicted_scores_batch,predicted_bboxes_batch = self.faster_rcnn(images_batch.float())
                        
                        predicted_labels_for_img_0 = predicted_labels_batch[0]
                        predicted_label_names_for_img_0 = []
                        for label_index in predicted_labels_for_img_0:
                            predicted_label_names_for_img_0.append(self.dataloader.dataset.get_label_names()[label_index.long().item()])

                        if len(predicted_label_names_for_img_0) >0:
                            label_names = [self.dataloader.dataset.get_label_names()[label_index] for label_index in labels_batch[0]] 
                            img_and_gt_bboxes = draw_img_bboxes_labels(images_batch[0],
                                                                        bboxes_batch[0],
                                                                        label_names, 
                                                                        resize_shape=[img_height,img_width],
                                                                        colors='green')

                            self.writer.add_images('gt_boxes',img_and_gt_bboxes.unsqueeze(0),steps)
                            
                            predicted_bboxes_for_img_0 = predicted_bboxes_batch[0]
                            
                            img_and_predicted_bboxes = draw_img_bboxes_labels(images_batch[0],
                                                                            predicted_bboxes_for_img_0,
                                                                            predicted_label_names_for_img_0,
                                                                            resize_shape=[img_height,img_width],
                                                                            colors='red')

                            self.writer.add_images('predicted_boxes',img_and_predicted_bboxes.unsqueeze(0),steps)

                            predicted_scores_for_img_0 = predicted_scores_batch[0]
                            map =self._evaluate(gt_bboxes, gt_labels, predicted_scores_for_img_0, predicted_labels_for_img_0, predicted_bboxes_for_img_0)
                            self.writer.add_scalar('map',map,steps)
                
                    # save checkpoint if needed
                    cpkt = {
                            'feature_extractor_model':self.feature_extractor.state_dict(),
                            'rpn_model': self.rpn.state_dict(),
                            'fast_rcnn_model': self.fast_rcnn.state_dict(),
                            'epoch': epoch,
                            'steps': steps,
                            'optimizer': self.optimizer.state_dict()
                            }
                    
                    save_checkpoint(cpkt, self.checkpoint_path)

                steps += 1

    def _resume(self):
        ckpt = load_checkpoint(self.checkpoint_path) # custom method for loading last checkpoint
        self.feature_extractor.load_state_dict(ckpt['feature_extractor_model'])
        self.rpn.load_state_dict(ckpt['rpn_model'])
        self.fast_rcnn.load_state_dict(ckpt['fast_rcnn_model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        steps = ckpt['steps']
        return steps,start_epoch

    def _evaluate(self, gt_bboxes, gt_labels, predicted_scores, predicted_labels, predicted_bboxes):
        preds = [dict(
                    # convert yxyx to xyxy
                    boxes = predicted_bboxes.index_select(1,torch.tensor([1,0,3,2],device=predicted_bboxes.device)),
                    scores = predicted_scores,
                    labels = predicted_labels,
                    )]
        
        target = [dict(
                    boxes = gt_bboxes.index_select(1,torch.tensor([1,0,3,2],device=gt_bboxes.device)),
                    labels = gt_labels,
                    )]  

        self.metric.update(preds,target)
        return self.metric.compute()['map'].item()
        

        