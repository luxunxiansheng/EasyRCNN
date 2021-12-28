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

from tqdm import tqdm

from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard.writer import SummaryWriter

import torch
from torch.types import Device
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torchmetrics.detection.map import MAP
from yacs.config import CfgNode
from faster_rcnn.faster_rcnn_evaluator import FasterRCNNEvaluator 

from faster_rcnn.faster_rcnn_network import FasterRCNN
from rpn.proposal_target_creator import ProposalTargetCreator
from rpn.region_proposal_network_loss import RPNLoss
from fast_rcnn.fast_rcnn_loss import FastRCNNLoss
from visual_tool import draw_img_bboxes_labels
from checkpoint_tool import  load_checkpoint, save_checkpoint

class FasterRCNNTrainer:
    def __init__(self,
                train_config:CfgNode,
                train_dataset:Dataset,
                writer:SummaryWriter,
                eval_config:CfgNode = None,
                eval_dataset:Dataset= None,
                device:Device='cpu') -> None:

        self.train_config = train_config
        self.eval_config = eval_config
        self.writer = writer
        self.device = device
        self.epoches = train_config.FASTER_RCNN.TRAIN.EPOCHS
        self.train_dataloader = DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=train_config.FASTER_RCNN.NUM_WORKERS)    
        self.faster_rcnn = FasterRCNN(train_config,device)
        self.feature_extractor = self.faster_rcnn.feature_extractor
        self.rpn = self.faster_rcnn.rpn
        self.fast_rcnn = self.faster_rcnn.fast_rcnn
        self.anchor_creator = self.faster_rcnn.anchor_creator
        self.proposal_creator = self.faster_rcnn.proposal_creator
        self.proposal_target_creator = ProposalTargetCreator(train_config)
        self.rpn_loss  = RPNLoss(train_config,device)   
        self.fast_rcnn_loss = FastRCNNLoss(train_config,device)

        params = list(self.feature_extractor.parameters()) + list(self.rpn.parameters()) + list(self.fast_rcnn.parameters())
        self.optimizer = optim.SGD( params=params,
                                    lr=train_config.FASTER_RCNN.TRAIN.LEARNING_RATE,
                                    momentum=train_config.FASTER_RCNN.TRAIN.MOMENTUM,
                                    weight_decay=train_config.FASTER_RCNN.TRAIN.WEIGHT_DECAY)
        self.scheduler = StepLR(self.optimizer,
                                step_size=train_config.FASTER_RCNN.TRAIN.STEP_SIZE,
                                gamma=train_config.FASTER_RCNN.TRAIN.LEARNING_RATE_DECAY)
        self.metric = MAP()

        self.resume = train_config.FASTER_RCNN.TRAIN.RESUME
        self.checkpoint_path = train_config.CHECKPOINT.CHECKPOINT_PATH
        
        
        if eval_config is not None:
            self.evaluator = FasterRCNNEvaluator(eval_config,eval_dataset,device)
            self.best_map_50 = 0
        else:
            self.evaluator = None

    def train(self):
        steps = 0 
        start_epoch= 0

        if self.resume:
            steps, start_epoch = self._resume()

        total_loss = torch.tensor(0.0,requires_grad=True,device=self.device)
        for epoch in tqdm(range(start_epoch,self.epoches)):
            # train the model for current epoch
            for _,(images_batch,bboxes_batch,labels_batch,_,img_file) in tqdm(enumerate(self.train_dataloader)):

                images_batch,bboxes_batch,labels_batch = images_batch.to(self.device),bboxes_batch.to(self.device),labels_batch.to(self.device)
                
                with torch.autograd.set_detect_anomaly(True): 
                    features_batch = self.feature_extractor.predict(images_batch.float())
                    rpn_predicted_scores_batch, rpn_predicted_offset_batch = self.rpn.predict(features_batch)
                
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
                    rpn_predicted_offsets = rpn_predicted_offset_batch[image_index]

                    anchors_of_img = self.anchor_creator.create(feature_height,feature_width)
                    
                    rpn_cls_loss,rpn_reg_los=self.rpn_loss.compute( anchors_of_img,
                                                            rpn_predicted_scores,
                                                            rpn_predicted_offsets,
                                                            gt_bboxes,
                                                            img_height,
                                                            img_width,
                                                        )
                    
                    total_rpn_cls_loss = total_rpn_cls_loss + rpn_cls_loss
                    total_rpn_reg_loss = total_rpn_reg_loss + rpn_reg_los

                    proposed_roi_bboxes =self.proposal_creator.create(anchors_of_img,
                                                                        rpn_predicted_scores.detach(),
                                                                        rpn_predicted_offsets.detach(),
                                                                        img_height,
                                                                        img_width,
                                                                        feature_height,
                                                                        feature_width)

                    sampled_roi,gt_roi_label,gt_roi_offset = self.proposal_target_creator.create(proposed_roi_bboxes,
                                                                                                gt_bboxes,
                                                                                                gt_labels
                                                                                            )
                    with torch.autograd.set_detect_anomaly(True): 
                        predicted_roi_cls_score,predicted_roi_offset = self.fast_rcnn.predict(feature,sampled_roi)
                    
                    roi_cls_loss,roi_reg_loss = self.fast_rcnn_loss.compute(predicted_roi_cls_score,
                                                                    predicted_roi_offset,
                                                                    gt_roi_label,
                                                                    gt_roi_offset)                                                                    
                    
                    total_roi_cls_loss = total_roi_cls_loss + roi_cls_loss
                    total_roi_reg_loss = total_roi_reg_loss + roi_reg_loss

                with torch.autograd.set_detect_anomaly(True): 
                    total_loss = total_rpn_cls_loss + \
                                total_rpn_reg_loss+ \
                                total_roi_cls_loss+ \
                                total_roi_reg_loss
                                
                    self.optimizer.zero_grad()
                    total_loss.backward()                    
                    self.optimizer.step()

                if steps%self.train_config.FASTER_RCNN.TRAIN.CHECK_FREQUENCY==0:
                    self._check_progress(steps, total_loss, images_batch, bboxes_batch, labels_batch, total_rpn_cls_loss, total_rpn_reg_loss, total_roi_cls_loss, total_roi_reg_loss, img_height, img_width, gt_bboxes, gt_labels)
                    
                steps += 1
            
            # adjust the learning rate if necessary
            self.scheduler.step()  
            
            # evaluate the model on test set for current epoch    
            is_best = False
            if self.evaluator is not None:
                eval_result =self.evaluator.evaluate(self.faster_rcnn)
                self.writer.add_scalar('eval/map',eval_result['map'].item(),steps)
                self.writer.add_scalar('eval/map_50',eval_result['map_50'].item(),steps)

                # is the best model so far?
                if eval_result['map_50'].item() > self.best_map_50:
                    is_best = True
                    self.best_map_50 = eval_result['map_50'].item()

            # save a checkpoint for current epoch. If it is better than the best model so far, save it as the best model
            self._save_checkpoint_per_epoch(epoch,steps,is_best)  

    def _check_progress(self, 
                        steps, 
                        total_loss,
                        images_batch,
                        bboxes_batch,
                        labels_batch,
                        total_rpn_cls_loss,
                        total_rpn_reg_loss,
                        total_roi_cls_loss,
                        total_roi_reg_loss,
                        img_height,
                        img_width,
                        gt_bboxes,
                        gt_labels):

        self.writer.add_scalar('rpn/cls_loss',total_rpn_cls_loss.item(),steps)
        self.writer.add_scalar('rpn/reg_loss',total_rpn_reg_loss.item(),steps)
        self.writer.add_scalar('roi/cls_loss',total_roi_cls_loss.item(),steps)
        self.writer.add_scalar('roi/reg_loss',total_roi_reg_loss.item(),steps)
        self.writer.add_scalar('total_loss',total_loss.item(),steps)
        self.writer.add_scalar('lr',self.optimizer.param_groups[0]['lr'],steps)

        with torch.no_grad():
            predicted_labels_batch, predicted_scores_batch,predicted_bboxes_batch = self.faster_rcnn.predict(images_batch.float())
                        
            predicted_labels_for_img_0 = predicted_labels_batch[0]
            predicted_label_names_for_img_0 = []
            for label_index in predicted_labels_for_img_0:
                predicted_label_names_for_img_0.append(self.train_dataloader.dataset.get_label_names()[label_index.long().item()])

            if len(predicted_label_names_for_img_0) >0:
                label_names = [self.train_dataloader.dataset.get_label_names()[label_index] for label_index in labels_batch[0]] 
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
                map =self._evaluate_on_train_set(gt_bboxes, gt_labels, predicted_scores_for_img_0, predicted_labels_for_img_0, predicted_bboxes_for_img_0)
                self.writer.add_scalar('train/map',map['map'].item(),steps)
                self.writer.add_scalar('train/map_50',map['map_50'].item(),steps)

    def _resume(self):
        ckpt = load_checkpoint(self.checkpoint_path) # custom method for loading last checkpoint
        self.feature_extractor.load_state_dict(ckpt['feature_extractor_model'])
        self.rpn.load_state_dict(ckpt['rpn_model'])
        self.fast_rcnn.load_state_dict(ckpt['fast_rcnn_model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        steps = ckpt['steps']
        return steps,start_epoch
    
    def _save_checkpoint_per_epoch(self,epoch,steps,is_best):
        cpkt = {
                'feature_extractor_model':self.feature_extractor.state_dict(),
                'rpn_model': self.rpn.state_dict(),
                'fast_rcnn_model': self.fast_rcnn.state_dict(),
                'epoch': epoch,
                'steps': steps,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
                }
        
        save_checkpoint(cpkt, self.checkpoint_path, is_best=is_best)

    def _evaluate_on_train_set(self, 
                gt_bboxes:torch.Tensor, 
                gt_labels:torch.Tensor, 
                predicted_scores:torch.Tensor, 
                predicted_labels:torch.Tensor,
                predicted_bboxes:torch.Tensor)->float:
        """
        Evaluate the model on one single train image .

        Args:
            gt_bboxes: (N,4)
            gt_labels: (N,)
            predicted_scores: (N,)
            predicted_labels: (N,)
            predicted_bboxes: (N,4)

        Returns:
            dict containing

            - map: ``torch.Tensor``
            - map_50: ``torch.Tensor``
            - map_75: ``torch.Tensor``
            - map_small: ``torch.Tensor``
            - map_medium: ``torch.Tensor``
            - map_large: ``torch.Tensor``
            - mar_1: ``torch.Tensor``
            - mar_10: ``torch.Tensor``
            - mar_100: ``torch.Tensor``
            - mar_small: ``torch.Tensor``
            - mar_medium: ``torch.Tensor``
            - mar_large: ``torch.Tensor``
            - map_per_class: ``torch.Tensor`` (-1 if class metrics are disabled)
            - mar_100_per_class: ``torch.Tensor`` (-1 if class metrics are disabled)

        """
        preds = [dict(
                    # convert yxyx to xyxy
                    boxes = predicted_bboxes.cpu().index_select(1,torch.tensor([1,0,3,2])),
                    scores = predicted_scores.cpu(),
                    labels = predicted_labels.cpu(),
                    )]
        
        target = [dict(
                    boxes = gt_bboxes.cpu().index_select(1,torch.tensor([1,0,3,2])),
                    labels = gt_labels.cpu(),
                    )]  

        self.metric.update(preds,target)
        return self.metric.compute()




