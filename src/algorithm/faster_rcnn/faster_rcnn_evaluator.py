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
import torch
from torch.types import Device
from torch.utils.data.dataloader import DataLoader
from torchmetrics.detection.map import MAP
from yacs.config import CfgNode 

from faster_rcnn.faster_rcnn_network import FasterRCNN

class FasterRCNNEvaluator(object):
    def __init__(self,
                config:CfgNode,
                dataset:Dataset,
                faster_rcnn:FasterRCNN,
                device:Device='cpu') -> None:

        self.config = config
        self.device = device
        
        self.dataloader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=8)    
        self.faster_rcnn = faster_rcnn
        
        self.metric = MAP()

    def evaluate(self):
        preds = list()
        target = list()
        
        for _,(images_batch,bboxes_batch,labels_batch,_,img_file) in tqdm(enumerate(self.dataloader)):
            images_batch,bboxes_batch,labels_batch = images_batch.to(self.device),bboxes_batch.to(self.device),labels_batch.to(self.device)
            with torch.no_grad():
                predicted_labels_batch, predicted_scores_batch,predicted_bboxes_batch = self.faster_rcnn.predict(images_batch.float())

            for img_idx in range(len(images_batch)):
                pred_bboxes= predicted_bboxes_batch[img_idx]
                pred_scores = predicted_scores_batch[img_idx]
                pred_labels = predicted_labels_batch[img_idx]
                gt_bboxes = bboxes_batch[img_idx]
                gt_labels = labels_batch[img_idx]

                single_image_predict_dict = dict(
                                            # convert yxyx to xyxy
                                            boxes = pred_bboxes.cpu().index_select(1,torch.tensor([1,0,3,2])),
                                            scores = pred_scores.cpu(),
                                            labels = pred_labels.cpu(),
                                            )
                preds.append(single_image_predict_dict)

                single_image_gt_dict = dict(
                                            boxes = gt_bboxes.cpu().index_select(1,torch.tensor([1,0,3,2])),
                                            labels = gt_labels.cpu(),
                                            )

                target.append(single_image_gt_dict)

        self.metric.update(preds,target)
        return self.metric.compute()
                        
