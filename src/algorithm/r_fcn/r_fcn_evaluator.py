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

from r_fcn.r_fcn_network import RFCN

class RFCNEvaluator(object):
    def __init__(self,
                config:CfgNode,
                dataset:Dataset,
                device:Device ='cpu') -> None:
        
        self.config = config
        self.device = device
        
        self.dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=8)    
        self.eval_r_fcn = RFCN(config,device)

        self.metric = MAP()
        
    def evaluate(self,model_states):
        self.metric.reset()
        
        self.eval_r_fcn.eval()
        self.eval_r_fcn.load_state_dict(model_states)

        for _,(images_batch,bboxes_batch,labels_batch,_,img_id,_) in tqdm(enumerate(self.dataloader)):
            images_batch,bboxes_batch,labels_batch = images_batch.to(self.device),bboxes_batch.to(self.device),labels_batch.to(self.device)
            with torch.no_grad():
                predicted_bboxes_batch,predicted_labels_batch, predicted_scores_batch= self.eval_r_fcn.predict(images_batch.float())

            for img_idx in range(len(images_batch)):
                pred_bboxes= predicted_bboxes_batch[img_idx]
                pred_scores = predicted_scores_batch[img_idx]
                pred_labels = predicted_labels_batch[img_idx]
                gt_bboxes = bboxes_batch[img_idx]
                gt_labels = labels_batch[img_idx]

                single_image_predict = [dict(
                                            # convert yxyx to xyxy
                                            boxes = pred_bboxes[:,[1,0,3,2]].float(),
                                            scores = pred_scores,
                                            labels = pred_labels,
                                            )]
            
                single_image_gt = [dict(boxes = gt_bboxes[:,[1,0,3,2]],
                                        labels = gt_labels,
                                        )]

                self.metric.update(single_image_predict,single_image_gt)

        return self.metric.compute()