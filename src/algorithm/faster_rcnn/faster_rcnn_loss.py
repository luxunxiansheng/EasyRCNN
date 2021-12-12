import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_rcnn.fast_rcnn_loss import FastRCNNLoss
from rpn.region_proposal_network_loss import RPNLoss


class FasterRCNNLoss(nn.AbstractModule):
    def __init__(self, config,device='cpu'):
        super().__init__()
        self.device = device
        self.rpn_loss = RPNLoss(config,device)
        self.fast_rcnn_loss = FastRCNNLoss(config,device)
    
    def forward(self,rpn_predicted_scores,
                    rpn_predicted_locs,
                    rpn_target_labels,
                    rpn_target_locs,
                    fast_rcnn_predicted_scores,
                    fast_rcnn_predicted_locs,
                    fast_rcnn_target_labels,
                    fast_rcnn_target_locs,):
        
        rpn_cls_loss=0.0,
        rpn_reg_loss=0.0
        if rpn_target_labels is not None:
            rpn_cls_loss,rpn_reg_loss = self.rpn_loss(rpn_predicted_scores,rpn_predicted_locs,rpn_target_labels,rpn_target_locs)

        fast_rcnn_cls_loss,fast_rcnn_reg_loss = self.fast_rcnn_loss(fast_rcnn_predicted_scores,fast_rcnn_predicted_locs,fast_rcnn_target_labels,fast_rcnn_target_locs)

        return rpn_cls_loss+rpn_reg_loss+fast_rcnn_cls_loss+fast_rcnn_reg_loss
    