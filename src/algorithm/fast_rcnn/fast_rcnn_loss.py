import torch
import torch.nn as nn
import torch.nn.functional as F


class FastRCNNLoss(nn.Module):
    def __init__(self,config,device='cpu'):
        super().__init__()
        self.roi_sigma = config.FAST_RCNN.ROI_SIGMMA
        self.device = device
    
    def forward(self,predicted_scores,predicted_offsets,target_labels,target_offsets):
        classfication_loss = F.cross_entropy(predicted_scores,target_labels)

        n_sample = target_offsets.shape[0]

        positive_wieight = torch.zeros(target_offsets.shape).to(self.device)
        positive_wieight[(target_labels > 0).view(-1,1).expand_as(positive_wieight)] = 1

        predicted_offsets = predicted_offsets.contiguous().view(n_sample,-1,4)
        predicted_offsets = predicted_offsets[torch.arange(0,n_sample).long(),target_labels.long()]
    
        predicted_offsets = positive_wieight * predicted_offsets
        target_offsets    = positive_wieight * target_offsets

        regression_loss = self._soomth_l1_loss(predicted_offsets,target_offsets,self.roi_sigma)
        regression_loss = regression_loss /((target_labels>0).sum().float())

        return classfication_loss,regression_loss

    def compute(self,predicted_scores,predicted_offsets,target_labels,target_offsets):
        return self.forward(predicted_scores,predicted_offsets,target_labels,target_offsets)
    
    def _soomth_l1_loss(self, predicted_offsets, target_offsets,sigma):
        sigma2 = sigma**2
        diff = predicted_offsets - target_offsets
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1.0 / sigma2)).float()
        loss = flag * (sigma2 / 2.) * (diff ** 2) +(1 - flag) * (abs_diff - 0.5 / sigma2)
        return loss.sum() 