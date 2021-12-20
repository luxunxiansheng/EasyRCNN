import torch
from torch.types import Device
from yacs.config import CfgNode

class AnchorCreator:
    def __init__(self,
                config:CfgNode,
                device:Device='cpu'):

        self.anchor_ratios = torch.tensor(config.RPN.ANCHOR_CREATOR.ANCHOR_RATIOS)
        self.anchor_scales = torch.tensor(config.RPN.ANCHOR_CREATOR.ANCHOR_SCALES)
        self.feature_stride = config.RPN.ANCHOR_CREATOR.FEATURE_STRIDE
        self.device = device
        self.anchor_base = self._create_anchor_base()

    def create(self,
                feature_height:int,
                feature_width: int):

        """Generate anchor windows by enumerating aspect ratio and scales.
        
        Args:
            feature_height (int): feature height
            feature_width (int): feature width
        
        Returns:
            return anchor windows [n_anchors,4]      
        """


        shift_y = torch.arange(0, feature_height * self.feature_stride,self.feature_stride,device=self.device)
        shift_x = torch.arange(0, feature_width * self.feature_stride, self.feature_stride,device=self.device)
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing='xy')
        shift = torch.stack((shift_y.ravel(), shift_x.ravel(),shift_y.ravel(), shift_x.ravel()), dim=1)

        A = self.anchor_base.shape[0]
        K = shift.shape[0]
        anchors = self.anchor_base.reshape([1, A, 4]) + shift.reshape([1, K, 4]).permute(1, 0, 2)
        anchors = anchors.reshape([K * A, 4]).to(torch.float32)
        return anchors

    def _create_anchor_base(self):
        """Generate anchor base windows by enumerating aspect ratio and scales.
        
        Returns:
            return anchor base windows [n_anchors,4]
        
        """
        ctr_y = self.feature_stride / 2.
        ctr_x = self.feature_stride / 2.

        anchor_base = torch.zeros((len(self.anchor_ratios) * len(self.anchor_scales), 4), dtype=torch.float32,device=self.device)
        for i in range(len(self.anchor_ratios)):
            for j in range(len(self.anchor_scales)):
                h = self.feature_stride * self.anchor_scales[j] * torch.sqrt(self.anchor_ratios[i])
                w = self.feature_stride * self.anchor_scales[j] * torch.sqrt(1. /self.anchor_ratios[i])

                index = i * len(self.anchor_scales) + j

                anchor_base[index, 0] = ctr_y - h / 2.
                anchor_base[index, 1] = ctr_x - w / 2.
                anchor_base[index, 2] = ctr_y + h / 2.
                anchor_base[index, 3] = ctr_x + w / 2.

        return anchor_base
        