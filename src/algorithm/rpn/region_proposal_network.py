import torch.nn as nn
from common import CNNBlock, weights_normal_init


class RPN(nn.Module):
    def __init__(self,config):
        super().__init__()
    
        self.num_base_anchors = len(config.RPN.ANCHOR_CREATOR.ANCHOR_SCALES)**2
        self.conv1 = CNNBlock(config.RPN.FEATURE_CHANNELS, config.RPN.MID_CHANNELS, 3, same_padding=True)
        self.score_conv = CNNBlock(config.RPN.MID_CHANNELS, self.num_base_anchors*2, 1, relu=False, same_padding=False)
        self.bbox_conv = CNNBlock(config.RPN.MID_CHANNELS,  self.num_base_anchors*4, 1, relu=False, same_padding=False)

        weights_normal_init(self.conv1, dev=0.01)
        weights_normal_init(self.score_conv, dev=0.01)
        weights_normal_init(self.bbox_conv, dev=0.001)

    def forward(self,features):
        # [batch_size, middle_channels, feature_height, feature_width]
        hidden = self.conv1(features)  

        #-------------------- predict cls score ----------------------#
        #[batch_size, num_base_anchors * 2, feature_height,feature_width]
        predicted_scores = self.score_conv(hidden) 
        
        #-------------------- predict locs ----------------------#
        # [batch_size, num_base_anchors * 4, feature_height,feature_width]
        predicted_locs = self.bbox_conv(hidden) 

        return predicted_scores,predicted_locs
