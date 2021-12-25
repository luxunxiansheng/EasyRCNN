# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /

import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision import transforms as T

from common import CNNBlock

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, 
                img_channels:int =3, 
                feature_channels:int =512, 
                bn:bool=False):
        """
            Args:
                img_channels (int): number of channels of input image
                feature_channels (int): number of channels of feature maps
                bn (bool): whether to use batch normalization
        
        """

        super().__init__()
        
        self.img_channels =  img_channels
        self.feature_channels = feature_channels
        
        self.conv1 = nn.Sequential( CNNBlock(img_channels, 64, 3, same_padding=True, bn=bn),
                                    CNNBlock(64, 64, 3, same_padding=True, bn=bn),
                                    nn.MaxPool2d(2))
        self.conv2 = nn.Sequential( CNNBlock(64, 128, 3, same_padding=True, bn=bn),
                                    CNNBlock(128, 128, 3, same_padding=True, bn=bn),
                                    nn.MaxPool2d(2))

        self.set_trainable(self.conv1, requires_grad=False)
        self.set_trainable(self.conv2, requires_grad=False)

        self.conv3 = nn.Sequential( CNNBlock(128, 256, 3, same_padding=True, bn=bn),
                                    CNNBlock(256, 256, 3, same_padding=True, bn=bn),
                                    CNNBlock(256, 256, 3, same_padding=True, bn=bn),
                                    nn.MaxPool2d(2))
        self.conv4 = nn.Sequential( CNNBlock(256, 512, 3, same_padding=True, bn=bn),
                                    CNNBlock(512, 512, 3, same_padding=True, bn=bn),
                                    CNNBlock(512, 512, 3, same_padding=True, bn=bn),
                                    nn.MaxPool2d(2))
        self.conv5 = nn.Sequential( CNNBlock(512, 512, 3, same_padding=True, bn=bn),
                                    CNNBlock(512, 512, 3, same_padding=True, bn=bn),
                                    CNNBlock(512, feature_channels, 3, same_padding=True, bn=bn))

    def forward(self, im_data:torch.Tensor)->torch.Tensor:
        """extract feature maps

        Args:
            im_data (torch.Tensor): shape = (batch_size, img_channel, img_size, img_size)

        Returns:
            torch.Tensor: shape = (batch_size, feature_channels, feature_height, feature_width)
        """

        assert im_data.size(1) == self.img_channels
        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def predict(self,im_data:torch.Tensor)->torch.Tensor:
        """
            A explicit function to predict the feature maps by calling forward function.        
        """

        return self.forward(im_data)


    @staticmethod
    def set_trainable( model, requires_grad):
        for p in model.parameters():
            p.requires_grad = requires_grad


class PretrainedVGG16FeatureExtractor(nn.Module):
    def __init__(self):
        """
            Args:
                img_channels (int): number of channels of input image
                feature_channels (int): number of channels of feature maps
                bn (bool): whether to use batch normalization
        
        """

        super().__init__()
        
        self.model = vgg16(pretrained=True)

        feature_layer = list(self.model.features)[:30]

        assert feature_layer[0].in_channels == 3
        assert feature_layer[28].out_channels == 512

        # freeze top4 conv
        for layer in feature_layer[:10]:
            for p in layer.parameters():
                p.requires_grad = False 
                
        self.feature_layer = nn.Sequential(*feature_layer)
    
    def predict(self,im_data:torch.Tensor)->torch.Tensor:
        """
            A explicit function to predict the feature maps by calling forward function.        
        """

        return self.forward(im_data)

    def forward(self, im_data:torch.Tensor)->torch.Tensor:
        """extract feature maps

        Args:
            im_data (torch.Tensor): shape = (batch_size, img_channel, img_size, img_size)

        Returns:
            torch.Tensor: shape = (batch_size, feature_channels, feature_height, feature_width)
        """         
        transform=T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])    
        im_data = transform(im_data/255.0)
        x = self.feature_layer(im_data)
        return x
        

class FeatureExtractorFactory:
    @staticmethod
    def create_feature_extractor(feature_extractor_type: str, **kwargs) -> torch.nn.Module:
        if feature_extractor_type == 'vgg16':
            return VGG16FeatureExtractor(**kwargs)
        elif feature_extractor_type == 'pretrained_vgg16':
            return PretrainedVGG16FeatureExtractor(**kwargs)
        else:
            raise ValueError('Unknown feature extractor type: {}'.format(feature_extractor_type))
        
        
    
