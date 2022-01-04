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
        
        pretrained_model = vgg16(pretrained=True)
        feature_layer = list(pretrained_model.features)[:30]

        assert feature_layer[0].in_channels == 3
        assert feature_layer[28].out_channels == 512

        # freeze top4 conv
        for layer in feature_layer[:10]:
            for p in layer.parameters():
                p.requires_grad = False 
                
        self.model = nn.Sequential(*feature_layer)
    
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
        
        
    
