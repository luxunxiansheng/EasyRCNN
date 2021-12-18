import torch
import torch.nn as nn

from common import CNNBlock

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, img_channels=3, feature_channels=512, bn=False):
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

    def forward(self, im_data):
        assert im_data.size(1) == self.img_channels
        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def predict(self,im_data):
        return self.forward(im_data)


    @staticmethod
    def set_trainable( model, requires_grad):
        for p in model.parameters():
            p.requires_grad = requires_grad


class FeatureExtractorFactory:
    @staticmethod
    def create_feature_extractor(feature_extractor_type: str, **kwargs) -> torch.nn.Module:
        if feature_extractor_type == 'vgg16':
            return VGG16FeatureExtractor(**kwargs)
        else:
            raise ValueError('Unknown feature extractor type: {}'.format(feature_extractor_type))
        
        
    
