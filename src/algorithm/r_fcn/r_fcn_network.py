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

from torch import nn

class RFCN(nn.module):
    """
    R-FCN: Object Detection via Region-based Fully Convolutional Networks
    By Jifeng Dai, Yi Li, Kaiming He, Jian Sun

    Pseudo code:
    _________________________________________________________
    feature_maps = process(image)
    ROIs = region_proposal(feature_maps)         
    score_maps = compute_score_map(feature_maps)
    for ROI in ROIs
        V = region_roi_pool(score_maps, ROI)     
        class_scores, box = average(V)                  
        class_probabilities = softmax(class_scores) 
    _________________________________________________________
    """

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x

    def predict(self, x):
        return self.forward(x)