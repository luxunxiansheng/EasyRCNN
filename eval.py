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

import sys
import os

work_folder= os.path.dirname(os.path.realpath(__file__))
sys.path.append(work_folder+'/src/algorithm')
sys.path.append(work_folder+'/src/config')
sys.path.append(work_folder+'/src/data')
sys.path.append(work_folder+'/src/tool')

import torch
from faster_rcnn.faster_rcnn_evaluator import FasterRCNNEvaluator
from voc_dataset import VOCDataset

from config import combine_configs

if __name__=="__main__":
    config_path = work_folder+'/src/config/eval/eval.yaml'
    config = combine_configs(config_path)
    voc_dataset = VOCDataset(config,split='test')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    evaluator = FasterRCNNEvaluator(config,voc_dataset,device)
    map = evaluator.evaluate()
    print("map:{}".format(map['map'].item()))
    print("map_50:{}".format(map['map_50'].item()))