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
from torch.functional import Tensor

class LocationUtility:
    @staticmethod
    def offset2bbox(src_anchor:Tensor, offset:Tensor) -> Tensor:
        """Decode bounding boxes from bounding box offsets and scales.
        Args:
            src_anchor (Tensor): source anchor, shape (n, 4)
            offset (Tensor): encoded offsets, shape (n, 4)
        
        Returns:
            Tensor: decoded bounding boxes, shape (n, 4)
        """

        if src_anchor.shape[0] == 0:
            return torch.zeros((0, 4), dtype=offset.dtype)

        src_height = src_anchor[:, 2] - src_anchor[:, 0]
        src_width =  src_anchor[:, 3] - src_anchor[:, 1]
        src_centor_y = src_anchor[:, 0] + 0.5 * src_height
        src_centor_x = src_anchor[:, 1] + 0.5 * src_width
        
        dy = offset[:, 0::4]
        dx = offset[:, 1::4]
        dh = offset[:, 2::4]
        dw = offset[:, 3::4]
    
        dst_centor_y = dy * src_height.unsqueeze(1) + src_centor_y.unsqueeze(1)
        dst_centor_x = dx * src_width.unsqueeze(1) + src_centor_x.unsqueeze(1)

        dst_h = torch.exp(dh) * src_height.unsqueeze(1)
        dst_w = torch.exp(dw) * src_width.unsqueeze(1)
        
        dst_bbox = torch.zeros(offset.shape, dtype=offset.dtype,device=offset.device)
        dst_bbox[:, 0::4] = dst_centor_y - 0.5 * dst_h
        dst_bbox[:, 1::4] = dst_centor_x - 0.5 * dst_w
        dst_bbox[:, 2::4] = dst_centor_y + 0.5 * dst_h
        dst_bbox[:, 3::4] = dst_centor_x + 0.5 * dst_w
        
        return dst_bbox

    @staticmethod
    def bbox2offset(source_bbox:Tensor, target_bbox:Tensor) -> Tensor:
        """Encode bounding boxes to bounding box offsets.
        Args:
            source_bbox (Tensor): source bounding boxes, shape (n, 4)
            target_bbox (Tensor): target bounding boxes, shape (n, 4)
        return:
            Tensor: encoded offsets, shape (n, 4)
        """
        source_roi_height = source_bbox[:, 2] - source_bbox[:, 0]
        source_roi_width  = source_bbox[:, 3] - source_bbox[:, 1]
        source_roi_ctr_y  = source_bbox[:, 0] + 0.5 * source_roi_height
        source_roi_ctr_x  = source_bbox[:, 1] + 0.5 * source_roi_width 
        
        target_height= target_bbox[:, 2] - target_bbox[:, 0]
        target_width = target_bbox[:, 3] - target_bbox[:, 1]
        target_ctr_y = target_bbox[:, 0] + 0.5 * target_height
        target_ctr_x = target_bbox[:, 1] + 0.5 * target_width
        
        eps = torch.tensor(torch.finfo().eps,device=source_bbox.device)
        source_roi_width = torch.maximum(source_roi_width, eps)
        source_roi_height = torch.maximum(source_roi_height, eps)
        
        dy = (target_ctr_y - source_roi_ctr_y) / source_roi_height
        dx = (target_ctr_x - source_roi_ctr_x) / source_roi_width
        dh = torch.log(target_height / source_roi_height)
        dw = torch.log(target_width / source_roi_width)
        
        offset = torch.vstack((dy, dx, dh, dw)).t()
        return offset




