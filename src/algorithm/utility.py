import torch
from torch.functional import Tensor


class Utility:
    @staticmethod
    def loc2bbox(src_anchor:Tensor, loc:Tensor) -> Tensor:
        """Decode bounding boxes from bounding box offsets and scales."""

        if src_anchor.shape[0] == 0:
            return torch.zeros((0, 4), dtype=loc.dtype)

        src_height = src_anchor[:, 2] - src_anchor[:, 0]
        src_width =  src_anchor[:, 3] - src_anchor[:, 1]
        src_centor_y = src_anchor[:, 0] + 0.5 * src_height
        src_centor_x = src_anchor[:, 1] + 0.5 * src_width
        
        dy = loc[:, 0::4]
        dx = loc[:, 1::4]
        dh = loc[:, 2::4]
        dw = loc[:, 3::4]
    
        centor_y = dy * src_height.unsqueeze(1) + src_centor_y.unsqueeze(1)
        centor_x = dx * src_width.unsqueeze(1) + src_centor_x.unsqueeze(1)

        w = torch.exp(dw) * src_width.unsqueeze(1)
        h = torch.exp(dh) * src_height.unsqueeze(1)
        
        dst_bbox = torch.zeros(loc.shape, dtype=loc.dtype,device=loc.device)
        dst_bbox[:, 0::4] = centor_y - 0.5 * h
        dst_bbox[:, 1::4] = centor_x - 0.5 * w
        dst_bbox[:, 2::4] = centor_y + 0.5 * h
        dst_bbox[:, 3::4] = centor_x + 0.5 * w
        
        return dst_bbox

    @staticmethod
    def bbox2loc(source_bbox:Tensor, target_bbox:Tensor) -> Tensor:
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
        
        loc = torch.vstack((dy, dx, dh, dw)).t()
        return loc




