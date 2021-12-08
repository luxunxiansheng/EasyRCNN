import torch
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
    

def draw_img_bboxes_labels(image, bboxes, labels,resize_shape=[800,800]):
    colors = ["red", "green", "blue", "yellow","black","white","purple","orange","pink","brown","gray","cyan","magenta"]

    bboxes = torch.index_select(bboxes,1,torch.tensor([1,0,3,2]))
    img_and_bbox = draw_bounding_boxes(image,bboxes,colors=colors,labels=labels,width=2)
    # the orgianl draw_bounding_boxes function outputs an inverted image, so we need to reverse it
    transform = transforms.Compose([transforms.Resize(resize_shape),transforms.RandomInvert(1.0)])
    img_and_bbox = transform(img_and_bbox)
    return img_and_bbox