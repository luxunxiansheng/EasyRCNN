import os
import xml.etree.ElementTree as ET

import torch
import torch.utils.data as data
from torchvision.io import read_image

class VOCDataset(data.Dataset):

    VOC_BBOX_LABEL_NAMES = (
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor')
    
    def __init__(self, 
                config,
                split='trainval',
                ):
        self.data_dir = config.VOC_DATASET.DATA_DIR
        self.use_difficult = config.VOC_DATASET.USE_DIFFICULT_LABEL
        self.ids = list()
        with open(os.path.join(self.data_dir, 'ImageSets', 'Main', split + '.txt')) as f:
            for line in f:
                self.ids.append(line.strip())

        self.label_names = VOCDataset.VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id_ = self.ids[index]
        annotation = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bboxs =     list()
        labels =    list()
        difficult = list()
        for obj in annotation.findall('object'):
            # when in not using difficult split, and the object is difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            boundingbox_annotation = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bboxs.append([int(boundingbox_annotation.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            labels.append(VOCDataset.VOC_BBOX_LABEL_NAMES.index(name))
        
        image_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        image = read_image(image_file)
        bboxs = torch.tensor(bboxs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        difficult = torch.tensor(difficult, dtype=torch.uint8)
        
        return image, bboxs,labels, difficult,image_file
