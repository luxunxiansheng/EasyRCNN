import os
import xml.etree.ElementTree as ET
from albumentations.pytorch.transforms import ToTensor

import torch
import torch.utils.data as data
from torchvision.io import read_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2


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
                augumented=False,
                ):
        self.data_dir = config.VOC_DATASET.DATA_DIR
        self.use_difficult = config.VOC_DATASET.USE_DIFFICULT_LABEL
        self.ids = list()
        with open(os.path.join(self.data_dir, 'ImageSets', 'Main', split + '.txt')) as f:
            for line in f:
                self.ids.append(line.strip())

        self.label_names = VOCDataset.VOC_BBOX_LABEL_NAMES

        self.augmented = augumented

        self.transforms = A.Compose([A.HorizontalFlip(p=0.6),
                                    A.VerticalFlip(p=0.3),
                                    ],
                                    bbox_params=A.BboxParams(format='pascal_voc',
                                                            label_fields=['category_id']))
        self.toTensor = ToTensorV2()
    
    def get_label_names(self):
        return self.label_names

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id_ = self.ids[index]
        annotation = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bboxes =     list()
        category_id =    list()
        difficult = list()
        for obj in annotation.findall('object'):
            # when in not using difficult split, and the object is difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            boundingbox_annotation = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bboxes.append([int(boundingbox_annotation.find(tag).text) - 1 for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            category_id.append(VOCDataset.VOC_BBOX_LABEL_NAMES.index(name))
        
        image_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        # HWC
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmented:
            augmented = self.transforms(image=image, bboxes=bboxes, category_id=category_id)
            image = augmented['image']
            bboxes = augmented['bboxes']
            category_id = augmented['category_id']
        
        # HWC->CHW
        image = self.toTensor(image=image)['image']
        
        # convert from xyxy to yxyx 
        bboxes = torch.tensor(bboxes,dtype=torch.float32).index_select(dim=1, index=torch.tensor([1,0,3,2]))     
        category_id = torch.tensor(category_id,dtype=torch.long)
        difficult = torch.tensor(difficult, dtype=torch.uint8)
        
        return image, bboxes,category_id, difficult,image_file
