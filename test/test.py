import os
import sys

current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('test')]
sys.path.append(work_folder+'src/algorithm')
sys.path.append(work_folder+'src/config')
sys.path.append(work_folder+'src/data')
sys.path.append(work_folder+'src/tool')

import unittest

import torch
from config import combine_configs
from voc_dataset import VOCDataset
from feature_extractor import FeatureExtractorFactory, VGG16FeatureExtractor
from rpn.anchor_creator import AnchorCreator
from rpn.anchor_target_creator import AnchorTargetCreator
from rpn.proposal_creator import ProposalCreator
from rpn.proposal_target_creator import ProposalTargetCreator
from rpn.region_proposal_network import RPN
from rpn.region_proposal_network_loss import RPNLoss
from fast_rcnn.fast_rcnn_loss import FastRCNNLoss
from rpn.region_proposal_network_trainer import RPNTrainer
from fast_rcnn.fast_rcnn_network import FastRCNN
from faster_rcnn.faster_rcnn_trainer import FasterRCNNTrainer
from faster_rcnn.faster_rcnn_network import FasterRCNN

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from location_utility import LocationUtility
from visual_tool import draw_img_bboxes_labels

IMG    =  torch.randn(1, 3, 800,800).float()
IMG_WIDTH = IMG.shape[-1]
IMG_HEIGHT = IMG.shape[-2]
BBOX   =  torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]])
LABELS =  torch.LongTensor([6, 8])  
FEATURE_STRIDE = 16 

FEATURE_HEIGHT = int(IMG_HEIGHT / FEATURE_STRIDE)
FEATURE_WIDTH  = int(IMG_WIDTH / FEATURE_STRIDE)

IN_CHANNEL = 4096
NUM_CLASSES = 21
ROI_SIZE = 7

config_path = work_folder+'src/config/train/experiments/exp03_config.yaml'
config = combine_configs(config_path)

@unittest.skip("passed")
class TestConfig(unittest.TestCase):
    def test_get_default_config(self) -> None:        
        print(config)
    

@unittest.skip("Passed")
class TestAnchorCreator(unittest.TestCase):
    def setUp(self) -> None:
        self.achor_creator = AnchorCreator(config)
    
    
    def test_anchor_base(self):
        self.assertEqual(self.achor_creator.anchor_base.shape,torch.Size([9, 4]))
    
    
    def test_anchor_creation(self):
        anchors = self.achor_creator.create(2,4)
        print(anchors.shape)
        print(anchors)

@unittest.skip("Passed")
class TestUtility(unittest.TestCase):
    def test_loc_transform(self):
        src_bbox = torch.tensor([[0, 0, 20, 10], [5, 5, 50, 10]])
        loc = torch.tensor([[0.1, 0.3, 0.8, 0.2], [0.3, 0.7, 0.4, 0.9]])
        dst_bbox = LocationUtility.offset2bbox(src_bbox, loc)

        locs_back = LocationUtility.bbox2offset(src_bbox, dst_bbox)   
        self.assertTrue(torch.allclose(loc, locs_back))

@unittest.skip("Passed")
class TestAnchorTargetCreator(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor_creator = AnchorCreator(config)
        self.anchor_target_creator = AnchorTargetCreator(config)
        self.feature_extractor = FeatureExtractorFactory.create_feature_extractor('vgg16')
    
    def test_anchor_target_creator_2226(self):
        self.voc_dataset = VOCDataset(config)
        image,bboxes,lables,diff,img_file= self.voc_dataset[2225]
        image = image.unsqueeze(0)
        feature = self.feature_extractor.predict(image.float())
        feature_height,feature_width = feature.shape[2:]
        anchors_of_img = self.anchor_creator.create(feature_height,feature_width)
        img_height,img_width = image.shape[2:]
        target_labels,target_locs = self.anchor_target_creator.create(anchors_of_img,bboxes,img_height,img_width)
        self.assertEqual((target_labels==1).nonzero().squeeze().shape,torch.Size([128]))
                    

    def test_anchor_target_creator(self):
        anchors_of_img = self.anchor_creator.create(FEATURE_HEIGHT,FEATURE_WIDTH)
        self.assertEqual(anchors_of_img.shape, torch.Size([FEATURE_WIDTH*FEATURE_HEIGHT*9, 4]))
        
        lables,locs = self.anchor_target_creator.create(anchors_of_img, BBOX,IMG_HEIGHT, IMG_WIDTH)
        
        if lables is not None:
            self.assertEqual(locs.shape, torch.Size([FEATURE_WIDTH*FEATURE_HEIGHT*9, 4]))
            self.assertEqual(lables.shape, torch.Size([FEATURE_WIDTH*FEATURE_HEIGHT*9]))
    
@unittest.skip("Passed")
class TestProposalCreator(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_extractor = FeatureExtractorFactory().create_feature_extractor("vgg16")
        self.anchor_creator = AnchorCreator(config)
        self.proposal_creator = ProposalCreator(config)
        self.rpn = RPN(config)
    
    def test_rpn(self):
        predcited_locs, predcited_scores = self.rpn.predict(self.feature_extractor(IMG))
        anchors_of_img = self.anchor_creator.create(FEATURE_HEIGHT,FEATURE_WIDTH)
        roi = self.proposal_creator.create(anchors_of_img, predcited_scores[0], predcited_locs[0],IMG_HEIGHT,IMG_WIDTH,FEATURE_HEIGHT,FEATURE_WIDTH)
        print(roi.shape)

@unittest.skip('passed')
class TestFeatureExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = FeatureExtractorFactory()
    
    @unittest.skip("passed")
    def test_vgg16_extractor(self):
        extractor = self.factory.create_feature_extractor('vgg16')
        features = extractor.predict(IMG)
        self.assertTrue(features.shape == torch.Size([1, 512, 50, 50]))   

    def test_pretrained_vgg16_extractor(self):
        extractor = self.factory.create_feature_extractor('pretrained_vgg16')
        features = extractor.predict(IMG)
        self.assertTrue(features.shape == torch.Size([1, 512, 50, 50]))


@unittest.skip('passed')
class TestRPN(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_extractor= FeatureExtractorFactory().create_feature_extractor('vgg16')
        self.feature = self.feature_extractor(IMG)
        self.rpn = RPN(config)
        #summary(self.rpn, (3, 800, 800),device='cpu')
        
    def test_rpn_forward(self):
        predicted_scores,predicted_locs = self.rpn.predict(self.feature)
        self.assertEqual(predicted_scores.shape, torch.Size([1, 18,50,50]))
        self.assertEqual(predicted_locs.shape,   torch.Size([1, 36,50,50]))
        
@unittest.skip('passed')
class TestRPNLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_extractor= FeatureExtractorFactory().create_feature_extractor('vgg16')
        self.feature = self.feature_extractor.predict(IMG)
        self.rpn = RPN(config)
        self.rpn_loss = RPNLoss(config)
        self.anchor_creator = AnchorCreator(config)
        
    def test_rpn_loss(self):
        predicted_scores,predicted_locs = self.rpn.predict(self.feature)
        anchors_of_image = self.anchor_creator.create(FEATURE_HEIGHT,FEATURE_WIDTH)
        cls_loss,reg_loss = self.rpn_loss(anchors_of_image,predicted_scores[0],predicted_locs[0],BBOX,IMG_HEIGHT,IMG_WIDTH)
        print(cls_loss)
        print(reg_loss)

@unittest.skip('passed')
class TestVOCDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.voc_dataset = VOCDataset(config)
        self.writer = SummaryWriter(config.LOG.LOG_DIR)

    def tearDown(self) -> None:
        self.writer.flush()
        self.writer.close()
        return super().tearDown()

    def test_voc_dataset(self):
        print(self.voc_dataset.__len__())
        
        samples = 1
        imgs=torch.zeros([samples,3,500,500])       
        for i in range(samples):
            image,bboxes,lables,diff,img_file= self.voc_dataset[1430]
            print(image.shape)
            print(bboxes.shape)
            print(lables.shape)
            print(diff.shape)
            print(img_file)
            
            lable_names = [VOCDataset.VOC_BBOX_LABEL_NAMES[i] for i in lables]
            img_and_bbox = draw_img_bboxes_labels(image=image, bboxes=bboxes,labels=lable_names)
            imgs[i,:,:,:] = img_and_bbox

        self.writer.add_images('image',imgs,) 

@unittest.skip('testing')
class TestRPNTrainer(unittest.TestCase):
    def setUp(self):
        self.voc_dataset = VOCDataset(config)
        self.writer = SummaryWriter(config.LOG.LOG_DIR)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.trainer = RPNTrainer(config,self.voc_dataset,writer=self.writer,device=device)
        
    def test_train(self):
        self.trainer.train()

@unittest.skip('tested')
class TestProposalCreator(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_extractor = FeatureExtractorFactory().create_feature_extractor('vgg16')
        self.rpn = RPN(config)
        self.proposal_creator = ProposalCreator(config)
        self.anchor_creator = AnchorCreator(config)

    def test_generate(self):
        feature= self.feature_extractor.predict(IMG)
        predicted_scores,predicted_locs= self.rpn.predict(feature)
        anchors_of_img = self.anchor_creator.create(FEATURE_HEIGHT,FEATURE_WIDTH)
        proposed_roi_bboxes =self.proposal_creator.create(anchors_of_img,predicted_scores[0],predicted_locs[0],IMG_HEIGHT,IMG_WIDTH,FEATURE_HEIGHT,FEATURE_WIDTH)
        print(proposed_roi_bboxes.shape)

@unittest.skip('passed')
class TestProposalTargetCreator(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_extractor = FeatureExtractorFactory().create_feature_extractor('vgg16')
        self.rpn = RPN(config)
        self.anchor_creator = AnchorCreator(config)
        self.proposal_creator = ProposalCreator(config)
        self.anchor_target_creator = ProposalTargetCreator(config)
    
    def test_generate(self):
        feature= self.feature_extractor(IMG)
        predicted_scores,predicted_locs  = self.rpn.predict(feature)   
        anchors_of_img = self.anchor_creator.create(FEATURE_HEIGHT,FEATURE_WIDTH)
        proposed_roi_bboxes =self.proposal_creator.create(anchors_of_img,predicted_scores[0],predicted_locs[0],IMG_HEIGHT,IMG_WIDTH,FEATURE_HEIGHT,FEATURE_WIDTH)
        roi,gt_roi_loc,gt_roi_label = self.anchor_target_creator.create(proposed_roi_bboxes,BBOX,LABELS)
        print(roi.shape)
        print(gt_roi_loc.shape)
        print(gt_roi_label)

@unittest.skip('passed')
class TestFastRCNN(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_extractor = FeatureExtractorFactory().create_feature_extractor('vgg16')
        self.rpn = RPN(config)
        self.fast_rcnn = FastRCNN(config)
        self.anchor_creator = AnchorCreator(config)
        self.proposal_creator = ProposalCreator(config)
        self.anchor_target_creator = ProposalTargetCreator(config)
        self.fast_rcnn_loss = FastRCNNLoss(config)
    
    def test_forward(self):
        feature= self.feature_extractor.predict(IMG)
        predicted_scores,predicted_locs  = self.rpn.predict(feature)
        anchors_of_img = self.anchor_creator.create(FEATURE_HEIGHT,FEATURE_WIDTH)
        
        proposed_roi_bboxes =self.proposal_creator.create(anchors_of_img,predicted_scores[0],predicted_locs[0],IMG_HEIGHT,IMG_WIDTH,FEATURE_HEIGHT,FEATURE_WIDTH)
        print('Proposed ROI BBOXES Size:{}'.format(proposed_roi_bboxes.shape))

        sampled_roi,gt_roi_label,gt_roi_loc= self.anchor_target_creator.create(proposed_roi_bboxes,BBOX,LABELS)
        print('Sampled ROI Size:{}'.format(sampled_roi.shape))
        print('GT ROI LOC Size:{}'.format(gt_roi_loc.shape))
        print('GT ROI LABEL Size:{}'.format(gt_roi_label.shape))

        
        predicted_roi_cls_score,predicted_roi_cls_loc = self.fast_rcnn.predict(feature[0],sampled_roi)

        print('Predicted ROI CLS LOC Size:{}'.format(predicted_roi_cls_loc.shape))
        print('Predicted ROI CLS SCORE Size:{}'.format(predicted_roi_cls_score.shape))

        cls_loss,reg_loss = self.fast_rcnn_loss(predicted_roi_cls_score,predicted_roi_cls_loc,gt_roi_label,gt_roi_loc)
        print(cls_loss)
        print(reg_loss)

@unittest.skip('passed')
class TestFasterRCNN(unittest.TestCase):
    def setUp(self) -> None:
        self.writer = SummaryWriter(config.LOG.LOG_DIR)
        self.faster_rcnn = FasterRCNN(config,writer=self.writer)

    def test_forward(self):
        bboxes,labels,scores = self.faster_rcnn.predict(IMG)

unittest.skip('passed')    
class TestFasterRCNNTrainer(unittest.TestCase):
    def setUp(self):
        self.voc_dataset = VOCDataset(config)
        self.writer = SummaryWriter(config.LOG.LOG_DIR)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.trainer = FasterRCNNTrainer(config,self.voc_dataset,writer=self.writer,device=device)
        
    def test_train(self):
        self.trainer.train()


if __name__ == "__main__":
    print("Running Faster_RCNN test:")
    unittest.main()
