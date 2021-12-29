# Lesion Tracker: A tool for tracking lesions in medical images

It is important to monitor the treatment response in longitudinal studies and to keep track of the progression of the disease in clinical trials, especially for patients with advanced disease states, such as cancer. Medical images play a crucial role in this process. Accuately identifying lesions across serial imaing follow-up is a challenge. In this project,we manage to construct a deep learning framework from ground to build a toolbox for tracking lesions in medical images.

## Maintainers

* Bin.Li, github: [luxunxiansheng](https://github.com/luxunxiansheng)
  
</br>
</br>

## Table of contents

* [Introduction](#introduction)
* [Release Information](#release-information)

## Introduction

In this project, besides the algoirthms, we also focus on the following tasks:

1. ****Enginerring friendly**** :  
   ***1.1***  In the last decades, there are many good practices used in software industry. We try to follow these practices to make our code more readable, testable and maintainable.
   Test-driven development (TDD) is one of the best practices and used in this project.

   ***1.2*** We use pytorch as the underlying framework. To make the code clean and easy to follow, we don't bring numpy or scipy into this project. That being said, the basic data structure is torch' tensors. There is no need (we hope so )to convert numpy arrays to torch tensors or vice versa in this project.

   ***1.3*** we follow the guide to organize the project: <https://julienbeaulieu.github.io/2020/03/16/building-a-flexible-configuration-system-for-deep-learning-models/>

   ***1.4*** For pytorch, we follow  <https://github.com/IgorSusmelj/pytorch-styleguide> as a style guide

2. ****Open source dataset**** :

   ***2.1*** we use the open source dataset <https://www.nih.gov/news-events/news-releases/nih-clinical-center-releases-dataset-32000-ct-images> to train our models The dataset contains 32,000 CT images from the NIH Clinical Center.

3. ****Classic algorithms**** :
   
   ***3.1*** It is a common practice today to leverage the existing algorithm libaray to build a deep learning framework for medical image analysis.But in this project, we decided to build our own on some of the classic algorthims including Faster R-CNN, Mask R-CNN and so on. We believe this will give us a better understanding of the underlying algorithms in particular when we will develop our own toolbox.

## Release information

1. Version: 1.0.0 Date: 2020-12-25  
   We release the first version of the project. In this version, we implement the basic framework of the project inculding the following tasks:

* log with the tensorboard
* checkpoint
* metrics
* faster rcnn
* notes:
   1. It is anti-pattern to use the __Call__ method in model classe to call the forward method. A better way is to use a explicit predict method to call the forward method and  return the output. It is easier for the user to know it is a prediction method.
   2. Loss class shoudl provide an explicit method, say , compute(), to comupte the loss and return the result instead of using the __call__ method.  
