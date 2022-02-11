<h1 align="center">
  <br>
  3D bounding box detection using deep learning and geomerty
</h1>
<div align="center">
  <h4>
    <a href="#introduction">Introduction</a> |
    <a href="#demo">Demo</a> |
    <a href="#2d-Object-Detection">2D Object Detection</a> |
    <a href="#3d-object-detection">3D Object Detection</a> |
    <a href="#contribution">Contribution</a> |
    <a href="#references">References</a>
  </h4>
</div>
<br>

## Introduction
The primary purpose of this project is to implement the 3D object detection pipeline introduced in "[3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496v1)" paper for detecting four classes: Car, Truck, Pedestrian and Cyclist.

The paper mainly focuses on the 3D bounding box estimation from given 2D detections. However, we implemented both 2D and 3D detection parts in this project.

This repository is organized to make it reproducible, and two notebooks and the necessary configuration files are made available.
The first Notebook is [2d-bounding-box](./2d-bounding-box.ipynb) dedicated to using transfer learning to create a 2D object detection on the Kitti Dataset. Tensorflow Object detection framework was used in this process. It includes: 
1. Data preparation
2. Model Training
3. Model evaluation
4. Model saving

The second Notebook is [3d-bounding-box](./3d-bounding-box.ipynb) where we implemented the 3D bounding box detector using Tensorflow.
The Notebook includes the following:
1. Data Preparation
2. 3D model implementation
3. Input and Output Precessing
4. Using both the 2D and 3D models for 3D predictions.
5. Visually evaluating the 3D model's performance.
6. Measuring dimension accuracy*.

*\** Other estimations are evaluated before and after training models (2D and 3D). However, since the dimension vector is regressed through geometry constraints, it is evaluated separately. 

## Demo
The following images presents diffrent senarious and objects. At the left presented the result of 2D bounding box detection and on the right presented results of 3D bounding box detection based on the 2D detection result.
![](test_images/Untitled-2.png)

And this is a video composed of diffrent sequences that shows the prediction results on videos.

https://drive.google.com/file/d/14NsNLniK3n9FU8o_lDs-N5S0KtqVG0W6/view?usp=sharing

## 2D Object Detection
The purpose of 2D detection is after receiving an image; we need to detect Cars, Trucks, Pedestrians and Cyclists with 2D bounding boxes. We also need to classify the resulting patches as when we estimate the object dimensions later, we will require those classes.

Since 2D object detection is an already well-addressed problem, we keep things simple. Thus, we used high-level API (TensorFlow object detection API) and pre-trained models.

We have used Faster RCNN checkpoints [faster_rcnn_resnet101_kitti](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) from Tensorflow 1.x model zoo. The model is pre-trained on Kitty 2D objects dataset, but unfortunately, it does not detect all the classes. Having said that, We had to finetune the model to include all the classes that we needed, and the training process was straightforward.
The following table summarizes the results after finetuning.
+ Train size: 6981 images.
+ Test size: 500 images.
+ IoU: 0.5

| Model | Previous Dataset | Input Dimensions | Iterations | val AP Car | val AP Cyclist | val AP Pedestrian | val AP Truck| mAP |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Faster RCNN (Resnet 101) | KITTI | 300\*993\*3 | 25000| 0.8590 | 0.6518 | 0.6079 | 0.7699 |0.722|

Training with the TensorFlow Object Detection API requires fixing some paths and training configurations. The [pipeline.config](./2d_model/pipeline.config) file for Faster RCNN is included, and you can use it straight away as we tried to find the best configuration possible. Please make sure to modify the data paths inside that file before using it.

The complete workflow, such as setting up the environment, creating TFRecords, extracting frozen graphs, and evaluating the test set, is also provided in the 2d-object-detection notebook. We highly recommend using Google Colab and Google drive for training and hosting the data. The environment is slightly sensible and requires specific packages versions and a Linux environment.

## 3D Object Detection

For implementing this 3D object detection, We used some open-source repository as our baseline and guide. We focused on updating the code to make it compatible with the newer versions of TensorFlow and, more importantly, improve the model's accuracy. We also prepared well-documented notebooks to simplify understanding the model architecture and the modelling process and add many useful functions for visualization and testing purposes.
The final results of the 3D detection are presented in the table below:
+ Train size: 19308 patches extracted directly from the dataset.
+ Test size: 3408 patches extracted directly from the dataset.
+ Confidence error metric: MSE
+ Dimension error metric: MSE
+ Angle error metric: 1 - cos(|pred| - true) (ideal would be 1 - cos(0) = 0)
  
| Feature extractor | Previous Dataset | Input Dimensions | Epochs | # Parameters | Confidence Error| Angle error | Dimension Error | val Confidence Error | val Angle Error | val Dimension Error
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |----- | ----- | ----- |
| Movilenetv2 | Imagenet | 224\*224\*3 | 50 with early stopping at epoch 28| 66,490,453 | 8.7705e-04 | 0.2608 | 0.0391 |0.0135 |0.2672 |0.0531|


The following table present the Translation vector estimation accuracy.

+ Error metric: L2_norm(true - pred)
+ Normalized Error metric: L2_norm( (true - pred) / |true| )

| Max Truncation | Max Occlusion | Min 2D Bbox Width | Min 2D Bbox Height | Final Sample Count | Average Error | Average Normalized Error * |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0.15 | 1 | 60 | 60 | 967 | 1.8176 | 0.1151 |
| 0.40 | 1 | 40 | 40 | 1788 | 2.4165 | 0.1168 |

## Contribution
<a href="https://github.com/GhaziXX/3d-bounding-box-detection/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=GhaziXX/3d-bounding-box-detection" />
</a>

## References
#### Papers and Datasets
- [3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) and their [2D object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
#### Codes and Tools
- [https://github.com/smallcorgi/3D-Deepbox](github.com/smallcorgi/3D-Deepbox)
- [https://github.com/cersar/3D_detection](github.com/cersar/3D_detection)
- [Tensorflow Object Detection API](github.com/tensorflow/models)
