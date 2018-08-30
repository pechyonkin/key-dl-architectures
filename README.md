# Key Deep Learning Architectures

This repository provides overview of some of the prominent neural network architectures. Reading through this guide and all supplement materials will help you develop understanding of the modern neural network architectures and main ideas behind them. Before starting, you should have some familiarity with the basics of neural networks, backpropagation algorithm and gradient descent.

### List of Architectures in this guide

Sorted in chronological order. Click an architecture name to jump to corresponding part of the guide.

- [LeNet](#lenet-5-1998-paper-by-lecun-et-al)
- [AlexNet](#alexnet-2012-paper-by-krizhevsky-et-al) ![imagenet_winner 2012](https://img.shields.io/badge/imagenet_winner-2012-brightgreen.svg?style=plastic)
- [ZFNet](#zfnet-2013-paper-by-zeiler-et-al), 2013 ![imagenet_winner 2012](https://img.shields.io/badge/imagenet_winner-2013-brightgreen.svg?style=plastic)
- [GoogLeNet](#googlenet-2014-paper-by-szegedy-et-al), 2014 ![imagenet_winner 2014](https://img.shields.io/badge/imagenet_winner-2014-brightgreen.svg?style=plastic)
- [Inception](#inception-2014-paper-by-szegedy-et-al), 2014
- [VGG](#vgg-2014-paper-by-simonyan-et-al), 2014
- [InceptionV2, InceptionV3](#inceptionv2-inceptionv3-2015-paper-by-szegedy-et-al), 2015
- [ResNet](#resnet-2015-paper-by-he-et-al), 2015 ![imagenet_winner 2015](https://img.shields.io/badge/imagenet_winner-2015-brightgreen.svg?style=plastic)
- [InceptionV4, InceptionResNetV2](#inceptionv4-inceptionresnetv2-2016-paper-by-szegedy-et-al), 2016
- [DenseNet](#densenet-2016-paper-by-huang-et-al), 2016
- [Xception](#xception-2016-paper-by-chollet), 2016
- [MobileNet](#mobilenet-2017-paper-by-howard-et-al), 2017
- [NASNet](#nasnet-2017-paper-by-zoph-et-al), 2017
- [SE-ResNet](#se-resnet-2017-paper-by-hu-et-al), 2017 ![imagenet_winner 2017](https://img.shields.io/badge/imagenet_winner-2017-brightgreen.svg?style=plastic)
- [MobileNetV2](#mobilenetv2-2018-paper-by-sandler-et-al), 2018

## LeNet-5 [1998, [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) by LeCun et al.]

![LeNet-5](/images/lenet-5.png)

### **Main ideas**

convolution, local receptive fields, shared weights, spacial subsampling

### **Why it is important**

LeNet-5 was used on large scale to automatically classify hand-written digits on bank cheques in the United States. This network is **the first convolutional neural network** (CNN). CNNs introduced in the paper are the foundation of modern state-of-the art deep learning. These networks are built upon 3 main ideas: local receptive fields, shared weights and spacial subsampling. Local receptive fields with shared weights are the essence of the convolutional layer and most [ALL?] architectures described below use convolutional layers in one form or another. 

Another reason why LeNet is an important architecture is that before it was invented, character recognition had been done mostly by using feature engineering by hand, followed by a machine learning model to learn to classify hand engineered features. LeNet made hand engineering features redundant, because the network learns the best internal representation from raw images automatically.  

### **Brief description**

By modern standards, LeNet-5 is a very simple network. It only has 7 layers, among which there are 3 convolutional layers (C1, C3 and C5), 2 sub-sampling (pooling) layers (S2 and S4), and 1 fully connected layer (F6), that are followed by the output layer. Convolutional layers use 5 by 5 convolutions with stride 1. Sub-sampling layers are 2 by 2 average pooling layers. Tanh sigmoid activations are used throughout the network. There are several interesting architectural choices that were made in LeNet-5 that are not very common in the modern era of deep learning. 

First, individual convolutional kernels in the layer C3 do not use all of the features produced by the layer S2, which is very unusual by today's standard. One reason for that is to made the network less computationally demanding. The other reason was to make convolutional kernels learn different patterns. This makes perfect sense: if different kernels receive different inputs, they will learn different patterns.

Second, the output layer uses 10 Euclidean Radial Basis Function neurons that compute L2 distance between the input vector of dimension 84 and **manually predefined weights vectors** of the same dimension. The number 84 comes from the fact that essentially the weights represent a 7x12 binary mask, one for each digit. This forces network to transform input image into an internal representation that will make outputs of layer F6 as close as possible to hand-coded weights of the 10 neurons of the output layer.

LeNet-5 was able to achieve error rate below 1% on the MNIST data set, which was very close to the state of the art at the time (which was produced by a boosted ensemble of three LeNet-4 networks).

### **Additional readings**

- [Understanding Neural Networks Using Excel](https://towardsdatascience.com/understanding-convolutions-using-excel-886ca0a964b7)
- [Convolutional Neural Networks (CNNs / ConvNets)](https://cs231n.github.io/convolutional-networks/)
- Paper: "[A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf)"
- [Visual explanation](http://setosa.io/ev/image-kernels/) of convolution kernels (which are also used also in image processing)
- [Convolution animations](https://github.com/vdumoulin/conv_arithmetic) GIFs
- Probability concepts explained: [Maximum likelihood estimation](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1)

## AlexNet [2012, [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) by Krizhevsky et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/alexnet-arch.png" width="600">
  AlexNet architecture
</p>

### **Main ideas**

ReLU nonlinearity, training on multiple GPUs, local response normalization, overlapping pooling, data augmentation, dropout

### **Why it is important**

AlexNet won the ImageNet competition in 2012 by a large margin. It was the biggest network at the time. The network demonstrated the potential of training large neural networks quickly on massive datasets using widely available gaming GPUs; before that neural networks had been trained mainly by using CPUs. AlexNet also used novel ReLU activation, data augmentation, dropout and local response normalization. All of these allowed to achieve state-of-the art performance in object recognition in 2012.

### **Brief description**

#### ReLU nonlinearity
ReLU is a so-called *non-saturating activation*. This

#### Trainig on multiple GPUs

#### Local responce normaliztion


<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/alexnet-norm-formula.png" width="600">
  Local response normalization formula
</p>

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/alexnet-norm-excel.png" width="300">
  An example of local response normalization
</p>

#### Overlapping pooling

#### Data augmentation

#### Dropout

#### Architecture

### **Additional readings**

- Paper: [Rectified Linear Units Improve Restricted Boltzmann Machines](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.6419&rep=rep1&type=pdf)
- Paper: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)
- Quora: [Why are GPUs well-suited to deep learning?](https://www.quora.com/Why-are-GPUs-well-suited-to-deep-learning)
- [Why are GPUs necessary for training Deep Learning models?](https://www.analyticsvidhya.com/blog/2017/05/gpus-necessary-for-deep-learning/)
- [Data Augmentation | How to use Deep Learning when you have Limited Data ](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced)

## ZFNet [2013, [paper](https://arxiv.org/pdf/1311.2901v3.pdf) by Zeiler et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/zfnet.png" width="600">
  ZFNet architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## GoogLeNet [2014, [paper](https://arxiv.org/abs/1409.4842) by Szegedy et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/googlenet.png" width="600">
  GoogLeNet architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## Inception [2014, [paper](https://arxiv.org/pdf/1409.4842v1.pdf) by Szegedy et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/inception.png" width="600">
  Inception architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## VGG [2014, [paper](https://arxiv.org/abs/1409.1556) by Simonyan et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/vgg.png" width="600">
  VGG architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## InceptionV2, InceptionV3 [2015, [paper](https://arxiv.org/abs/1512.00567) by Szegedy et al.]

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/inceptionv2.png" width="600">
  InceptionV2 architecture
</p>

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/inceptionv3.png" width="600">
  InceptionV3 architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## ResNet [2015, [paper](https://arxiv.org/abs/1512.03385) by He et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/resnet.png" width="600">
  ResNet architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## InceptionV4, InceptionResNetV2 [2016, [paper](https://arxiv.org/abs/1602.07261) by Szegedy et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/inceptionv4.png" width="600">
  InceptionV4 architecture
</p>

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/inceptionresnet.png" width="600">
  InceptionResNetV2 architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## DenseNet [2016, [paper](https://arxiv.org/abs/1608.06993) by Huang et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/densenet.png" width="600">
  DenseNet architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## Xception [2016, [paper](https://arxiv.org/abs/1610.02357) by Chollet] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/xception.png" width="600">
  Xception architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## MobileNet [2017, [paper](https://arxiv.org/abs/1704.04861) by Howard et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/mobilenet.png" width="600">
  MobileNet architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## NASNet [2017, [blogpost](https://ai.googleblog.com/2017/11/automl-for-large-scale-image.html) | [paper](https://arxiv.org/pdf/1707.07012.pdf) by Zoph et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/nasnet.png" width="600">
  NASNet architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## SE-ResNet [2017, [paper](https://arxiv.org/pdf/1709.01507v1.pdf) by Hu et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/se-resnet.png" width="600">
  SE-ResNet architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## MobileNetV2 [2018, [paper](https://arxiv.org/abs/1801.04381) by Sandler et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/mobilenetv2.png" width="600">
  MobileNetV2 architecture
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**

## XXX [20xx, [paper]() by XXX et al.] 

<p align="center">
  <img src="https://github.com/pechyonkin/key-dl-architectures/blob/master/images/xxx.png" width="600">
  xxx
</p>

### **Main ideas**

### **Why it is important**

### **Brief description**

### **Additional readings**


[Badge generator](https://rozaxe.github.io/factory/)