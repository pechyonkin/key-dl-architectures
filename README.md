# Key Deep Learning Architectures

This repository provides overview of some of the prominent neural network architectures. Reading through this guide and all supplement will help you develop understanding of the modern neural network architectures and main ideas behind them.

### List of Architectures in this guide

Sorted in chronological order. Click an architecture name to jump to corresponding part of the guide.

- [LeNet](#lenet-5-1998-paper-by-lecun-et-al), 1998
- AlexNet, 2012, [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) ![imagenet_winner 2012](https://img.shields.io/badge/imagenet_winner-2012-brightgreen.svg?style=plastic)
- ZFNet, 2013, [paper](https://arxiv.org/pdf/1311.2901v3.pdf) ![imagenet_winner 2012](https://img.shields.io/badge/imagenet_winner-2013-brightgreen.svg?style=plastic)
- GoogLeNet, 2014, [paper](https://arxiv.org/abs/1409.4842) ![imagenet_winner 2014](https://img.shields.io/badge/imagenet_winner-2014-brightgreen.svg?style=plastic)
- Inception, 2014, [paper](https://arxiv.org/pdf/1409.4842v1.pdf), "Going deeper with convolutions"
- VGG, 2014, [paper](https://arxiv.org/abs/1409.1556)
- InceptionV2, InceptionV3, 2015, [paper](https://arxiv.org/abs/1512.00567)
- ResNet, 2015, [paper](https://arxiv.org/abs/1512.03385) ![imagenet_winner 2015](https://img.shields.io/badge/imagenet_winner-2015-brightgreen.svg?style=plastic)
- InceptionV4, 2016, [paper](https://arxiv.org/abs/1602.07261)
- InceptionResNetV2, [paper](https://arxiv.org/abs/1602.07261) (same paper as InceptionV4)
- DenseNet, 2016, [paper](https://arxiv.org/abs/1608.06993)
- Xception, 2016, [paper](https://arxiv.org/abs/1610.02357)
- MobileNet, 2017, [paper](https://arxiv.org/abs/1704.04861)
- NASNet, 2017, [paper](https://arxiv.org/pdf/1707.07012.pdf) | [blogpost](https://ai.googleblog.com/2017/11/automl-for-large-scale-image.html)
- SE-ResNet, 2017, [paper](https://arxiv.org/pdf/1709.01507v1.pdf) ![imagenet_winner 2017](https://img.shields.io/badge/imagenet_winner-2017-brightgreen.svg?style=plastic)
- MobileNetV2, 2018, [paper](https://arxiv.org/abs/1801.04381)

---

- Faster R-CNN, 2015, [paper](https://arxiv.org/abs/1506.01497)

## LeNet-5 [1998, [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) by LeCun et al.]

![LeNet-5](/images/lenet-5.png)

**Main ideas**: local receptive fields, shared weights, spacial subsampling

**Why it is important**: LeNet-5 was used on large scale to automatically classify hand-written digits on bank cheques in the United States. This network is **the first convolutional neural network** (CNN). CNNs introduced in the paper are the foundation of modern state-of-the art deep learning. These networks are built upon 3 main ideas: local receptive fields, shared weights and spacial subsampling. Local receptive fields with shared weights are the essence of the convolutional layer and most [ALL?] architectures described below use convolutional layers in one form or another. 

**Brief description**: by modern standards, LeNet-5 is a very simple network. It only has 7 layers, among which there are 2 convolutional layers (C1 and C3), 2 sub-sampling layers (S2 and S4), and 2 fully connected layers (C5 and F6), that are followed by the output layer.

**Additional readings**:
- Understanding convolutions

## AlexNet [2012, [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) by Krizhevsky et al.] 

![xxx](/images/alexnet.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## ZFNet [2013, [paper](https://arxiv.org/pdf/1311.2901v3.pdf) by Zeiler et al.] 

![xxx](/images/zfnet.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## GoogLeNet [2014, [paper](https://arxiv.org/abs/1409.4842) by Szegedy et al.] 

![xxx](/images/googlenet.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## Inception [2014, [paper](https://arxiv.org/pdf/1409.4842v1.pdf) by Szegedy et al.] 

![xxx](/images/inception.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## VGG [2014, [paper](https://arxiv.org/abs/1409.1556) by Simonyan et al.] 

![xxx](/images/vgg.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## InceptionV2, InceptionV3 [2015, [paper](https://arxiv.org/abs/1512.00567) by Szegedy et al.] 

![xxx](/images/inceptionv2.png)

![xxx](/images/inceptionv3.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## ResNet [2015, [paper](https://arxiv.org/abs/1512.03385) by He et al.] 

![xxx](/images/resnet.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## InceptionV4, InceptionResNetV2 [2016, [paper](https://arxiv.org/abs/1602.07261) by Szegedy et al.] 

![xxx](/images/inceptionv4.png)

![xxx](/images/inceptionresnet.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## DenseNet [2016, [paper](https://arxiv.org/abs/1608.06993) by Huang et al.] 

![xxx](/images/densenet.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## Xception [2016, [paper](https://arxiv.org/abs/1610.02357) by Chollet] 

![xxx](/images/xception.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## MobileNet [2017, [paper](https://arxiv.org/abs/1704.04861) by Howard et al.] 

![xxx](/images/mobilenet.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## NASNet [2017, [paper](https://arxiv.org/pdf/1707.07012.pdf) by Zoph et al.] 

![xxx](/images/nasnet.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## SE-ResNet [2017, [paper](https://arxiv.org/pdf/1709.01507v1.pdf) by Hu et al.] 

![xxx](/images/se-resnet.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## MobileNetV2 [2018, [paper](https://arxiv.org/abs/1801.04381) by Sandler et al.] 

![xxx](/images/mobilenetv2.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:

## XXX [20xx, [paper]() by XXX et al.] 

![xxx](/images/xxx.png)

**Main ideas**: 

**Why it is important**: 

**Brief description**:

**Additional readings**:


[Badge generator](https://rozaxe.github.io/factory/)