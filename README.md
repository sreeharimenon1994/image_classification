# Image classification

This code pattern demonstrates how images can be classified using Convolutional Neural Network (CNN). Two methods are used: VGG and ResNet.


## Cost and Accuracy 
The graph represent the values of both of `cost` and `accuracy` each epoch. The first is ResNet and the second is VGG using CIFAR10 dataset. 

![graph_resnet](/images/ResNet34_CIFAR10.jpg)
![graph_vgg](/images/VGG_CIFAR10.jpg)


## Installation

Anaconda:

    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

pip:

    pip install torch torchvision

Source: Follow instructions at this URL: https://github.com/pytorch/pytorch#from-source

## How to Use

To using this repo, some things you should to know:

* Compatible with both of CPU and GPU, this code can automatically train on CPU or GPU
* To execute run  `python vgg.py` or `python resnet.py` to perform image classification

## Documentation

You can find the API documentation on the pytorch website:

https://pytorch.org/docs/stable/index.html
http://pytorch.org/docs/master/torchvision/