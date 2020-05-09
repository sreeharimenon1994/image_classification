# Image classification

Two methods are used: VGG and ResNet.

## Installation

Anaconda:

    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

pip:

    pip install torch torchvision

## Cost and Accuracy 
graph represent the values of both of `cost` and `accuracy` each epoch

![graph_resnet](/images/ResNet_CIFAR10.jpg)
![graph_vgg](/images/VGG_CIFAR10.jpg)

## How to Use

To using this repo, some things you should to know:

* Compatible with both of CPU and GPU, this code can automatically train on CPU or GPU
* To execute run  `python vgg.py` or `python resnet.py` to perform image classification

## Documentation

You can find the API documentation on the pytorch website: 
https://pytorch.org/docs/stable/index.html
http://pytorch.org/docs/master/torchvision/