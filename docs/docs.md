## Image classification using VGG and ResNet.

* To change the dataset for training and testing, change the parameter in the variable trainset, testset.
* To switch between 1 channel and 3 channel images change the `self.inchannel` variable in vgg.py and `self.conv1` in resnet.py
* The batch size of dataset can be change by varying the variable "batch_size".
* The number of iteration can be changed by changing the variable "epochs".