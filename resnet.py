import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from google.colab import files

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 50
image_size = 32
PREPROCESS = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

batch_size = 128
trainset = torchvision.datasets.CIFAR10(root = ".", train = True, download = True, transform = PREPROCESS)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root = ".", train = False, download = True, transform = PREPROCESS)
test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle=True)

train_dataset, train_dataset_total = train_loader, len(trainset)
test_dataset, test_dataset_total = test_loader, len(testset)

class DefaultBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(DefaultBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                        nn.BatchNorm2d(outchannel))
        self.conv2 = nn.Sequential(
                        nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(outchannel))
        
        self.skip = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.skip = nn.Sequential(
                nn.Conv2d(inchannel,  outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel))

    def forward(self, X):
        out = torch.relu(self.conv1(X))
        out = self.conv2(out)
        out += self.skip(X)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, DefaultBlock, num_classes):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.layer1 = self.make_layer(DefaultBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(DefaultBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(DefaultBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(DefaultBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


resnet = ResNet(DefaultBlock, num_classes = 10)
if torch.cuda.is_available():
    resnet.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr = 0.001)
train_accuracy, test_accuracy = [], []

for epoch in range(epochs):
    print("\nEpoch:", epoch)
    correct = 0
    resnet.train()
    for (images,target) in train_dataset:
        images = images.to(device)
        target = target.to(device)
        out = resnet(images)
        loss = criterion(out,target)

        # Back-propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out.data,1)
        correct += (pred == target).sum().item()

    train_accuracy.append((correct/train_dataset_total) * 100)
    print(f"Accuracy Train: {train_accuracy[-1]}")

    resnet.eval()
    correct_test = 0
    for (images_test, target_test) in test_dataset:
        images_test = images_test.to(device)
        target_test = target_test.to(device)

        out_test = resnet(images_test)
        _, pred_test = torch.max(out_test.data, 1)
        correct_test += (pred_test == target_test).sum().item()
    test_accuracy.append((correct_test/test_dataset_total) * 100)
    print(f"Accuracy Test: {test_accuracy[-1]}")

plt.plot(range(1, epochs+1), train_accuracy)
plt.plot(range(1, epochs+1), test_accuracy)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train Accuracy", "Test Accuracy"])
name = 'resnet - Adam - CIFAR10.jpg'
plt.savefig(name)
files.download(name)
plt.show()