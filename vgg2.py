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
PREPROCESS = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                        # transforms.Normalize(mean = [0.485,0.456,0.406], std = [1.0,1.0,1.0])
                        ])

batch_size = 128
trainset = torchvision.datasets.CIFAR10(root = ".", train = True, download = True, transform = PREPROCESS)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root = ".", train = False, download = True, transform = PREPROCESS)
test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle=True)

train_dataset, train_dataset_total = train_loader, len(trainset)
test_dataset, test_dataset_total = test_loader, len(testset)

class DefaultBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DefaultBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(outchannel)
                        )

    def forward(self, out):
        out = self.conv1(out)
        out = F.relu(out)
        return out


class VGGModel(nn.Module):
    def __init__(self, DefaultBlock, num_classes):
        super(VGGModel, self).__init__()
        self.inchannel = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = self.make_layer(DefaultBlock, 128,  2)
        self.layer3 = self.make_layer(DefaultBlock, 256, 3)
        self.layer4 = self.make_layer(DefaultBlock, 512, 3)
        self.layer5 = self.make_layer(DefaultBlock, 512, 3)

        self.fc1 = nn.Linear(512, 4096)
        self.bn_fc1 = nn.BatchNorm1d(num_features=4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn_fc2 = nn.BatchNorm1d(num_features=4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.drop = nn.Dropout(0.7)

    def make_layer(self, block, out_channels, num_blocks):
        layers = []
        for x in range(num_blocks):
            layers.append(block(self.inchannel, out_channels))
            self.inchannel = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn_fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.bn_fc2(out)
        out = self.drop(out)
        out = self.fc3(out)
        return out


model = VGGModel(DefaultBlock, num_classes = 10)
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
train_accuracy, test_accuracy = [], []

for epoch in range(epochs):
    print("\nEpoch:", epoch)
    correct = 0
    model.train()
    for (images,target) in train_dataset:
        images = images.to(device)
        target = target.to(device)
        out = model(images)
        loss = criterion(out,target)

        # Back-propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out.data,1)
        correct += (pred == target).sum().item()

    train_accuracy.append((correct/train_dataset_total) * 100)
    print(f"Accuracy Train: {train_accuracy[-1]}")

    model.eval()
    correct_test = 0
    for (images_test, target_test) in test_dataset:
        images_test = images_test.to(device)
        target_test = target_test.to(device)

        out_test = model(images_test)
        _, pred_test = torch.max(out_test.data, 1)
        correct_test += (pred_test == target_test).sum().item()
    test_accuracy.append((correct_test/test_dataset_total) * 100)
    print(f"Accuracy Test: {test_accuracy[-1]}")

plt.plot(range(1, epochs+1), train_accuracy)
plt.plot(range(1, epochs+1), test_accuracy)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Train Accuracy", "Test Accuracy"])
name = 'vgg - Adam - CIFAR10.jpg'
plt.savefig(name)
files.download(name)
plt.show()
