import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 50
inchannel = 3
image_size = 32
PREPROCESS = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [1.0, 1.0, 1.0])
                        ])

PREPROCESS_TEST = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

batch_size = 256
trainset = torchvision.datasets.CIFAR10(root = ".", train = True, download = True, transform = PREPROCESS)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root = ".", train = False, download = True, transform = PREPROCESS_TEST)
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
        self.inchannel = inchannel
        self.layer1 = self.make_layer(DefaultBlock, 64, 2)
        self.layer2 = self.make_layer(DefaultBlock, 128, 2)
        self.layer3 = self.make_layer(DefaultBlock, 256, 3)
        self.layer4 = self.make_layer(DefaultBlock, 512, 3)
        self.layer5 = self.make_layer(DefaultBlock, 512, 3)

        self.fc1 = nn.Linear(512, 4096)
        self.bn_fc1 = nn.BatchNorm1d(num_features=4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn_fc2 = nn.BatchNorm1d(num_features=4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.drop = nn.Dropout(0.5)

    def make_layer(self, block, out_channels, num_blocks):
        layers = []
        for x in range(num_blocks):
            layers.append(block(self.inchannel, out_channels))
            self.inchannel = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
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
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
train_accuracy, test_accuracy, train_cost = [], [], []

for epoch in range(epochs):
    start = time.time()
    correct = 0
    cost = 0
    model.train()
    for (images,target) in train_dataset:
        images = images.to(device)
        target = target.to(device)
        out = model(images)
        loss = criterion(out, target)

        # Back-propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out.data, 1)
        correct += (pred == target).sum().item()
        cost += loss.item()

    train_cost.append(cost)
    train_accuracy.append((correct/train_dataset_total) * 100)

    model.eval()
    correct_test = 0
    for (images_test, target_test) in test_dataset:
        images_test = images_test.to(device)
        target_test = target_test.to(device)

        out_test = model(images_test)
        _, pred_test = torch.max(out_test.data, 1)
        correct_test += (pred_test == target_test).sum().item()
    test_accuracy.append((correct_test/test_dataset_total) * 100)
    print("\nEpoch:", epoch, " | Accuracy Train:", train_accuracy[-1], " | Accuracy Test:", test_accuracy[-1], " | Time:", time.time() - start)


top5 = []
for (images_test, target_test) in test_dataset:
    images_test = images_test.to(device)
    target_test = target_test.to(device)
    out_test = model(images_test)
    out_test, pred_test = torch.sort(out_test, descending=True)
    target_test = target_test.view(-1, 1)
    for i, y in enumerate(pred_test[:,0:5]):
        top5.append(y in target_test[i])


print("\n\nFinal Train Accuracy:", train_accuracy[-1])
print("Final Test Accuracy:", test_accuracy[-1])
print("Top-1 error rate:", 1 - test_accuracy[-1]/100)
print("Top-5 error rate:", 1 - sum(top5)/test_dataset_total)


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(train_cost, color=color)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cost', color=color)
ax1.tick_params(axis='y', color=color)
ax2 = ax1.twinx()  
color = 'tab:orange'
ax2.set_ylabel('Accuracy') 
ax2.plot( test_accuracy, color=color) 
color = 'tab:blue'
ax2.plot( train_accuracy, color=color)
fig.tight_layout()
fig.legend(['cost', 'train', 'test'])
name = 'VGG - CIFAR10 - SGD - 0.001 - Experiment.jpg'
plt.savefig(name)
