from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import time
torch.manual_seed(0)
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 28
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

dataset_train = dsets.MNIST(root='./data', train=True, download=True, transform=composed)
dataset_val = dsets.MNIST(root='./data', train=False, download=True, transform=composed)

b_size = 64
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=b_size )
test_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=b_size )


class CNN_batch(nn.Module):
    
    # Contructor
    def __init__(self, channel_in=1, layer_1=64, layer_2=128, layer_3=256, layer_4=512, number_of_classes=10):
        super(CNN_batch, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=channel_in, out_channels=layer_1, kernel_size=5, padding=2)
        # self.conv1_bn = nn.BatchNorm2d(layer_1)
        self.cnn12 = nn.Conv2d(in_channels=layer_1, out_channels=layer_1, kernel_size=5, padding=2)
        # self.conv12_bn = nn.BatchNorm2d(layer_1)

        self.cnn2 = nn.Conv2d(in_channels=layer_1, out_channels=layer_2, kernel_size=5, padding=2)
        # self.conv2_bn = nn.BatchNorm2d(layer_2)
        self.cnn22 = nn.Conv2d(in_channels=layer_2, out_channels=layer_2, kernel_size=5, padding=2)
        # self.conv22_bn = nn.BatchNorm2d(layer_2)

        self.cnn3 = nn.Conv2d(in_channels=layer_2, out_channels=layer_3, kernel_size=5, padding=2)
        # self.conv3_bn = nn.BatchNorm2d(layer_3)
        self.cnn32 = nn.Conv2d(in_channels=layer_3, out_channels=layer_3, kernel_size=5, padding=2)
        # self.conv32_bn = nn.BatchNorm2d(layer_3)
        self.cnn33 = nn.Conv2d(in_channels=layer_3, out_channels=layer_3, kernel_size=5, padding=2)
        # self.conv33_bn = nn.BatchNorm2d(layer_3)

        self.cnn4 = nn.Conv2d(in_channels=layer_3, out_channels=layer_4, kernel_size=5, padding=2)
        # self.conv4_bn = nn.BatchNorm2d(layer_4)
        self.cnn42 = nn.Conv2d(in_channels=layer_4, out_channels=layer_4, kernel_size=5, padding=2)
        # self.conv42_bn = nn.BatchNorm2d(layer_4)
        self.cnn43 = nn.Conv2d(in_channels=layer_4, out_channels=layer_4, kernel_size=5, padding=2)
        # self.conv43_bn = nn.BatchNorm2d(layer_4)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(layer_4, 4096)
        self.bn_fc1 = nn.BatchNorm1d(num_features=4096)
        self.fc2 = nn.Linear(4096, 4096)
        # self.bn_fc2 = nn.BatchNorm1d(num_features=4096)
        self.fc3 = nn.Linear(4096, number_of_classes)
    
    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        # x = self.conv1_bn(x)
        x = torch.relu(x)
        x = self.cnn12(x)
        # x = self.conv12_bn(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        x = self.cnn2(x)
        # x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.cnn22(x)
        # x = self.conv22_bn(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        x = self.cnn3(x)
        # x = self.conv3_bn(x)
        x = torch.relu(x)
        x = self.cnn32(x)
        # x = self.conv32_bn(x)
        x = torch.relu(x)
        x = self.cnn33(x)
        # x = self.conv33_bn(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        x = self.cnn4(x)
        # x = self.conv4_bn(x)
        x = torch.relu(x)
        x = self.cnn42(x)
        # x = self.conv42_bn(x)
        x = torch.relu(x)
        x = self.cnn43(x)
        # x = self.conv43_bn(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.bn_fc2(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x


model = CNN_batch(channel_in=1, layer_1=64, layer_2=128, layer_3=256, layer_4=512, number_of_classes=10)
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)


cost_list=[]
accuracy_list_test=[]
accuracy_list_train=[]
N_test=len(dataset_val)
N_train=len(dataset_train)
n_epochs = 5
for epoch in range(n_epochs):
    cost=0
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        cost+=loss.item()
    
    correct=0
    #perform a prediction on the validation  data 
    model.eval()
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
    accuracy = correct / N_test
    accuracy_list_test.append(accuracy)
    # ______________
    # train accuracy
    correct=0
    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)
        z = model(x_train)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_train).sum().item()
    accuracy = correct / N_train
    accuracy_list_train.append(accuracy)
    cost_list.append(cost)
    
    # if epoch % 5 == 0:
    #     print(epoch)
    print(epoch)
    

fig, ax1 = plt.subplots()
color = 'tab:green'
ax1.plot(cost_list, color=color)
ax1.set_xlabel('epoch')
ax1.set_ylabel('Cost', color=color)
ax1.tick_params(axis='y', color=color)
    
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy') 
ax2.plot( accuracy_list_test, color=color)
# ax2.tick_params(axis='y', color=color)
 
color = 'tab:red'
ax2.plot( accuracy_list_train, color=color)

fig.tight_layout()