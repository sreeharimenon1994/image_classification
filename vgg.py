import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Functin to initialize the weight using xavier
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

# To check if gpu is available, otherwise run in cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = torchvision.datasets.FashionMNIST(root = ".", train = True ,
download = True , transform = transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root = ".", train = False ,
download = True , transform = transforms.ToTensor())
training_loader = torch.utils.data.DataLoader(train_set , batch_size = 32,
shuffle = False)
test_loader = torch.utils.data.DataLoader(test_set , batch_size = 32,
shuffle = False)
torch.manual_seed(0)


class Net(nn.Module):

    def __init__(self):
        #  Initializing the function to use at necessary stages
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.reshape(x.size(0), -1) # reshaping so as to group the values for the Linear layer 
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # dropout helps in preventing overfitting.
        x = F.relu(self.fc2(x))
        return x

model = Net()
if torch.cuda.is_available():
    model.cuda()

model.apply(init_weights)
loss_fn = torch.nn.CrossEntropyLoss()

# setting the learnring rate
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
acc_train = []
acc_test = []
epoch = 50

#  Function used to calculate the accuracy
def evaluation(dataloader, network):
  total, correct = 0, 0
  network.eval()
  for data in dataloader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = network(inputs)
    _, pred = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (pred == labels).sum().item()
  return 100 * correct / total


def train(training_loader, network, epoch, optimizer):
  for e in range(epoch):
    for i, data in enumerate(training_loader, 0):
      network.train()
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      # predicting
      outputs = network(inputs)
      # calculatingt the loss
      loss = loss_fn(outputs, labels)
      # updating the weights after using the error
      loss.backward()
      optimizer.step()
    acc_train.append(evaluation(training_loader, model))
    acc_test.append(evaluation(test_loader, model))
    if e%5 == 0:
      print(e)


train(training_loader, model, epoch, optimizer)


tot = list(range(1, epoch+1))
plt.plot(tot, acc_train)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(tot, acc_test)
plt.legend(['Train', 'Test'], loc=4)
name = 'Drop .3 - '+str(learning_rate)+'.jpg'
plt.savefig(name)