from json import load
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import CIFAR10

DATA_ROOT = "./dataset"
DEVICE = 'mps'

class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x):
        """Compute forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x

def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

    return trainloader, testloader

def load_partition(cid, data_dir='./dataset/'):
    filename = Path(data_dir) / f"train_{cid}.pt"

    trainset = torch.load(filename)
    X_train = torch.Tensor(trainset[0])
    X_train = X_train.permute(0, 3, 1, 2)
    y_train = torch.Tensor(trainset[1]).type(torch.LongTensor)
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)

    _, testloader = load_data()
    return trainloader, testloader

def train(net, trainloader, epochs, device, lr, cur_epoch=1):
    """ 
    Return number of examples that are trained
    """
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    net.to(device)
    net.train()
    for epoch in tqdm(range(cur_epoch, cur_epoch + epochs)):
        total, correct = 0, 0
        epoch_acc, epoch_loss = 0.0, 0.0
        for i, data in enumerate(trainloader):
            imgs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optim.zero_grad()
            # forward + backward + optimize
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()
            # calculate monitored metrics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

        epoch_acc = correct / total 
        epoch_loss /= len(trainloader.dataset)
        print(f"Epoch {epoch}: train loss {epoch_loss}, accuracy {epoch_acc}")

    return total


def test(net, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.to(device)
    net.eval()
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = net(imgs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
    loss /= len(testloader.dataset)
    acc = correct / total
    return loss, acc, total

if __name__ == '__main__':
    trainloader, testloader = load_data()
    # trainloader = load_partition(cid=1)
    model = Net()
    print("Start Training")
    freq = 5
    epochs = 10
    epoch = 1
    for iter in range(epochs // freq):
        num_samples = train(model, trainloader, freq, DEVICE, lr=0.001, cur_epoch=epoch)
        loss, accuracy, num_samples = test(model, testloader, DEVICE)
        print("Model evaluation on test set")
        print(f"Test loss = {loss}| Test acc = {accuracy} | num_samples {num_samples}")
        epoch += freq