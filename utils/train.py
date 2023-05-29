import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor
from torch.utils.data import DataLoader

from Model.ResNet_CIFAR10 import ResNet18

def get_train_loader(batch_size, num_workers):
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    train_dataset = CIFAR10(
        root='./datasets', train=True, download=True, transform=transform_train
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader


def get_test_loader(batch_size, num_workers):
    transform_test = Compose([
        ToTensor(),
    ])

    test_dataset = CIFAR10(
        root='./datasets', train=False, download=True, transform=transform_test
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_loader


def train(net, train_loader, criterion, optimizer, device):
    print('\n[ Train ]')
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        current_correct = predicted.eq(targets).sum().item()
        correct += current_correct

        if batch_idx % 100 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current batch average train accuracy:', current_correct / targets.size(0))
            print('Current batch average train loss:', loss.item() / targets.size(0))

    print('\nTotal average train accuracy:', correct / total)
    print('Total average train loss:', train_loss / total)


def test(net, test_loader, criterion, device, file_name):
    print('\n[ Test ]')
    net.eval()
    loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = net(inputs)
        loss += criterion(outputs, targets).item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

    print('\nTotal average test accuracy:', correct / total)
    print('Total average test loss:', loss / total)

    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model saved!')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = ResNet18().to(device)

    learning_rate = 0.1
