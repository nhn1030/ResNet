import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.ResNet_CIFAR10 import ResNet18

device = 'cuda'

net = ResNet18()
net = net.to(device)

learning_rate = 0.1
file_name = 'resnet18_cifar10.pth'

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum= 0.9, weight_decay= 0.0002)

def train(epoch):
    print('\n[Train epoch: %d]' %epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
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
            
