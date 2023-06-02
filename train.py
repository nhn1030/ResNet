import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from Model.ResNet_CIFAR10 import ResNet18
from datasets.data_CIFAR10 import get_train_loader as train_loader
from datasets.data_CIFAR10 import get_test_loader as test_loader

if __name__ == '__main__':
    train_loader = train_loader(batch_size=128, num_workers=2)
    test_loader = test_loader(batch_size=128, num_workers=2)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = ResNet18()
    net = net.to(device)

    learning_rate = 0.1
    file_name = 'resnet18_cifar10.pth'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)


    def train(epoch):
        print('\n[ Train epoch: %d ]' % epoch)
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

        print('\nTotal average train accuarcy:', correct / total)
        print('Total average train loss:', train_loss / total)


    def test(epoch):
        print('\n[ Test epoch: %d ]' % epoch)
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

        print('\nTotal average test accuarcy:', correct / total)
        print('Total average test loss:', loss / total)

        state = {
            'net': net.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + file_name)
        print('Model Saved!')


    def adjust_learning_rate(optimizer, epoch):
        lr = learning_rate
        if epoch >= 50:
            lr /= 10
        if epoch >= 100:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    start_time = time.time()

    for epoch in range(0, 150):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
        print('\nTime elapsed:', time.time() - start_time)