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

        print('\nTotal average train accuracy:', correct / total)
        return train_loss / total

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

        print('\nTotal average test accuracy:', correct / total)
        return loss / total

    early_stopping_epochs = 10  # 조기 종료를 위한 에포크 수
    best_loss = float('inf')  # 최적의 손실값 초기화
    early_stopping_counter = 0  # 조기 종료 카운터 초기화
    best_model_state = None  # 최적 모델의 상태

    start_time = time.time()

    for epoch in range(0, 150):
        train_loss = train(epoch)
        test_loss = test(epoch)
        print('\nTime elapsed:', time.time() - start_time)

        # 최적 성능 갱신 여부 확인
        if test_loss < best_loss:
            best_loss = test_loss  # 최적 성능 갱신
            early_stopping_counter = 0  # 개선되지 않은 에포크 카운터 초기화
            best_model_state = net.state_dict()  # 최적 모델의 상태 저장
        else:
            early_stopping_counter += 1  # 개선되지 않은 에포크 카운터 증가

        # 개선되지 않은 에포크 횟수가 일정 값 이상인 경우 조기 종료
        if early_stopping_counter >= early_stopping_epochs:
            print("조기 종료!")
            break

    # 최적 모델 저장
    if best_model_state is not None:
        state = {
            'net': best_model_state
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + file_name)
        print('Best Model Saved!')
