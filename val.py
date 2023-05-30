import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Model.ResNet_CIFAR10 import ResNet18, ResNet34
from datasets.data_CIFAR10 import get_test_loader
from tqdm import tqdm

# pth 파일 경로
file1_path = '/workspace/ResNet/checkpoint/resnet18_cifar10.pth'
file2_path = '/workspace/ResNet/checkpoint/resnet34_cifar10.pth'

# 모델 및 loss 기록을 저장할 리스트
models = []
losses = []

# 첫 번째 pth 파일 로드
model1 = ResNet18()
state_dict1 = torch.load(file1_path, map_location=torch.device('cuda'))
model1.load_state_dict(state_dict1['net'])
models.append(model1)

# 두 번째 pth 파일 로드
model2 = ResNet34()
state_dict2 = torch.load(file2_path, map_location=torch.device('cuda'))
model2.load_state_dict(state_dict2['net'])
models.append(model2)

# loss 함수 정의 (예시로 CrossEntropyLoss 사용)
criterion = nn.CrossEntropyLoss()

# 데이터 로더 설정
batch_size = 64
num_workers = 2
test_loader = get_test_loader(batch_size, num_workers)

# 각 모델의 loss 기록
for model in models:
    # 모델을 평가 모드로 설정
    model.eval()

    # loss 계산을 위한 변수 초기화
    batch_losses = []  # 배치별 loss를 기록할 리스트

    # 데이터셋에 대한 반복
    progress_bar = tqdm(test_loader, desc='Processing model')
    for batch_data in progress_bar:
        inputs, targets = batch_data
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        batch_losses.append(loss.item())  # 배치별 loss 기록

    losses.append(batch_losses)  # 각 모델의 배치별 loss 기록

# 그래프 그리기
plt.figure(figsize=(10, 6))
for i, model_losses in enumerate(losses):
    plt.plot(model_losses, label=f'Model {i+1}')

plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss Variation per Batch')
plt.legend()
plt.savefig('loss_comparison.png')
plt.show()
