import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Model.ResNet_CIFAR10 import ResNet18
from Model.ResNet_CIFAR10 import ResNet34
from datasets.data_CIFAR10 import get_test_loader as dataloader

# CSV 파일 경로
file1_path = '/workspace/ResNet/utils/resnet18_cifar10.csv'
file2_path = '/workspace/ResNet/utils/resnet34_cifar10.csv'

# 모델 및 loss 기록을 저장할 리스트
models = []
losses = []

# 첫 번째 CSV 파일 로드
model1 = ResNet18()
state_dict1 = torch.load(file1_path)
model1.load_state_dict(state_dict1)
models.append(model1)

# 두 번째 CSV 파일 로드
model2 = ResNet34()
state_dict2 = torch.load(file2_path)
model2.load_state_dict(state_dict2)
models.append(model2)

# loss 함수 정의 (예시로 CrossEntropyLoss 사용)
criterion = nn.CrossEntropyLoss()

# 각 모델의 loss 기록
for model in models:
    # 모델을 평가 모드로 설정
    model.eval()
    
    # loss 계산을 위한 변수 초기화
    total_loss = 0
    num_batches = 0
    
    # 데이터셋에 대한 반복
    for batch_data in dataloader:  # 적절한 데이터로더를 사용해야 합니다.
        inputs, targets = batch_data
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        num_batches += 1
    
    # 평균 loss 계산
    avg_loss = total_loss / num_batches
    losses.append(avg_loss)

# 그래프 그리기
plt.plot(losses, marker='o')
plt.xlabel('Model')
plt.ylabel('Loss')
plt.xticks(range(len(models)), ['Model 1', 'Model 2'])  # x축 레이블 설정
plt.title('Comparison of Loss between Models')
plt.show()
