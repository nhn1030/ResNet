import torch
import pandas as pd

# .pth 파일 경로
file_path1 = '/Users/nohyeonnam/Documents/exercise/checkpoint/resnet18_cifar10.pth'
file_path2 = '/Users/nohyeonnam/Documents/exercise/checkpoint/resnet34_cifar10.pth'
# .pth 파일 로드
state_dict1 = torch.load(file_path1, map_location=torch.device('cpu'))
state_dict2 = torch.load(file_path2, map_location=torch.device('cpu'))

# state_dict를 데이터프레임으로 변환
df1 = pd.DataFrame.from_dict(state_dict1)
df2 = pd.DataFrame.from_dict(state_dict2)


# CSV 파일로 저장
csv_path1 = 'resnet18_cifar10.csv'
csv_path2 = 'resnet34_cifar10.csv'
df1.to_csv(csv_path1, index=True)
df2.to_csv(csv_path2, index=True)
