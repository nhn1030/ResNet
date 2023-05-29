import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os    

class BasicBlock(nn.Module):  ## 파이토치의 nn.module을 상속받아 BasicBlock을 생성
    def __init__(self, in_planes, out_planes, stride=1): #BasicBlock 클래스의 초기화 매서드입니다. in_planes(입력 필터 갯수), out_planes(출력 필터 갯수)
        super(BasicBlock, self).__init__() # (super = 부모 클래스의 매서드 호출 및 초기화 하는 파이썬 내장 함수)nn.Module의 초기화 메서드를 호출합니다.
        
        # nn.Conv2d 를 사용하여 2d 컨볼루션 레이어 conv1을 생성함, 입력 채널(in_planes), 출력채널 (out_planes), 커널 크기 3x3, 스트라이드 1, 편향 비활성화를 설정
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias = False)
        # nn.BatchNorm2d 를 사용하여 배치 정규화 과정을 수행한다. 입력채널수는 outplanes의 값을 받음
        self.bn1 = nn.BatchNorm2d(out_planes)
        
    
        # 두번째 컨볼루션 레이어를 생성하며 설정은 이전 레이어와 동일
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        # shortcut 연결을 위한 빈 Sequential 객체 생성 및 self.shortcut에 인스턴스에 할당 stride 값이 1인 경우 해당하는 부분에서 shortcut 연결이 필요하지 않기 때문
        self.shortcut = nn.Sequential()
        
        # 입력 데이터의 stride 값이 1이 아닌 경우에만 shortcut 연결을 수행함, 1인 경우에는 입력과 출력의 크기가 동일하므로 추가적인 처리가 필요하지 않음
        if stride != 1:
            # shortcut 연산을 위해 Sequential 객체를 새롭게 정의한다 stride 값이 1이 아닌경우, 이 객체는 1x1 컨볼루션 연산과 batch normalization을 수행함
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size= 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_planes)
        )
            
    # forward(순전파) 주어진 입력 x를 기반으로 신경망의 순전파 연산을 수행함
        
    def forward(self, x):
        out = F.relu(self.bnl(self.conv1(x))) # 컨볼루션 연산을 통해 처리한 결과(self.conv1) 'x' 를 self.bnl과 relu 활성화 함수를 거쳐 out 변수에 저장
        out = self.bn2(self.conv2(out)) # 이전 단계의 결과인 out을 self.conv2 연산을 통해 다시 처리하고, 그 결과를 배치 정규화를 거쳐 다시 out 변수에 저장함
        out += self.shortcut(x) # (핵심) skip connection 
        out = F.relu(out) # skip connection을 거쳐 최종적으로 합 연산된 out을 relu를 통해 activation
        return(out)
    
class ResNet(nn.Module): # ResNet 클래스 정의 nn.Module 상속 받음
    # block, num_blocks, num_classes 는 클래스의 생성자인 init의 매개변수로 전달되는 값들이다. block은 레즈넷에서 사용되는 기본 블록을 나타내고 
    # num_blocks는 각 레이어에서 반복되는 블록의 수를 나타냄 num_classes는 모델이 분류해야 할 클래스의 수를 나타냄 (해당 코드에서는 cifar10 사용으로 클래스 10개)
    def __init__(self, block, num_blocks, num_classes=10): 
        super(ResNet, self).__init__()
        
        # resnet은 초기입력으로 64개의 채널을 사용하기 때문에, 초기입력 채널 64개로 설정
        self.in_planes = 64
        
        # 입력으로 3개의 채널을 받고, 3x3 커널로 64개의 필터를 활용하여 이미지를 컨볼루션, 스트라이드는 1로 설정되어, 입력과 동일한 크기의 특징맵을 생성
        # 최소한의 조건으로 모델을 재구현 하기위해 편향은 false로 설정
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=3, stride=1, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # resnet 모델의 구성을 정의하는 부분
        # _make_layer 매서드를 통해 각각의 레이어를 생성 (논문의 레즈넷 구조와 동일하게 설정)                   
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)         
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes) # fully connected layer로 통과시켜 분류 작업을 수행하고, 특성의 클래스를 10개로 반환
    
    # _make_layer 함수 정의 <- ResNet 레이어를 생성하는 역할을 함, 입력으로 받은 매개변수에 따라 여러 개의 블록을 반복하는 구조
    # block : 사용할 블록 클래스 (BasicBlock) outplanes : 출력 특징맵의 채널 수, numblocks : 해당 레이어에어서 반복되는 블록 수, stride : 스트라이드
    def _make_layer(self, block, outplanes, num_blocks, stride):
        # strides 리스트는 stride 값을 리스트의 첫번째 요소로 받고, 나머지는 1로 채워진 리스트를 반환 
        # 이유는 첫번째 블록은 입력과 동일한 크기를 유지하기위해 스트라이드를 그대로 사용하고, 나머지는 스트라이드가 1로 고정되도록 하기 위함임
        strides = [stride] + [1] * (num_blocks - 1)
        # 반복문을 통해 블록을 생성하여 추가함, 각 블록은 block 클래스의 인스턴스로 생성되며 괄호 안의 매개변수를 전달함
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, outplanes, stride))
            # 이후 self.in_planes 를 outplanes 로 업데이트하여 다음 블록의 입력 채널 수를 설정함.
            self.in_planes = outplanes 
        # layers리스트에 생성된 블록을 개별요소로 분해한뒤 nn.Sequential을 사용하여 순차적으로 연결한 후, 생성된 레이어를 반환
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 입력 데이터 x 를 conv1 망에 통과시켜 컨볼루션 연산을 수행한다. 그후 배치정규화를 거쳐 렐루로 엑티베이트헤 비선형성을 도입 후 out에 저장
        out = F.relu(self.bn1(self.conv1(x)))
        # out을 각각 self.layer1,2,3,4에 전달하여 레이어를 통과시킴 통과 시킬때 마다 업데이트
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out에 avg pooling of 연산을 적용하여 차원을 축소시킴, (입력이미지의 공간적 특징을 압축하는 역할)
        out = F.avg_pool2d(out, 4)
        # out의 텐서 크기를 재조정하고 배치 차원을 유지하면서 나머지 차원들을 하나의 차원으로 펼처준다. (1차원으로 flatten)
        out = out.view(out.size(0), -1)
        # out을 선형반환하여 최종출력을 계산
        out = self.linear(out)
        return(out)

# ResNet18 - ResNet 클래스의 생성자인 block 매개변수에 BasicBlock의 클래스를 전달하여 해당블록을 사용하도록 하는 것
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2]) 

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,4])
