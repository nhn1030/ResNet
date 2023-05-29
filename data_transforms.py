import torchvision
import torchvision.transforms as transforms

transform_train = transforms.compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.train_dataset