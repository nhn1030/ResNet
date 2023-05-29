import torch
import torchvision
import torchvision.transforms as transforms

def get_train_loader(batch_size, num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./datasets', train=True, download=True, transform=transform_train
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader

def get_test_loader(batch_size, num_workers):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./datasets', train=False, download=True, transform=transform_test
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return test_loader
