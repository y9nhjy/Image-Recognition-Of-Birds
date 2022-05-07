# -*- coding = utf-8 -*-
# @Time : 2022/4/26 0026 16:31
# @Author : 小白
# @File : data_load.py
# @Software : PyCharm
import torch
import torchvision
import config
def train_data_process():
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=448),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=448),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])
    train_data = torchvision.datasets.ImageFolder(
        root=config.PATH['cub200_train'],
        transform=train_transforms
    )
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True
                                               )
    return train_loader

def test_data_process():
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=448),
        torchvision.transforms.RandomCrop(size=448),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])
    test_data = torchvision.datasets.ImageFolder(root=config.PATH['cub200_test'],
                                                 transform=test_transforms
                                                 )
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=True,num_workers=8,
                                              pin_memory=True
                                              )
    return test_loader

if __name__ == '__main__':
    train_data_process()
    test_data_process()