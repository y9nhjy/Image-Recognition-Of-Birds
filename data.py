# -*- coding = utf-8 -*-
# @Time : 2022/4/26 0026 17:01
# @Author : 小白
# @File : data.py
# @Software : PyCharm
import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import config
"""实验在CUB200-2011, FGVC-Aircraft, Stanford_Cars和Oxford_102_Flowers四个数据集上进⾏."""
#############################################################################
# CUB200-2011数据集
if config.args.dataset == 'CUB200_2011':
    def default_loader(path):
        return Image.open(path).convert('RGB')

    class MyDataset(Dataset):
        def __init__(self, txt, transform, loader=default_loader):
            fh=open(txt,'r')
            imgs=[]
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))
            self.imgs=imgs
            self.transform = transform
            self.loader = loader

        def __getitem__(self, index):
            fn, label = self.imgs[index]
            img = self.loader(config.data_path + fn)
            img = self.transform(img)
            return img,label

        def __len__(self):
            return len(self.imgs)
