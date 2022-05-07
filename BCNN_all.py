# -*- coding = utf-8 -*-
# @Time : 2022/4/26 0026 16:31
# @Author : 小白
# @File : BCNN_all.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torchvision


class BCNN_all(nn.Module):
    def __init__(self):
        super(BCNN_all, self).__init__()
        #VGG16的卷积层与池化层
        self.features = torchvision.models.vgg16(pretrained=False).features
        self.features = torch.nn.Sequential(*list(self.features.children())[:-1]) #去除最后的pool层
        #全连接层
        self.fc = torch.nn.Linear(512*512, 200)

    def forward(self,x):
        N=x.size()[0]
        assert x.size() == (N, 3, 448, 448)
        x=self.features(x)
        assert x.size() == (N, 512, 28, 28)
        x = x.view(N, 512, 28*28)
        x = torch.bmm(x, torch.transpose(x, 1, 2))/(28*28) #双线性
        assert x.size() == (N, 512, 512)
        x = x.view(N, 512*512)
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
        x=self.fc(x)
        assert x.size() == (N, 200)
        return x
