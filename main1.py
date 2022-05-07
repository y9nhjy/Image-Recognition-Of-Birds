# -*- coding = utf-8 -*-
# @Time : 2022/4/26 0026 16:31
# @Author : 小白
# @File : main.py
# @Software : PyCharm
import torch.nn as nn
import argparse
import config
from BCNN_fc import BCNN_fc
from BCNN_all import BCNN_all
# import data.data as data
import torch.backends.cudnn as cudnn
from load import *


# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#导⼊数据集
# train_set = data.MyDataset_train(config.train_txt_path,
#                                 transform=transforms.Compose([
#                                     transforms.Resize((config.Resize, config.Resize), Image.BILINEAR),
#                                     transforms.RandomHorizontalFlip(),
#                                     transforms.RandomCrop(config.Crop_Size),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                                          std=(0.229, 0.224, 0.225))
#                                 ]))
# train_loader = torch.utils.data.DataLoader(train_set,
#                                           batch_size=config.Batch_Size,
#                                           shuffle=True,
#                                           num_workers=config.num_workers
#                                           )
# test_set = data.MyDataset_test(config.test_txt_path,
#                               transform=transforms.Compose([
#                                   transforms.Resize((config.Resize, config.Resize), Image.BILINEAR),
#                                   transforms.CenterCrop(config.Crop_Size),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                                        std=(0.229, 0.224, 0.225))
#                               ]))
# test_loader = torch.utils.data.DataLoader(test_set,
#                                          batch_size=config.Batch_Size,
#                                          shuffle=False,
#                                          num_workers=config.num_workers
#                                          )
transform_train = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.Resize((448, 448)),
        # transforms.RandomCrop(224, padding=4),
        transforms.RandomCrop(448, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
dataset = CUB(root='./CUB_200_2011', is_train=False, transform=transform_train,)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                                              drop_last=True)



cudnn.benchmark = True

# 加载数据
# train_loader = train_data_process()
# test_loader = test_data_process()

# 主程序
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='network_select')
    parser.add_argument('--net_select',
                        dest='net_select',
                        default='BCNN_all',
                        help='select which net to train/test.')
    args = parser.parse_args()
    if args.net_select == 'BCNN_fc':
        net = BCNN_fc().to(device)
    else:net = BCNN_all().to(device)

    #损失
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.fc.parameters(),
                                lr=config.BASE_LEARNING_RATE,
                                momentum=0.9,
                                weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=3,
        verbose=True,
        threshold=1e-4
    )

    # 训练模型
    print('Start Training ==>')
    total_step = len(trainloader)
    best_acc=0.0
    best_epoch=None
    for epoch in range(config.EPOCHS):
        epoch_loss=[]
        num_correct=0
        num_total=0
        for i, (images, labels) in enumerate(trainloader):
            # 数据转为cuda
            images = torch.autograd.Variable(images.cuda())
            labels = torch.autograd.Variable(labels.cuda())
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = net(images)
            loss = criterion(outputs, labels)
            # aaaa=loss.dataepoch_loss.append(loss.data)
            # 预测
            _, prediction = torch.max(outputs.data, 1)
            num_total += labels.size(0)
            num_correct += torch.sum(prediction == labels.data)
            # 后向传播
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}'.format(
                    epoch + 1, config.EPOCHS, i+1, total_step, loss.item()))
        train_Acc = 100*num_correct/num_total
        train_Acc = 100*num_correct/num_total
        # print('Epoch:%d Training Loss:%.03f Acc: %.03f' % (epoch+1, sum(epoch_loss)/len(epoch_loss),
        #                                                    train_Acc))
        torch.save(net.state_dict(), config.PATH['model'] + 'vgg16_epoch_%d.pth' % (epoch + 1))

        # 在测试集上进⾏测试
        # print('Watting for Test ==>')
        # with torch.no_grad():
        #     num_correct=0
        #     num_total=0
        #     for images, labels in test_loader:
        #         net.eval()
        #         images = torch.autograd.Variable(images.cuda())
        #         labels = torch.autograd.Variable(labels.cuda())
        #         outputs = net(images)
        #         _, prediction = torch.max(outputs.data, 1)
        #         num_total += labels.size(0)
        #         num_correct += torch.sum(prediction == labels.data).item()
        #     test_Acc = 100 * num_correct / num_total
        #     print('第%d个Epoch下的测试精度为: %.03f' % (epoch+1, test_Acc))
        #     #保存模型
        #     torch.save(net.state_dict(), config.PATH['model'] + 'vgg16_epoch_%d.pth' % (epoch + 1))
