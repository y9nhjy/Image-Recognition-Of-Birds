# Image-Recognition-Of-Birds
Fine grained image recognition of birds\
鸟类细粒度图像识别
## Usage
需要下载CUB200_2011数据集至项目中
### main1.py
用一个全新的VGG16模型从零训练，每轮训练好的模型会保存至 ./model/... 中
### main_read.py
从 ./model/... 中加载最后一个模型，以这个模型为初始模型开始训练
### config.py
参数设置
