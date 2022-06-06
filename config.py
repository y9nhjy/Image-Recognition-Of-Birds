# -*- coding = utf-8 -*-
import os
CUB_PATH = '/media/lm/C3F680DFF08EB695/细粒度数据集/birds/CUB200/CUB_200_2011/dataset'
PROJECT_ROOT = os.getcwd()
PATH = {
    'cub200_train': CUB_PATH + '/train/',
    'cub200_test': CUB_PATH + '/test/',
    'model': os.path.join(PROJECT_ROOT, 'model/')
}
BASE_LEARNING_RATE = 0.05
EPOCHS = 100
BATCH_SIZE = 8
WEIGHT_DECAY = 0.00001
