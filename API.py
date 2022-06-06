# encoding:utf-8
import os
import requests
import base64
import time

'''
动物识别
'''

root='./CUB_200_2011'
img_txt_file = open(os.path.join(root, 'images.txt'))
label_txt_file = open(os.path.join(root, 'image_class_labels.txt'))
train_val_file = open(os.path.join(root, 'train_test_split.txt'))
# 图片索引
img_name_list = []
for line in img_txt_file:
    # 最后一个字符为换行符
    img_name_list.append(line[:-1].split(' ')[-1])
# 标签索引，每个对应的标签减１，标签值从0开始
label_list = []
for line in label_txt_file:
    label_list.append(int(line[:-1].split(' ')[-1]) - 1)
# 训练集和测试集
train_test_list = []
for line in train_val_file:
    train_test_list.append(int(line[:-1].split(' ')[-1]))
test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
test_file_list = [os.path.join(root, 'images', test_file) for test_file in test_file_list]
test_label_list = [x for i, x in zip(train_test_list, label_list) if not i]

imgs_path = 'CUB_200_2011/images'
map = {}
for i in os.listdir(imgs_path):
    map[i[:3]] = i[4:].replace('_',' ')
# for x in map.values():
#     print(x)

request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/animal"
access_token = '24.a3d763e02bc84bf49ef8a5e35196feac.2592000.1656768008.282335-26369567'
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
fr = open('API预测结果.txt','w')
for x,img_path in enumerate(test_file_list):
    # 二进制方式打开图片文件
    f = open(img_path, 'rb')
    img = base64.b64encode(f.read())
    params = {"image":img}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        fr.write(str(x)+' '+response.json()['result'][0]['name']+'\n')
        # print(str(x) + ' ' + response.json()['result'][0]['name'] + '\n')
        print (response.json()) # response.json()['result'][0]['name']
        print(str(x) + ' ' + response.json()['result'][0]['name']+'\n')
    time.sleep(0.5)
fr.close()
