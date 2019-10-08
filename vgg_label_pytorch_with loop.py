import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os
import json
import numpy as np
from PIL import Image

vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
folders = []
dirname = os.path.dirname(__file__)
class_index = json.load(open('imagenet_class_index.json', 'r'))
os.chdir(os.path.join(dirname, 'data'))
for r, d, f in os.walk(os.getcwd()):
    if((r.endswith("tsukurepo"))==False):
        folders.append(r)

#for f in folders:
   # print(f)

folders.remove(folders[0])
for i in range(len(folders)):
    print(folders[i])
    filename = os.path.join(folders[i],'thumbnail.jpg')
    print(filename)
    img = Image.open(filename)
    img
    # 4チャンネルだったので最初の3チャンネルを使う
    img_tensor = preprocess(img)[:3]
    #print(img_tensor.shape)

    # Tensorにする前なら画像として表示可能
    preprocess2 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    trans_img = preprocess2(img)
    #print(type(trans_img))
    trans_img
    # モデルに入力するときはバッチサイズを入れた4Dテンソルにする必要がある
    img_tensor.unsqueeze_(0)
    #print(img_tensor.size())
    out = vgg16(img_tensor)
    #print(out.size())
    # 出力確率が一番高いインデックスを取得
    out.max(dim=1)
    # 出力確率が高いトップ5のインデックスを取得
    out.topk(5)
    #print(class_index['0'])

    labels = {int(key):value for (key, value) in class_index.items()}
    #print(labels[0])
    #print(labels[1])
    index = out.max(dim=1)[1].item()
    print(labels[index][1])