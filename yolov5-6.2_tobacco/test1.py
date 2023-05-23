import torch
import cv2
import os
from models import MobileNetv3_small_ca,MobileNetv3_se
import numpy as np
models = []
modelpath = 'weight/se'
for i in os.listdir(modelpath):
    path = os.path.join(modelpath,i)
    net = MobileNetv3_se.MobileNetV3_Small(96)

    net.load_state_dict(torch.load(path))
    net.eval()
    models.append(net)

pic = 'runs/detect/exp7/crops/cigarette'
ims = []
for i in os.listdir(pic):
    res = torch.zeros((1,96))
    for m in models:
        pic_path = os.path.join(pic,i)
        im = cv2.imread(pic_path)

    # print(im)

        im = cv2.resize(im,(224,224))
        # print(im)

        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

        im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
        # im = normalize(im,mean=[0.578,0.516,0.474], std=[0.261, 0.266, 0.259])
        im /=255
        # print(im)

        im = im.reshape([1, im.shape[0], im.shape[1], im.shape[2]])

        # im = torch.randn((224,224,3))
        # print(im.shape)
        # print(im)
        res += m(torch.Tensor(im))
        # print(im)
        # print(im.shape)
        # print(i(ims))
    print(res.argmax(1))
# [0.04705882 0.05098039 0.03137255 ... 0.57254905 0.6039216  0.6117647 ]
#   [0.06666667 0.05098039 0.04705882 ... 0.58431375 0.6        0.60784316]
#   [0.10196079 0.09019608 0.07843138 ... 0.58431375 0.58431375 0.59607846]