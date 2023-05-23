import cv2
import numpy as np
import torch
import os
from models import MobileNetv3_small_ca,MobileNetv3_se

def classify(pic):
    models = []
    modelpath = 'weight/se'
    for i in os.listdir(modelpath):
        path = os.path.join(modelpath, i)
        net = MobileNetv3_se.MobileNetV3_Small(96)

        net.load_state_dict(torch.load(path))
        net.eval()
        models.append(net)

    res = torch.zeros((1, 96))
    for m in models:
        # im = cv2.imread(pic)
        im = cv2.resize(pic, (224, 224))
        im = im[:, :, ::-1].transpose(2, 0, 1)
        im = np.ascontiguousarray(im, dtype=np.float32)
        im /= 255
        im = im.reshape([1, im.shape[0], im.shape[1], im.shape[2]])
        res += m(torch.Tensor(im))
    print(res.argmax(1))