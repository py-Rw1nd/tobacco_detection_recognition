import cv2 as cv
import os

path = 'data/ruixiang/train/images'
path1 = 'data/ruixiang/train/images/new'
for r, d,f in os.walk(path):

    for name in f:
        print(name,'start')
        tmp = os.path.join(r, name)
        img = cv.imread(tmp)
        newname = os.path.join(path1,name)
        cv.imwrite(f'{newname}',img)
        print(name, '')


# a = cv.imread('C:/Users/Administrator/Pycode/cv2/1.jpg')