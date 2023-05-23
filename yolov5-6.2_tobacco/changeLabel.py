import os

def changeLabel(path,path2):

    for r,d,f in os.walk(path):
        for file in f:
            tmp = os.path.join(r,file)
            with open(tmp) as f:
                lines = f.readlines()
                tmp1 = os.path.join(path2,file)
                for i in lines:
                    tmp_line = '0'+i[1:]
                    f2 = open(tmp1,'a')
                    f2.write(tmp_line)
                f2.close()



def k_fold(path,path2,key_value):
    from shutil import copyfile
    # copyfile(path,path2)
    lst = ['flip','Rotation180','Rotation90','Rotation270','shrink']
    for index,flag in enumerate(lst):
        for r,d,f in os.walk(path):
            for file in f:
                if file.split('+')[-1].split('.')[0] == flag:
                    old_path = os.path.join(path,file)
                    flag_file = f'fold{index+1}\\val\\{key_value}'
                    new_path = os.path.join(path2,flag_file,file)
                    copyfile(old_path,new_path)
                else:
                    old_path = os.path.join(path, file)
                    flag_file = f'fold{index+1}\\train\\{key_value}'
                    new_path = os.path.join(path2, flag_file, file)
                    copyfile(old_path, new_path)

path = r'D:\yolo\yolov5-6.2\data\sample\moreChange\newlabels'
path2 = r'D:\yolo\yolov5-6.2\data\sample\moreChange'
path3 = r'D:\yolo\yolov5-6.2\data\sample\moreChange\fold1\val'
k_fold(path,path2,key_value='labels')