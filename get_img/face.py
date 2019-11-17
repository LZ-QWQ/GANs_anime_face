import cv2
import sys
import os
from multiprocessing import Pool
from glob import glob
import signal
import sys
import time

#这是一个不可以打断的多进程！！！！！
def detect(filename,cascade_file="lbpcascade_animeface.xml"):
    save_path1='anime_H_style'
    save_path2='anime_H_style_low'
    if os.path.exists(os.path.join(save_path1,'%s-%d.png' % (os.path.basename(filename)[:-4], 0))) or\
        os.path.exists(os.path.join(save_path2,'%s-%d.png' % (os.path.basename(filename)[:-4], 0))):
        print('好像截过了')
        return 
    
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    try:
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(256,256))
    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        raise KeyboardInterrupt
        
    except:
        #我放弃你了....奇奇怪怪
        print('啥错误？？？')
        print(filename)
        return

    for i, (x, y, w, h) in enumerate(faces):
        flag=True#图像太差的就要用waifu2x了
        add_h_1=0#我觉得吧不扩大好像也行，，
        add_h_2=0
        add_w=0
        face = image[y-add_h_1: y+h+add_h_2, x-add_w:x + w+add_w, :]
        try:
            if h==w and h>=512:
                face = cv2.resize(face, (512, 512),interpolation=cv2.INTER_AREA)
                flag=True
            elif h==w and h<512:
                #face = cv2.resize(face, (512, 512),interpolation=cv2.INTER_CUBIC)
                flag=False
            else:
                print('未知错误')
                continue

        except:
            print('八成就是越界导致数组空掉了，，')
            continue

        try:
            save_filename = '%s-%d.png' % (os.path.basename(filename)[12:-4], i)#不要开头的Konachan.com和结尾的.png或.jpg
            if flag: 
                cv2.imwrite(os.path.join(save_path1,save_filename), face)
            else:
                cv2.imwrite(os.path.join(save_path2,save_filename), face)
            print('%d done~~~'% i)
        except KeyboardInterrupt:
            if os.path.exists(save_filename):
                os.remove(save_filename)
            raise KeyboardInterrupt
        
        except :
            print('未知错误')
            if os.path.exists(save_filename):
                os.remove(save_filename)

    print('done~~~QAQ')

if __name__ == '__main__':
    save_path1='anime_H_style'
    save_path2='anime_H_style_low'
    if os.path.exists(save_path1) is False:
        os.makedirs(save_path1)
    if os.path.exists(save_path2) is False:
        os.makedirs(save_path2)
    file_list = glob('imgs_no_explicit\\*.*')+glob('imgs_no_explicit_1w_after\\*.*')+glob('imgs_no_explicit_reverse\\*.*')
    print(len(file_list))
    with Pool(8) as pool:
        mua=pool.map(detect,file_list,chunksize=2048)
    print('all~done~')