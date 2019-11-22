import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import sys
import glob

#周二弄好这里！！！
def main():
    tflib.init_tf()
    model_path='QAQ.pkl'
    image_path='data\\*'
    save_path='results\\score.txt'
    _G, D, _Gs = pickle.load(open('QAQ.pkl', 'rb'))
    image_filenames = glob.glob(image_path)
    out={}
    temp=len(image_filenames)
    
    for i in range(0, temp):
        img = np.asarray(PIL.Image.open(image_filenames[i]))
        img = img.reshape(1, 3,512,512)
        score = D.run(img, None)
        print(image_filenames[i], score[0][0])
        out[image_filenames[i]]=score[0][0]
        print(str(i)+'\n'+str(temp))

    out=sorted(out.items(),key=lambda item:item[1])
    with open(save_path,mode='w',encoding='UTF-8') as file:
        for filename,score in out:
            file.write(filename+'\t'+str(score)+'\n')

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()