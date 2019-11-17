import PIL.Image
import numpy as np
import os
temp='anime_H_style\Konachan.com%20-%20267480%20animal%20barefoot%20blue_eyes%20blush%20bow%20breasts%20cleavage%20fang%20group%20hug%20ixveria%20loli%20navel%20nude%20onsen%20ponytail%20red_eyes%20scan%20towel%20turtle%20water%20wet%20wink-2.png'
print(len(temp))
temp=os.path.split(temp)
temp[1]=temp[1][-200:]
print(len(temp))
#os.rename(temp,temp[-200:])
img = np.asarray(PIL.Image.open(temp))
print('emmm')