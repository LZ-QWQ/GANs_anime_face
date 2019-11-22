import tensorflow as tf
import numpy as np
import keras
import PIL
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load model
classif_model = keras.models.load_model("classify_model\\model.hdf5")

# Load tags
with open("classify_model\\tags.txt", 'r') as tags_stream:
    tags = np.array([tag for tag in (tag.strip() for tag in tags_stream) if tag])
    # Load image
#in_paths = glob.glob('G_images\\*')
labels=[]#按顺序的
for i in range(0,7000):
    image=np.array(PIL.Image.open(os.path.join('G_images','example-'+str(i)+'.png')).convert('RGB').resize((299, 299))) / 255.0

# Decode
    results = classif_model.predict(np.array([image])).reshape(tags.shape[0])
# Threshold and get tags
    #threshold = 0.1
    #result_tags = {}
    #for i in range(len(tags)):
    #    if results[i] > threshold:
    #        result_tags[tags[i]] = results[i]
    if tags[612]=='blue_hair':
        white_hair=results[612]
        if white_hair<0.5:
            labels.append(0)
        elif white_hair>0.5:
            labels.append(1)
        else:
            labels.append(np.random.randint(0,2,1)[0])
    print ('%d' % i,end='\r')
print('done')
print(np.sum(labels))
np.save('tag_labels.npy',labels)
# Print in order        
    #sorted_tags = reversed(sorted(result_tags.keys(), key = lambda x: result_tags[x]))
    #for tag in sorted_tags:
    #    print('{0: <32} {1:04f}'.format(tag, result_tags[tag]))

#perceptual_model = keras.Model(classif_model.input, classif_model.layers[-5].output)

#in_paths = [
#    "results\\example-0.png",
#    "results\\example-1.png",
#    "results\\example-2.png",
#    "results\\example-3.png",
#    "results\\example-4.png",
#    "results\\example-5.png",
#]

#image_results = []
#for in_path in in_paths:
#    image = np.array(PIL.Image.open(in_path).convert('RGB').resize((299, 299))) / 255.0
#    image_results.append(perceptual_model.predict(np.array([image])).flatten())

#import matplotlib.pyplot as plt

#matrix = []
#for y in range(6):
#    for x in range(6):
#        matrix.append(np.mean(np.square(image_results[x] - image_results[y])))
#matrix = np.array(matrix).reshape(6, 6)
#plt.imshow(matrix)
#plt.show()