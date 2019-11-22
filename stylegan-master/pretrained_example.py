# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import tensorflow as tf
def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    #url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    #with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        #_G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
    
    #G, _D, Gs = pickle.load(open('results/anime_QAQ.pkl', 'rb'))
    #G, _D, Gs = pickle.load(open('QAQ.pkl', 'rb'))
    #G, _D, Gs = pickle.load(open('QwQ.pkl', 'rb'))
    Gs = pickle.load(open('QAQ_Gs.pkl', 'rb'))
    # Print network details.
    #Gs.print_layers()
    
    import training.misc
    #training.misc.save_pkl(Gs, 'QAQ_Gs.pkl')
    #training.misc.save_pkl(Gs, 'QwQ_Gs.pkl')
    for i in range(0,7000):
        #方向向量
        direction=np.load('vector.npy')
        # Pick latent vector.
        #rnd = np.random.RandomState(5)
        SEED = np.random.randint(0,10000)
        rnd = np.random.RandomState(None)
        latents = rnd.randn(1, Gs.input_shape[1])

        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)
        images_new= Gs.run(latents+2*direction, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)
        
        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        #np.save('G_latents(0.7)\\example-'+str(i)+'.npy',latents)
        png_filename = os.path.join(config.result_dir,'example-'+str(i)+'.png')
        new = os.path.join(config.result_dir,'example-new'+str(i)+'.png')
        #png_filename = os.path.join('G_images','example-'+str(i)+'.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        PIL.Image.fromarray(images_new[0], 'RGB').save(new)


        print ('%d' % i,end='\r')

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
