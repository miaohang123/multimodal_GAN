import os
import time
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
#from keras.applications.inception_v3 import preprocess_input
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.35
set_session(tf.Session(config=config))

image_width = 224
image_height = 224

class ImageFeature(object):
    def __init__(self, text_path, image_dir):
        self.text_path = text_path
        self.image_dir = image_dir
        self.image_path_list = []
        with open(self.text_path, 'r') as f:
            for line in f:
                self.image_path_list.append(os.path.join(self.image_dir, line.strip()))
        self.model()

    def model(self):
        base_model = VGG19(weights='imagenet', include_top=False)
        #self.model = InceptionV3(weights='imagenet', include_top=False)
        #self.model = VGG16(weights='imagenet', include_top=False)
        #self.model = VGG19(weights='imagenet', include_top=False)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
        # print(self.model.summary())

    def preprocess_input(self, x):
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] /= 255.0
        x[..., 1] /= 255.0
        x[..., 2] /= 255.0

        return x

    def extract_feature(self, filepath):
        # print(filepath)
        img = image.load_img(filepath, target_size=(image_width, image_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        #x = preprocess_input(x)
        # print(x.shape)
        # print(x[0, :, :, 0])
        features = self.model.predict(x)
        return features

    def save_feature(self, save_path):
        res = []
        for image_path in self.image_path_list:
            image_feature = self.extract_feature(image_path)
            res.append(image_feature)
        res = np.array(res)
        np.save(save_path, res)


if __name__ == '__main__':
    image_feature_object = ImageFeature(text_path='../../data/Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt', 
                                        image_dir='../../data/Flickr8k/Flickr8k_Dataset')

    image_feature_object.save_feature(save_path='../../data/Flickr8k/Flickr8k_feature/vgg19_train.npy')
    

