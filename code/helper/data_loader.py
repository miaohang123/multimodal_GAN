import os
import sys
import re
import tensorflow.contrib.keras as kr
import pandas as pd
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer

import utils

class data_loader(object):
	def __init__(self, batch_size, dataset_dir, max_words, is_training):
		self.batch_size = batch_size
		self.dataset_dir = dataset_dir
		self.max_words = max_words
		self.is_training = is_training
		_, self.word_to_id = utils.read_vocab(vocab_path='../../data/Flickr8k/Flickr8k_text/vocab.txt')
		self.token_dict = utils.get_flicker8k_token(datapath=self.dataset_dir + '/Flickr8k_text/Flickr8k.token.txt')

	def load_cap_data(self, load_train):
	    text_dir = os.path.join(self.dataset_dir, 'Flickr8k_text') 
	    # train_caps, dev_caps, test_caps = [],[],[]
	    #load captions
	    
	    if load_train:
	    	train_caps = utils.process_caption(data_path='../../data/Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt', 
	    									   word_to_id=self.word_to_id)
	    else:
	    	train_caps = None
	    dev_caps = utils.process_caption(data_path='../../data/Flickr8k/Flickr8k_text/Flickr_8k.devImages.txt', 
	    								  word_to_id=self.word_to_id)
	    test_caps = utils.process_caption(data_path='../../data/Flickr8k/Flickr8k_text/Flickr_8k.testImages.txt', 
	    								  word_to_id=self.word_to_id)


	    # load image features
	    # if load_train:
	    #     train_ims = numpy.load(loc+name+'_train_ims.npy')
	    # else:
	    #     train_ims = None
	    # dev_ims = numpy.load(loc+name+'_dev_ims.npy')
	    # test_ims = numpy.load(loc+name+'_test_ims.npy')
	    return train_caps, dev_caps, test_caps

	def get_train_list(self):
	 	train_caps = utils.process_caption(data_path='../../data/Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt', 
	 										word_to_id=self.word_to_id)

	 	return train_caps

	def load_candidate_samples(self, caption, image, candidate):
		samples = []

		for neg in candidate:
			sample.append((caption, image, neg))

		return samples



if __name__ == '__main__':
	loader = data_loader(batch_size=32, dataset_dir='../../data/Flickr8k', max_words=15, is_training=False)
	# test_caps = loader.load_data(load_train=False)[2]
	# print(test_caps[0])
	train_caps = loader.get_train_list()
	print(train_caps[0:3])
