import os
import sys
import re
import random
import numpy as np
import tensorflow.contrib.keras as kr
from collections import Counter
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from helper import utils

class data_iterator(object):
	def __init__(self, data, batch_size=32, max_cap_length=50):
		self.data = data
		self.batch_size = batch_size
		self.max_cap_length = max_cap_length
		self.num_images = len(self.data['img'])
		_, self.word_to_id = utils.read_vocab(vocab_path='../data/Flickr8k/Flickr8k_text/vocab.txt')
		#print(self.num_images)
		#print(len(self.data['caps']))
		self.reset()
	
	def reset(self):
		self.idx = 0
		self.order = np.random.permutation(self.num_images)

	def __next__(self):
		image_ids = []
		caption_ids = []
		
		while len(image_ids) < self.batch_size:
			image_id = self.order[self.idx]
			# print('image_id: ', image_id)
			caption_id = image_id * 5 + random.randrange(5)
			image_ids.append(image_id)
			caption_ids.append(caption_id)

			self.idx += 1
			if self.idx >= self.num_images:
				self.reset()
				raise StopIteration()

		x = self.prepare_caps(caption_ids)
		im = self.data['img'][np.array(image_ids)]

		return x, im

	def all(self):
		return self.prepare_caps(range(0,len(self.data['caps']))) , self.data['img']

	def __iter__(self):
		return self

	def prepare_caps(self, indices):
	    seqs = []
	    for i in indices:
	        cc = self.data['caps'][i]#text_to_word_sequence(self.data['caps'][i])
	        data_id = [self.word_to_id[w] if w in self.word_to_id else 1 for w in cc]
	        padding = [0]*(self.max_cap_length-len(data_id))
	        seqs.append(data_id + padding)

	    return np.array(seqs)


if __name__ == '__main__':
	dataset = {}
	dataset['dev'] = {}
	dataset['dev']['caps'] = utils.get_caption(datapath='../../data/Flickr8k/Flickr8k_text/Flickr_8k.devImages.txt')
	dataset['dev']['img'] = np.zeros([1000, 5])
	iterator = data_iterator(dataset['dev'])
	for batch in iterator:
		print(batch[0].shape, batch[1].shape)
	# print(batch_cap)
	# print(batch_img)
