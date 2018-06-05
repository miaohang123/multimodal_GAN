# coding: utf-8
import os
import sys
import re
import math
import numpy as np
import tensorflow.contrib.keras as kr
from collections import Counter
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')

def get_flicker8k_token(datapath):
	token_dict = {}
	with open(datapath, 'r') as f:
		for line in f:
			img_name, caption = line.strip('\n').split('\t')
			key = img_name.split('#')[0]
			if key not in token_dict.keys():
				token_dict[key] = [caption.strip()]
			else:
				token_dict[key].append(caption.strip())

	return token_dict

def get_caption(datapath):
    token_dict = get_flicker8k_token('../data/Flickr8k/Flickr8k_text/Flickr8k.token.txt')
    captions = []
    with open(datapath, 'r') as f:
        for line in f:
            key = line.strip('\n')
            # content = text_to_word_sequence(token_dict[key].strip('\n').strip())
            # captions.append(content)
            for sentence in token_dict[key]:
                captions.append(sentence)
    return captions

def load_txt_caption(datapath):
    captions = []
    with open(datapath, 'r') as f:
        for line in f:
            captions.append(text_to_word_sequence(line.strip()))
    return captions

def build_vocab(train_path, vocab_path, vocab_size=10000):
    """根据训练集构建词汇表，存储"""
    #caption_train = get_caption(train_path)
    caption_train = load_txt_caption(train_path)
    all_data = []

    for content in caption_train:
        #print(content)
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open_file(vocab_path, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_path):
    """读取词汇表"""
    words = open_file(vocab_path).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def process_caption(data_path, word_to_id, max_length=50):
    """将文件转换为id表示"""
    #contents = get_caption(data_path)
    contents = load_txt_caption(data_path)
    data_id = []

    # tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    # tokenizer.fit_on_texts(contents)
    # data_id = tokenizer.texts_to_sequences(contents)
    # print(word_to_id)
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)

    return x_pad

def load_candidate_samples(caption, image, candidate):
        samples = []

        for neg in candidate:
            sample.append((caption, image, neg))

        return samples


def load_val_batch(caption_val, image_feature_val, index, batch):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, batch):
        true_index = index + i
        if (true_index >= len(caption_val)):
            true_index = len(caption_val) - 1
        cap = caption_val[true_index]
        img = image_feature_val[true_index]
        x_train_1.append(cap)
        x_train_2.append(img)
        x_train_3.append(img)
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_sample_val(caption_val, image_feature_val, index, batch_size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, batch_size):
        true_index = index + i
        if (true_index >= len(caption_val)):
            true_index = len(testList) - 1
        # for cap_index in range(len(caption_val)):
        for img_index in range(len(image_feature_val)):

            x_train_1.append(caption_val[true_index])
            x_train_2.append(image_feature_val[img_index])
            x_train_3.append(image_feature_val[img_index])

    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)



def batch_iter(data, batch_size, num_epochs=1, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch =int(math.ceil(len(data)/batch_size))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        print('shuffled data tensor ', shuffled_data.shape)
        for batch_num in range(num_batches_per_epoch):
            # start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[end_index-batch_size:end_index]

def order_violations(s, im):
    """ Computes the order violations (Equation 2 in the paper) """
    return np.power(np.linalg.norm(np.maximum(0, s - im)),2)

def compute_errors(s_emb, im_emb):
    """ Given sentence and image embeddings, compute the error matrix """
    erros = [order_violations(x, y) for x in s_emb for y in im_emb]
    return np.asarray(erros).reshape((len(s_emb), len(im_emb)))


def t2i(c2i, k=5):
    """
    Text->Images (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    vis_details: if true, return a dictionary for ROC visualization purposes
    """

    ranks = np.zeros(c2i.shape[0])

    for i in range(len(ranks)):
        d_i = c2i[i]
        inds = np.argsort(d_i)
        #print(inds[:10])
        #print(np.where(inds == i))
        rank = np.where(inds/1 == i)[0][0]
        ranks[i] = rank

        def image_dict(k):
            return {'id': k, 'score': float(d_i[k])}

    r_k = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
    meanr = ranks.mean() + 1

    stats = map(float, [r_k, meanr])

    return stats

def i2t(c2i, k=5):
    """
    Text->Images (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    """

    ranks = np.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)
        #print(inds[:10])
        rank = np.where(inds/1 == i)[0][0]
        ranks[i] = rank

    # Compute metrics
    r_k = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
    meanr = ranks.mean() + 1

    return map(float, [r_k, meanr])

if __name__ == '__main__':
    token_dict = get_flicker8k_token('../../data/Flickr8k/Flickr8k_text/Flickr8k.token.txt')
    # print(token_dict['2926786902_815a99a154.jpg'])
    # build_vocab(train_path='../../data/Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt', vocab_path='../../data/Flickr8k/Flickr8k_text/vocab.txt')
    words, word_to_id =read_vocab(vocab_path='../../data/Flickr8k/Flickr8k_text/vocab.txt')
    captions = get_caption(datapath='../../data/Flickr8k/Flickr8k_text/Flickr_8k.devImages.txt')
    print(captions[0:2])
    # x = process_caption(data_path='../../data/Flickr8k/Flickr8k_text/Flickr_8k.devImages.txt', word_to_id=word_to_id)
    # print(len(word_to_id))
