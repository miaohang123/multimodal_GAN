#coding=utf-8
import os
import tensorflow as tf 
import numpy as np 
import  pickle
import time

class MCNN():
    
    def __init__(self, sequence_length, batch_size,vocab_size, embedding_size,filter_sizes, num_filters, dropout_keep_prob=1.00,l2_reg_lambda=0.0,paras=None,learning_rate=1e-2,embeddings=None,loss="pair",trainable=True):
        self.sequence_length=sequence_length
        self.learning_rate=learning_rate
        self.paras=paras
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.l2_reg_lambda=l2_reg_lambda
        self.dropout_keep_prob = dropout_keep_prob
        self.embeddings=embeddings
        self.embedding_size=embedding_size
        self.batch_size=batch_size
        self.model_type="base"
        self.num_filters_total=self.num_filters * len(self.filter_sizes)
        self.trainable = trainable

        self.input_caption = tf.placeholder(tf.int32, [None, sequence_length], name="input_caption")
        self.input_image_pos = tf.placeholder(tf.float32, [None, 4096], name="input_image_pos")#tf.placeholder(tf.float32, [None, 14, 14, 512], name="input_image_pos")
        self.input_image_neg = tf.placeholder(tf.float32, [None, 4096], name="input_image_neg")
        
        #self.keep_prob = tf.placeholder(tf.float32)
        #self.input_image_neg = tf.placeholder(tf.float32, [None, 14, 14, 512], name="input_image_neg")

        # self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_1")
        # self.input_x_2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_2")
        # self.input_x_3 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_3")
        
        self.label=tf.placeholder(tf.float32, [batch_size], name="binary_label")
        
        # Embedding layer
        self.updated_paras=[]
        with tf.name_scope("embedding"):
            if self.paras==None:
                if self.embeddings ==None:
                    print ("random embedding")
                    self.Embedding_W = tf.Variable(
                        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                        name="random_W")
                else:
                    self.Embedding_W = tf.Variable(np.array(self.embeddings),name="embedding_W" ,dtype="float32",trainable=trainable)
            else:
                print ("load embeddings")
                self.Embedding_W=tf.Variable(self.paras[0],trainable=trainable,name="embedding_W")
            self.updated_paras.append(self.Embedding_W)

        #caption CNN
        self.kernels=[]        
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, self.num_filters]
                if self.paras==None:
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="kernel_W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="kernel_b")
                    self.kernels.append((W,b))
                else:
                    _W,_b=self.paras[1][i]
                    W=tf.Variable(_W)                
                    b=tf.Variable(_b)
                    self.kernels.append((W,b))   
                self.updated_paras.append(W)
                self.updated_paras.append(b)

        
        # L2 loss
        self.l2_loss = tf.constant(0.0)
        for para in self.updated_paras:
            self.l2_loss+= tf.nn.l2_loss(para)
        

        with tf.name_scope("output"):
            self.cap =self.getCaptionRNNRepresentation(self.input_caption)
            cap_size = self.cap.get_shape().as_list()[1]
            print('caption representation tensor; ', self.cap.get_shape().as_list())
            self.img_pos=self.getImgDenseRepresentation(self.input_image_pos, cap_size)
            print('img_pos tensor: ', self.img_pos.get_shape().as_list())
            self.img_neg=self.getImgDenseRepresentation(self.input_image_neg, cap_size)

            self.score12 = self.cosine(self.cap, self.img_pos)
            self.score13 = self.cosine(self.cap, self.img_neg)

            self.positive= tf.reduce_mean(self.score12)
            self.negative= tf.reduce_mean( self.score13)

    def getCaptionCNNRepresentation(self,sentence):
        embedded_chars_1 = tf.nn.embedding_lookup(self.Embedding_W, sentence)
        embedded_chars_expanded_1 = tf.expand_dims(embedded_chars_1, -1)
        output=[]
        for i, filter_size in enumerate(self.filter_sizes): 
            conv = tf.nn.conv2d(
                embedded_chars_expanded_1,
                self.kernels[i][0],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="poll-1"
            )
            output.append(pooled)
        pooled_reshape = tf.reshape(tf.concat(output, 3), [-1, self.num_filters_total]) 
        pooled_flat = tf.nn.dropout(pooled_reshape, self.dropout_keep_prob)
        return pooled_flat

    def getCaptionRNNRepresentation(self, sentence, hidden_size=256):
        embedded_chars_1 = tf.nn.embedding_lookup(self.Embedding_W, sentence)
        embedded_chars_expanded_1 = tf.expand_dims(embedded_chars_1, -1)
        print('embedded_chars_expanded tensor: ', embedded_chars_expanded_1.get_shape().as_list())
        gru_cell = tf.contrib.rnn.GRUCell(hidden_size)
        cell = tf.contrib.rnn.DropoutWrapper(gru_cell)#, output_keep_prob=self.dropout_keep_prob)
        outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=embedded_chars_1, time_major=False, dtype=tf.float32)
        print('outputs tensor: ', outputs.get_shape().as_list())
        last = outputs[:, -1, :]
        print('last tensor: ', last.get_shape().as_list())
        last = tf.nn.l2_normalize(last, dim=1)
        last = tf.abs(last)
        return last


    def getImgDenseRepresentation(self, image, hidden_size):
        flat = image
        #flat = tf.reshape(image, [-1, image.get_shape().as_list()[1]*image.get_shape().as_list()[2]*image.get_shape().as_list()[3]])
        print('flat tensor: ', flat.get_shape().as_list())
        W = tf.Variable(tf.truncated_normal([flat.get_shape().as_list()[-1], hidden_size],
                                            stddev=0.1,dtype=tf.float32), name='W')
        # Zero initialization
        b = tf.Variable(tf.constant(0.01, dtype = tf.float32, shape=[hidden_size],name='b'))

        logits = tf.nn.relu(tf.matmul(flat, W) + b)
        #if self.trainable == True:
        #    logits = tf.nn.dropout(logits, self.dropout_keep_prob)
            
        logits = tf.abs(tf.nn.l2_normalize(logits, dim=1))
    
        return logits
        

    def cosine(self, cap, img):

        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(cap, cap), 1)) 
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(img, img), 1))

        pooled_mul_12 = tf.reduce_sum(tf.multiply(cap, img), 1) 
        score = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2)+1e-8, name="scores") 
        return score 
      
    def l2_dis(self, cap, img):
        score = tf.reduce_sum(tf.square(cap - img), 1)
        return score
    
    def save_model(self, sess, rk_c_current=0, rk_i_current=0):

        now = int(time.time())             
        timeArray = time.localtime(now)
        timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
        filename="model/"+self.model_type+str(rk_c_current) + "&" + str(rk_i_current) +"-"+timeStamp+".model"

        param = sess.run([self.Embedding_W,self.kernels])
        pickle.dump(param, open(filename, 'wb+'))
        return filename


