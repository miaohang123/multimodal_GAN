#coding=utf-8

import os
import time
import datetime
import operator
import random
import pickle
import math
import numpy as np
import tensorflow as tf
import Discriminator
from keras.backend.tensorflow_backend import set_session
from helper import utils, data_iterator

now = int(time.time()) 
        
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))

# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 50, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 20, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.00001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning_rate (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 2, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("pools_size", 100, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("sampled_size", 100, " the real selectd set from the The sampled pools")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# Data Preparatopn
# flickr8k data path
#vocab_path = '../data/Flickr8k/Flickr8k_text/vocab.txt'
#caption_train_path = '../data/Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
#image_feature_train_path = '../data/Flickr8k/Flickr8k_feature/vgg19_train.npy'
#caption_dev_path = '../data/Flickr8k/Flickr8k_text/Flickr_8k.devImages.txt'
#image_feature_dev_path = '../data/Flickr8k/Flickr8k_feature/vgg19_dev.npy'

# coco data path
vocab_path = '/home/miaohang/myproject/image_caption-master/datasets/coco/vocab.txt'
caption_train_path = '/home/miaohang/myproject/image_caption-master/datasets/coco/train.txt'
image_feature_train_path = '/home/miaohang/myproject/image_caption-master/datasets/coco/images/10crop/train.npy'
caption_dev_path = '/home/miaohang/myproject/image_caption-master/datasets/coco/val.txt'
image_feature_dev_path = '/home/miaohang/myproject/image_caption-master/datasets/coco/images/10crop/val.npy'
caption_test_path = '/home/miaohang/myproject/image_caption-master/datasets/coco/test.txt'
image_feature_test_path = '/home/miaohang/myproject/image_caption-master/datasets/coco/images/10crop/test.npy'

save_dir = 'checkpoints/discriminator'
save_path = os.path.join(save_dir, 'best_validation')  # best val result save path

# ==================================================
# Load data
print(("Loading data..."))

if os.path.exists(vocab_path) == False:
    utils.build_vocab(train_path=caption_train_path, vocab_path=vocab_path)
words, word_to_id = utils.read_vocab(vocab_path=vocab_path)

print('Vocabulary size: ', len(word_to_id))

# train dataset
dataset = {}
dataset['train'] = {}
dataset['train']['caps'] = utils.load_txt_caption(datapath=caption_train_path)#[:5000]#get_caption(datapath=caption_train_path)
dataset['train']['img'] = np.load(image_feature_train_path)#[:1000]#[:, 0, :, :, :]
print('train dataset caption size: ', len(dataset['train']['caps']))
print('train dataset image feature size: ', dataset['train']['img'].shape)

# val dataset
fold = 0
dataset['dev'] = {}
dataset['dev']['caps'] = utils.load_txt_caption(datapath=caption_dev_path)[fold*5000:(fold+1)*5000]#get_caption(datapath=caption_dev_path)
dataset['dev']['img'] = np.load(image_feature_dev_path)[fold*1000:(fold+1)*1000]#[:, 0, :, :, :]


# caption_train = utils.process_caption(data_path=caption_train_path, word_to_id=word_to_id)
# image_feature_train = np.load(image_feature_train_path)[:, 0, :, :, :]

# print('caption train tensor: ', caption_train.shape)
# print('image_feature_train tensor: ', image_feature_train.shape)

# caption_dev = utils.process_caption(data_path=caption_dev_path, word_to_id=word_to_id)
# image_feature_dev = np.load(image_feature_dev_path)[:, 0, :, :, :]

print("Load done...")
if os.path.exists('log') == False:
    os.mkdir('log')
if os.path.exists('checkpoints') == False:
    os.mkdir('checkpoints')
precision = 'log/dev.dns'+timeStamp


from functools import wraps
def log_time_delta(func):
        @wraps(func)
        def _deco(*args, **kwargs):
                start = time.time()
                ret = func(*args, **kwargs)
                end = time.time()
                delta = end - start
                print( "%s runed %.2f seconds"% (func.__name__,delta))
                return ret
        return _deco

@log_time_delta
def generate_uniform_pair():
    samples=[]
    for cap, img in zip(caption_train, image_feature_train):
        #samples.append([cap, img])
        #print('cap tensor: ', cap.shape)
        #print('img tensor: ', img.shape)
        index = np.random.randint(0, len(caption_train) - 1, [1])
        neg= image_feature_train[index]
        for i in range(len(neg)):
            samples.append([cap, img, neg[i]])
    return samples

 

@log_time_delta    
def generate_dns_pair(sess, model):
    samples=[]
    for _index ,pair in enumerate (zip(caption_train, image_feature_train)):
        if _index %100==0:
            print( "have sampled %d pairs" % _index)
        cap=pair[0]
        img=pair[1]
        #print(caption_train.shape)
        pools=image_feature_train[np.random.choice(image_feature_train.shape[0],size=[FLAGS.pools_size])]    
        canditates=utils.load_candidate_samples(cap, img, pools)    
        predicteds=[]

        for batch in utils.batch_iter(canditates,batch_size=FLAGS.batch_size):                            
            feed_dict = {model.input_x_1: batch[:,0],model.input_x_2: batch[:,1],model.input_x_3: batch[:,2]}         
            predicted=sess.run(model.score13,feed_dict)
            predicteds.extend(predicted)    

        index=np.argmax(predicteds)
        samples.append([cap, img, pools[index]])    

    return samples


def dev_step(sess, cnn, caption_dev, image_feature_dev, k=10):
    # scoreList = []
    correct_k = 0
    #print('caption dev tensor: ', len(caption_dev))
    batch_num = int(len(caption_dev)/FLAGS.batch_size)

    feed_dict = {
            cnn.input_caption: caption_dev,
            cnn.input_image_pos: image_feature_dev,
    }

    pred_cap, pred_img = np.array(sess.run([cnn.cap, cnn.img_pos], feed_dict))
    #print('===============')
    #print(pred_cap[0])
    #print('===============')
    #print(pred_img[0])
    #print('pred_cap tensor: ', pred_cap.shape)
    #print('pred_img tensor: ', pred_img.shape)
    dev_errs = utils.compute_errors(pred_cap, pred_img)
    print('dev_errors tensor: ', dev_errs.shape)
    rk_c, rmean_c = utils.t2i(dev_errs, k=k)
    rk_i, rmean_i = utils.i2t(dev_errs, k=k)

    return(rk_c, rmean_c, rk_i, rmean_i)
    # for i in range(5):#(batch_num):
    #     x_test_1, x_test_2, x_test_3 = utils.load_sample_val(caption_dev, image_feature_dev, i * FLAGS.batch_size, FLAGS.batch_size)
    #     #print('x_test_1 tensor: ', x_test_1.shape)
    #     for j in range(FLAGS.batch_size):
    #         cap_index = i * FLAGS.batch_size + j
    #         single_x_test_1 = x_test_1[j*len(caption_dev) : (j+1)*len(caption_dev)]
    #         single_x_test_2 = x_test_2[j*len(caption_dev) : (j+1)*len(caption_dev)]
    #         single_x_test_3 = x_test_3[j*len(caption_dev) : (j+1)*len(caption_dev)]
    #         feed_dict = {
    #             cnn.input_caption: single_x_test_1,
    #             cnn.input_image_pos: single_x_test_2,
    #             cnn.input_image_neg: single_x_test_3
    #             # cnn.dropout_keep_prob: 1.0
    #         }
    #         predicted = np.array(sess.run([cnn.score12], feed_dict)) 
    #         print('predicted score')
    #         print(predicted)
    #         #print(predicted.shape)
    #         #print('caption index: ', cap_index)
    #         predicted_index = np.argsort(-predicted)
    #         #print(predicted_index[0].shape)
    #         #print('predicted index: ', predicted_index[0][:k])
    #         if(cap_index in predicted_index[0][:k]):
    #             # scoreList.append(1)
    #             correct_k += 1
    #         # else:
    #             # scoreList.append(0)
    # return correct_k * 1.0 / (len(caption_dev))
        
    



def evaluation(sess, model, log, num_epochs=0):
    current_step = tf.train.global_step(sess, model.global_step)
    dev_iterator =  data_iterator.data_iterator(dataset['dev'])
    cap_dev, image_feature_dev = dev_iterator.all()
    cap_dev = cap_dev[range(0, len(cap_dev), 5)]
    print('image_feature_dev tensor: ', image_feature_dev.shape)
    print('cap_dev tensor: ', cap_dev.shape)
    rk_c_current, rmean_c_current, rk_i_current, rmean_i_current = dev_step(sess, model, cap_dev, image_feature_dev, k=10)
    #line="test1: %d epoch: precision %f"%(current_step,precision_current)
    print ("Image to text: %.4f %.4f" % (rk_i_current, rmean_i_current))
    print ("Text to image: %.4f %.4f" % (rk_c_current, rmean_c_current))
    # print(model.save_model(sess, rk_i_current, rk_c_current))
    line = "Image to text: %.4f %.4f" % (rk_i_current, rmean_i_current) + " Text to image: %.4f %.4f" % (rk_c_current, rmean_c_current)
    log.write(line+"\n")

    return rk_i_current, rk_c_current

def test():
    print("Loading test data...")
    # test dataset
    fold = 0
    dataset['test'] = {}
    dataset['test']['caps'] = utils.load_txt_caption(datapath=caption_test_path)[fold*5000:(fold+1)*5000]
    dataset['test']['img'] = np.load(image_feature_test_path)[fold*1000:(fold+1)*1000]

    test_iterator =  data_iterator.data_iterator(dataset['test'])
    cap_test, image_feature_test = test_iterator.all()
    cap_test = cap_test[range(0, len(cap_test), 5)]
    print('image_feature_test tensor: ', image_feature_test.shape)
    print('cap_test tensor: ', cap_test.shape)

    sess= tf.Session()
    sess.run(tf.global_variables_initializer())
    #with tf.Graph().as_default():
    saver = tf.train.Saver()#import_meta_graph(os.path.join(save_dir, 'best_validation-106200.meta'))
    
    #ckpt = tf.train.latest_checkpoint(save_path)
    #if ckpt:
    #    saver.restore(sess, ckpt)
    #    print('restore from ckpt{}'.format(ckpt))
    saver.restore(sess=sess, save_path='checkpoints/discriminator/best_validation')  # 读取保存的模型
    #else:
    #    print('cannot restore')

    param = None
    discriminator = Discriminator.Discriminator(
                            sequence_length=FLAGS.max_sequence_length,
                            batch_size=FLAGS.batch_size,
                            vocab_size=len(word_to_id),
                            embedding_size=FLAGS.embedding_dim,
                            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                            num_filters=FLAGS.num_filters,
                            # dropout_keep_prob=1.0,
                            learning_rate=FLAGS.learning_rate,
                            l2_reg_lambda=FLAGS.l2_reg_lambda,
                            embeddings=None,
                            paras=param,
                            loss="pair",
                            trainable=False)

    print('Testing...')
    rk_c_test, rmean_c_test, rk_i_test, rmean_i_test = dev_step(sess, discriminator, cap_test, image_feature_test, k=10)
    print ("Image to text: %.4f %.4f" % (rk_i_test, rmean_i_test))
    print ("Text to image: %.4f %.4f" % (rk_c_test, rmean_c_test))

def main():
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/gpu:1"):
            #configure tensorboard
            # tensorboard_dir = 'tensorboard/discriminator'

            # if not os.path.exists(tensorboard_dir):
            #     os.makedirs(tensorboard_dir)
            saver = tf.train.Saver()
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(graph=graph, config=session_conf)
            with sess.as_default() ,open(precision,"w") as log:
                    # DIS_MODEL_FILE="model/Discriminator20170107122042.model"
                    # param = pickle.load(open(DIS_MODEL_FILE))
                    # print( param)
                    param= None
                    # DIS_MODEL_FILE="model/pre-trained.model"
                    # param = pickle.load(open(DIS_MODEL_FILE,"rb"))
                    discriminator = Discriminator.Discriminator(
                            sequence_length=FLAGS.max_sequence_length,
                            batch_size=FLAGS.batch_size,
                            vocab_size=len(word_to_id),
                            embedding_size=FLAGS.embedding_dim,
                            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                            num_filters=FLAGS.num_filters,
     #                       dropout_keep_prob=FLAGS.dropout_keep_prob,
                            learning_rate=FLAGS.learning_rate,
                            l2_reg_lambda=FLAGS.l2_reg_lambda,
                            embeddings=None,
                            paras=param,
                            loss="pair")

                    #saver = tf.train.Saver()    
                    sess.run(tf.global_variables_initializer())
                    # evaluation(sess,discriminator,log,0)
                    best_average_recall = 0.0
                    total_batch = 0  
                    last_improved = 0  # 记录上一次提升批次
                    require_improvement = 50  # 如果超过1000轮未提升，提前结束训练
                    flag = False
                    last_step = 0
                    for epoch in range(FLAGS.num_epochs):
                        print('Epoch:', epoch + 1)
                        # x1,x2,x3=generate_dns(sess,discriminator)
                        # samples=generate_dns(sess,discriminator)#generate_uniform_pair() #generate_dns(sess,discriminator) #generate_uniform() #                        
                        # samples=generate_dns_pair(sess,discriminator) #generate_uniform() # generate_uniform_pair() #                     
                        # samples = generate_uniform_pair()
                        train_iterator = data_iterator.data_iterator(dataset['train'], batch_size=FLAGS.batch_size)
                        #print('==========================')
                        for batch in train_iterator:#utils.batch_iter(samples,batch_size=FLAGS.batch_size,num_epochs=1,shuffle=True):
                           
                            cap_sample = batch[0]#np.array([item[0] for item in batch])
                            #print('cap_sample tensor: ', cap_sample.shape)
                            img_pos_sample = batch[1]#np.array([item[1] for item in batch])
                            #print(img_pos_sample.shape)
                            img_neg_sample = batch[1]#np.array([item[2] for item in batch])

                            feed_dict = {
                                    discriminator.input_caption: cap_sample,
                                    discriminator.input_image_pos: img_pos_sample,
                                    discriminator.input_image_neg: img_neg_sample,
                                 
                                }
                            
                         
                            _, step, current_loss, accuracy, positive, negative = sess.run(
                                    [discriminator.train_op, discriminator.global_step, discriminator.loss,discriminator.accuracy, discriminator.positive, discriminator.negative],
                                    feed_dict)
                            time_str = datetime.datetime.now().isoformat()
                            if(step % 100 == 0):
                                print(("%s: DIS step %d, loss %f with acc %f, positive score %f and negative score %f "%(time_str, step, current_loss,accuracy, positive, negative)))
                            last_step = step
                            # evaluate the performance of the model on the val dataset
                        if((total_batch+1) % FLAGS.evaluate_every == 0):
                            rk_i_current, rk_c_current = evaluation(sess, discriminator, log, epoch)

                            # save the best performed model
                            if (rk_i_current + rk_c_current) * 0.5 > best_average_recall:
                                best_average_recall = (rk_i_current + rk_c_current) * 0.5   
                                last_improved = total_batch
                                saver.save(sess=sess, save_path=save_path, global_step=last_step)
                                improved_str = '*'
                                break
                            else:
                                improved_str = ''
                        
                        total_batch += 1

                        if total_batch - last_improved > require_improvement:
                            # early stop
                            print("No optimization for a long time, auto-stopping...")
                            flag = True
                            break  

                        if flag:
                            break

if __name__ == '__main__':
    main()
    test()
                 
