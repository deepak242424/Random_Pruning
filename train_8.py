
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import skimage.io as io
import vgg_preprocessing as preprocess
import numpy as np
from tqdm import tqdm
from IPython import embed
import datetime,argparse
import cPickle as pkl
from config import *

def parse_args():
  global model_number,checkpoint_number
  parser = argparse.ArgumentParser()
  parser.add_argument("--model")
  parser.add_argument("--checkpoint")
  args = parser.parse_args()
  model_number = args.model
  checkpoint_number = args.checkpoint


def weight_variable(shape, name):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape,name):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def fetch_images_valid(serialized_example, IMAGE_HEIGHT=224, IMAGE_WIDTH=224,is_training=False):
    features = tf.parse_single_example(
          serialized_example, features = {
          'image/encoded': tf.FixedLenFeature(
              (), tf.string, default_value=''),
          'image/format': tf.FixedLenFeature(
              (), tf.string, default_value='jpeg'),
          'image/class/label': tf.FixedLenFeature(
              [], dtype=tf.int64, default_value=-1),
          'image/class/text': tf.FixedLenFeature(
              [], dtype=tf.string, default_value=''),
          'image/height' : tf.FixedLenFeature(
            [],dtype=tf.int64),
          'image/width' :  tf.FixedLenFeature(
            [],dtype=tf.int64),
          'image/channels' :  tf.FixedLenFeature(
            [],dtype=tf.int64),
          'image/object/class/label': tf.VarLenFeature(
              dtype=tf.int64),
      })

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    label = tf.cast(features['image/class/label'],tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    num_channels = tf.cast(features['image/channels'], tf.int32)
    image_shape = tf.stack([height,width,num_channels])
    image = tf.cast(image, tf.float32)
    preprocessed_image = preprocess.preprocess_image(image,IMAGE_HEIGHT,IMAGE_WIDTH,is_training=False)
    images, labels = tf.train.batch( [preprocessed_image, label],
                                                     batch_size=batch_size,
                                                     capacity=batch_size+num_threads*batch_size,
                                                     num_threads=num_threads)
    return images,labels

def fetch_images_train(serialized_example, IMAGE_HEIGHT=224, IMAGE_WIDTH=224,is_training=True):
    features = tf.parse_single_example(
          serialized_example, features = {
          'image/encoded': tf.FixedLenFeature(
              (), tf.string, default_value=''),
          'image/format': tf.FixedLenFeature(
              (), tf.string, default_value='jpeg'),
          'image/class/label': tf.FixedLenFeature(
              [], dtype=tf.int64, default_value=-1),
          'image/class/text': tf.FixedLenFeature(
              [], dtype=tf.string, default_value=''),
          'image/height' : tf.FixedLenFeature(
            [],dtype=tf.int64),
          'image/width' :  tf.FixedLenFeature(
            [],dtype=tf.int64),
          'image/channels' :  tf.FixedLenFeature(
            [],dtype=tf.int64),
          'image/object/class/label': tf.VarLenFeature(
              dtype=tf.int64),
      })

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    label = tf.cast(features['image/class/label'],tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    num_channels = tf.cast(features['image/channels'], tf.int32)
    image_shape = tf.stack([height,width,num_channels])
    image = tf.cast(image, tf.float32)
    preprocessed_image = preprocess.preprocess_image(image,IMAGE_HEIGHT,IMAGE_WIDTH,is_training=is_training)
    images, labels = tf.train.shuffle_batch( [preprocessed_image, label],
                                                     batch_size=batch_size,
                                                     capacity=batch_size+num_threads*batch_size,
                                                     num_threads=num_threads,
                                                     min_after_dequeue=batch_size)
    return images,labels


def configure_lr(global_step,learning_rate):
    decay_steps = int(num_data_samples / batch_size *
                    num_epochs_per_decay)
    return tf.train.exponential_decay(learning_rate,global_step,
                    decay_steps,learning_rate_decay_factor,
                    staircase=True,name='exponential_decay_learning_rate')


parse_args()
# placeholders
labels = tf.placeholder(dtype= tf.float32,shape=[batch_size,1])
images = tf.placeholder(dtype= tf.float32,shape=[batch_size,224,224,3])
keep_prob6 = tf.placeholder(dtype= tf.float32)
keep_prob7 = tf.placeholder(dtype= tf.float32)

# sval_acc = tf.placeholder(dtype= tf.float32)
# tf.summary.scalar('Validation Accuracy', sval_acc)
# strain_acc = tf.placeholder(dtype= tf.float32)
# tf.summary.scalar('Training Accuracy', strain_acc)

# Generate Mask
# mask = []
# for k in range(len(list_filters)):
#     mask_temp = np.zeros(list_filters[k])
#     idx = np.argsort(np.asarray(entropy[k]))[int(list_filters[k]*prune_factor[k]):]
#     # idx = np.argsort(np.asarray(entropy[k]))[:]
#     mask_temp[idx] = 1
#     mask.append(mask_temp)

# Model
with tf.variable_scope('vgg_16'):
    with tf.variable_scope('conv1'):
        W_conv1_1 = weight_variable([3,3,3,64], name="conv1_1/weights")
        b_conv1_1 = bias_variable([64], name="conv1_1/biases")
        W_conv1_2 = weight_variable([3,3,64,64],name="conv1_2/weights")
        b_conv1_2 = bias_variable([64],name="conv1_2/biases")

    with tf.variable_scope('conv2'):
        W_conv2_1 = weight_variable([3,3,64,128],name="conv2_1/weights")
        b_conv2_1 = bias_variable([128],name="conv2_1/biases")
        W_conv2_2 = weight_variable([3,3,128,128],name="conv2_2/weights")
        b_conv2_2 = bias_variable([128],name="conv2_2/biases")

    with tf.variable_scope('conv3'):    
        W_conv3_1 = weight_variable([3,3,128,256],name="conv3_1/weights")
        b_conv3_1 = bias_variable([256],name="conv3_1/biases")
        W_conv3_2 = weight_variable([3,3,256,256],name="conv3_2/weights")
        b_conv3_2 = bias_variable([256],name="conv3_2/biases")
        W_conv3_3 = weight_variable([3,3,256,256],name="conv3_3/weights")
        b_conv3_3 = bias_variable([256],name="conv3_3/biases")

    with tf.variable_scope('conv4'):           
        W_conv4_1 = weight_variable([3,3,256,512],name="conv4_1/weights")
        b_conv4_1 = bias_variable([512],name="conv4_1/biases")
        W_conv4_2 = weight_variable([3,3,512,512],name="conv4_2/weights")
        b_conv4_2 = bias_variable([512],name="conv4_2/biases")
        W_conv4_3 = weight_variable([3,3,512,512],name="conv4_3/weights")
        b_conv4_3 = bias_variable([512],name="conv4_3/biases")

    with tf.variable_scope('conv5'):           
        W_conv5_1 = weight_variable([3,3,512,512],name="conv5_1/weights")
        b_conv5_1 = bias_variable([512],name="conv5_1/biases")
        W_conv5_2 = weight_variable([3,3,512,512],name="conv5_2/weights")
        b_conv5_2 = bias_variable([512],name="conv5_2/biases")
        W_conv5_3 = weight_variable([3,3,512,512],name="conv5_3/weights")
        b_conv5_3 = bias_variable([512],name="conv5_3/biases")


    W_fc6 = weight_variable([7,7,512,4096],name="fc6/weights")
    b_fc6 = weight_variable([4096],name="fc6/biases")
    W_fc7 = weight_variable([1,1,4096,4096],name="fc7/weights")
    b_fc7 = bias_variable([4096],name="fc7/biases")
    W_fc8 = weight_variable([1,1,4096,num_classes],name="fc8/weights")
    b_fc8 = bias_variable([num_classes],name="fc8/biases")
    
#Inference

# layer 1
a_conv1_1 = tf.nn.bias_add(conv2d(images, W_conv1_1),b_conv1_1)
# a_conv1_1_masked = tf.multiply(a_conv1_1,mask[0])
h_conv1_1 = tf.nn.relu(a_conv1_1)
a_conv1_2 = tf.nn.bias_add(conv2d(h_conv1_1, W_conv1_2),b_conv1_2)
# a_conv1_2_masked = tf.multiply(a_conv1_2,mask[1])
h_pool1   = max_pool_2x2(tf.nn.relu(a_conv1_2))


# layer 2
a_conv2_1 = tf.nn.bias_add(conv2d(h_pool1, W_conv2_1),b_conv2_1)
# a_conv2_1_masked = tf.multiply(a_conv2_1, mask[2])
h_conv2_1 = tf.nn.relu(a_conv2_1)
a_conv2_2 = tf.nn.bias_add(conv2d(h_conv2_1, W_conv2_2),b_conv2_2)
# a_conv2_2_masked = tf.multiply(a_conv2_2, mask[3])
h_pool2   = max_pool_2x2(tf.nn.relu(a_conv2_2))


# layer 3
a_conv3_1 = tf.nn.bias_add(conv2d(h_pool2, W_conv3_1),b_conv3_1)
# a_conv3_1_masked = tf.multiply(a_conv3_1,mask[4])
h_conv3_1 = tf.nn.relu(a_conv3_1)
a_conv3_2 = tf.nn.bias_add(conv2d(h_conv3_1, W_conv3_2),b_conv3_2)
# a_conv3_2_masked = tf.multiply(a_conv3_2, mask[5])
h_conv3_2 = tf.nn.relu(a_conv3_2)
a_conv3_3 = tf.nn.bias_add(conv2d(h_conv3_2, W_conv3_3),b_conv3_3)
# a_conv3_3_masked = tf.multiply(a_conv3_3, mask[6])
h_pool3   = max_pool_2x2(tf.nn.relu(a_conv3_3))

a_conv4_1 = tf.nn.bias_add(conv2d(h_pool3, W_conv4_1),b_conv4_1)
a_conv4_1_masked = tf.multiply(a_conv4_1, mask[7])
h_conv4_1 = tf.nn.relu(a_conv4_1_masked)
a_conv4_2 = tf.nn.bias_add(conv2d(h_conv4_1, W_conv4_2),b_conv4_2)
a_conv4_2_masked = tf.multiply(a_conv4_2, mask[8])
h_conv4_2 = tf.nn.relu(a_conv4_2_masked)
a_conv4_3 = tf.nn.bias_add(conv2d(h_conv4_2, W_conv4_3),b_conv4_3)
a_conv4_3_masked = tf.multiply(a_conv4_3, mask[9])
h_pool4   = max_pool_2x2(tf.nn.relu(a_conv4_3_masked))

a_conv5_1 = tf.nn.bias_add(conv2d(h_pool4, W_conv5_1),b_conv5_1)
a_conv5_1_masked = tf.multiply(a_conv5_1, mask[10])
h_conv5_1 = tf.nn.relu(a_conv5_1_masked)
a_conv5_2 = tf.nn.bias_add(conv2d(h_conv5_1, W_conv5_2),b_conv5_2)
# a_conv5_2_masked = tf.multiply(a_conv5_2, mask[11])
h_conv5_2 = tf.nn.relu(a_conv5_2)
a_conv5_3 = tf.nn.bias_add(conv2d(h_conv5_2, W_conv5_3),b_conv5_3)
# a_conv5_3_masked = tf.multiply(a_conv5_3, mask[12])
h_pool5   = max_pool_2x2(tf.nn.relu(a_conv5_3))

# In place of FC, use conv2d with VALID padding

a_fc6 = tf.nn.bias_add(tf.nn.conv2d(h_pool5, W_fc6, strides = [1,1,1,1],padding=fc_conv_padding),b_fc6)
h_fc6 = tf.nn.relu(a_fc6)
d_fc6 = tf.nn.dropout(h_fc6,keep_prob=keep_prob6)

a_fc7 = tf.nn.bias_add(conv2d(d_fc6, W_fc7),b_fc7)
h_fc7 = tf.nn.relu(a_fc7)
d_fc7 = tf.nn.dropout(h_fc7,keep_prob=keep_prob7)

a_fc8  = tf.nn.bias_add(conv2d(d_fc7, W_fc8),b_fc8)
logits = tf.squeeze(a_fc8,[1,2])

tf.logging.set_verbosity(tf.logging.INFO)

# Input pipeline for validation data
list_filenames = get_data_files('validation')
filename_queue = tf.train.string_input_producer(
     list_filenames, capacity=batch_size, shuffle=False )
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
[valid_images,valid_labels] = fetch_images_valid(serialized_example)
valid_labels -= 1
valid_labels = tf.cast(valid_labels,tf.int64)
valid_labels = tf.squeeze(valid_labels)

# Input pipeline for train data
list_filenames_train = get_data_files('train')
filename_queue_train = tf.train.string_input_producer(
     list_filenames_train, capacity=batch_size, shuffle=True )
reader_train = tf.TFRecordReader()
_, serialized_example_train = reader_train.read(filename_queue_train)
[train_images,train_labels] = fetch_images_train(serialized_example_train)
train_labels -= 1
train_labels = tf.cast(train_labels,tf.int64)
train_labels = tf.squeeze(train_labels)


labels = tf.cast(labels,tf.int64)
labels = tf.squeeze(labels)
lbl_one_hot = tf.one_hot(labels, num_classes, 1.0, 0.0)

prediction = tf.argmax(logits,1)
correct_prediction = tf.reduce_sum(tf.cast(tf.equal(prediction,labels),tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= lbl_one_hot, logits= logits)) 

# optimization
global_step = tf.Variable(0,name='global_step',trainable=False)
learning_rate = configure_lr(global_step,learning_rate)
all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="vgg_16")
optimizer = tf.train.RMSPropOptimizer(
            learning_rate,decay=rmsprop_decay,
            momentum=momentum,epsilon=opt_epsilon)
train_step = optimizer.minimize(cross_entropy,global_step=global_step,var_list= all_vars)

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

# graph ends ! you can create session now!

# merged = tf.summary.merge_all()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init_op)
    list_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_16')
    saver = tf.train.Saver(list_vars)
    saver.restore(sess, checkpoint+str(checkpoint_number)+"/model.ckpt")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess = sess)

    # train_writer = tf.summary.FileWriter(summary_location, sess.graph)
    steps = num_batches*num_epochs

    train_instances = num_batches*batch_size
    val_instances = (int((num_valid_samples)/batch_size))*batch_size
    epoch_counter = 0
    best_Acc = 0

    train_acc = 0
    for i in tqdm(range(steps)):

        if i%(num_batches-1) ==0 :
            valid_acc = 0
            for k in tqdm(range(int((num_valid_samples)/batch_size))):
                valid_img,valid_lbl = sess.run([valid_images,valid_labels])
                # valid_lbl = np.reshape(valid_lbl,[batch_size,1])
                acc = sess.run(correct_prediction,feed_dict={labels:valid_lbl, images:valid_img, keep_prob6:1.0, keep_prob7:1.0})
                valid_acc += acc
 
            epoch_val_acc = float(valid_acc)/val_instances
            epoch_train_acc = float(train_acc)/train_instances

            if epoch_val_acc > best_Acc:
                saver.save(sess,save_dir+str(model_number)+'/model.ckpt')
                best_Acc = epoch_val_acc
            file_ptr = open(logfile,'a')
            file_ptr.write(str(model_number)+","+str(best_Acc)+"\n")
            file_ptr.close()
            # _merged = sess.run(merged, feed_dict={strain_acc: epoch_train_acc, sval_acc:epoch_val_acc})
            # train_writer.add_summary(_merged, epoch_counter)
            epoch_counter += 1
            train_acc = 0

            print('Epoch number: ', epoch_counter-1)
            print('epoch_train_acc: ', epoch_train_acc)
            print('epoch_val_acc: ', epoch_val_acc)
            print('Best val till: ', best_Acc)

        img,lbl = sess.run([train_images,train_labels])
        # lbl = np.reshape(lbl,[batch_size,1])
        loss, pred,true_lbl,c_pred,_ = sess.run([cross_entropy,prediction,labels,correct_prediction,train_step],feed_dict={labels:lbl, images:img, keep_prob6:0.7, keep_prob7:0.7})
        train_acc += c_pred

    

    coord.request_stop()
    coord.join(threads)
