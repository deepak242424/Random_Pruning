import cPickle as pkl
import numpy as np

IMAGE_HEIGHT = 224   
IMAGE_WIDTH = 224
num_classes = 1000
num_data_samples = 1281167
num_valid_samples = 50*1000

data_dir = "/datasets/data1/imagenet_data/"
save_dir = "models/model"
checkpoint = save_dir
logfile = "logs.txt"

dataset_split_name = "train"
fc_conv_padding='VALID'
mask = pkl.load(open('random_mask.save','rb'))

# TRAIN
batch_size = 50
num_threads = 5
learning_rate = 0.0001

num_epochs = 1
num_batches = int(num_data_samples/batch_size)

# OPTIMIZER
rmsprop_decay = 0.9
momentum = 0.9
opt_epsilon = 1.0
learning_rate_decay_type = 'exponential'
num_epochs_per_decay = 2.0
learning_rate_decay_factor =  0.94

# Pruning 
list_filters = [ 64,   64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
prune_factor = [0.5,  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,   0,   0]
# mask = []
# for k in range(len(list_filters)):
#     mask_temp = np.zeros(list_filters[k])
#     idx = np.argsort(np.asarray(entropy[k]))[int(list_filters[k]*prune_factor[k]): ]
#     # idx = np.argsort(np.asarray(entropy[k]))[:]
#     mask_temp[idx] = 1
#     print(np.sum(mask_temp))
#     mask.append(mask_temp)



def get_data_files(dataset_split_name):
    tfrecords_filename = []
    if dataset_split_name == 'train':
        length = 1024
        data = "-of-01024"
    if dataset_split_name == 'validation':
        length = 128
        data = "-of-00128"
    
    for k in range(length): # Train data tfrecords
        j = "-0%04d"%k        # pad with 0's
        tfrecords_filename.append(data_dir+dataset_split_name+j+data)
    return tfrecords_filename
