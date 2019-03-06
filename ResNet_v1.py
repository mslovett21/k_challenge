%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import numpy as np
import re

# image dimensions


img_height = 32
img_width = 32
img_channels = 1

### Data dimensions
sample_size       = 10000
fingerprint_size  = 1024
fingerprint_width = 32
targets_num       = 420
weights_num       = 420
num_channels      = 1


BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

# Load Data
drug_fingerprints_fh = "./sample/sample_fingerprints.csv"
drug_targets_fh      = "./sample/sample_targets.csv"
drug_weights_fh      = "./sample/sample_weights.csv"


def populate_data(file_handle,data_matrix, data_size):
    with open(file_handle) as fh:
        j=0
        content = fh.readlines()
        content = [x.strip() for x in content]
        for line in content:
            result = re.split(r'[,\t]\s*',line)
            for i in range(1,data_size+1):
                data_matrix[j][i-1] = np.float32(result[i])
            j = j+1
    print(j)
    fh.close()
    
drug_fingerprints = []
drug_targets      = []
drug_weights      = []


for i in range(sample_size):
    fingerprint_holder = [0]* fingerprint_size
    drug_fingerprints.append(fingerprint_holder)
    
for i in range(sample_size):
    target_holder = [0]* targets_num
    drug_targets.append(target_holder)

for i in range(sample_size):
    weight_holder = [0]* weights_num
    drug_weights.append(weight_holder)
    
populate_data(drug_weights_fh, drug_weights, weights_num)
populate_data(drug_targets_fh, drug_targets, targets_num)
populate_data(drug_fingerprints_fh, drug_fingerprints, fingerprint_size)


drug_fingerprints = np.array(drug_fingerprints)
drug_targets      = np.array(drug_targets)
drug_weights      = np.array(drug_weights)

### PARAMETERS OF THE NETWORK
is_training          = True
data_format          = 'channels_first'
filter_size_par      = 3
num_classes          = 420

block_1_filters      = 64
num_blocks_1         = 3

block_2_filters      = 128
num_blocks_2         = 4

block_3_filters      = 256
num_blocks_3         = 6

block_4_filters      = 512
num_blocks_4         = 3




# 1. Placeholders

x = tf.placeholder(tf.float32, [None, fingerprint_size],name = "In_Flat_Drug_Fingerprint")

drug_image = tf.reshape(x, [-1, fingerprint_width, fingerprint_width, num_channels], name="Drug_Image_32x32")

y_true = tf.placeholder(tf.float32, [None, targets_num],name='True_Labels')

cross_entropy_weights = tf.placeholder(tf.float32, [None, weights_num],name = "Cross_Entropy_Weights")

inputs = tf.transpose(drug_image, [0, 3, 1, 2])



# 2. Variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name="Weights")
  
  
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]), name="Biases")
  
 
# Network Description
#*  **conv1** (7x7 conv, 64,/2) -> ***filter_size***=7, ***out_channels***=64, ***stride***=2
#*  **pooling layer** ***stride***=2
#*  **block1** layers 6x[(3x3 con,64,)] -> 6 conv layers with:    ***filter_size***=3, ***out_channels***=64, ***stride***=1
#*  **block2** layers 8x[(3x3 con,128,)] -> 8 conv layers with:   ***filter_size***=3, ***out_channels***=128, ***stride***=1
#*  **block3** layers 12x[(3x3 con,256,)] -> 12 conv layers with: ***filter_size***=3, ***out_channels***=265, ***stride***=1 
#*  **block4** layers 6x[(3x3 con,512,)] -> 6 conv layers with:   ***filter_size***=3, ***out_channels***=512, ***stride***=1
#* **average pooling**
#* **fully connected layer**



### HELPERS FUNCTION

def fixed_padding(inputs, kernel_size, data_format):
    
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],[pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs
  
  
  
def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):

  if strides > 1:
      inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                          padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
                          kernel_initializer=tf.variance_scaling_initializer(), data_format=data_format)

  
def batch_norm_relu(inputs, is_training, relu=True, init_zero=False, data_format='channels_first'):
    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()
        
    axis = 3
    inputs = tf.layers.batch_normalization(inputs=inputs, axis=axis, momentum=BATCH_NORM_DECAY,
                                           epsilon=BATCH_NORM_EPSILON, center=True, scale=True, 
                                           training=is_training,fused=True, gamma_initializer=gamma_initializer)
    if relu:
        inputs = tf.nn.relu(inputs) 
    
    return inputs
  

### ORIGINAL IMPLEMENTATION OF THE RESIDUAL UNIT
def residual_unit(inputs, filters, is_training, strides,
                 use_projection=False, data_format='channels_first'):
  shortcut = inputs

  if use_projection:
      shortcut = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=strides,
      data_format=data_format)

  shortcut = batch_norm_relu(shortcut, is_training, relu=False,
                             data_format=data_format)

  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=3, strides=strides,
    data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=3, strides=1,
    data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, relu=False, init_zero=True,
                         data_format=data_format)


  sum_with_shortcut = inputs + shortcut

  return tf.nn.relu( sum_with_shortcut )

  
  
  
  
def block_group(inputs, filters, blocks, strides, is_training, name,data_format='channels_first'):

  inputs = residual_unit(inputs, filters, is_training, strides,
                  use_projection=True, data_format=data_format)

  for _ in range(1, blocks):
      inputs = residual_unit(inputs, filters, is_training, 1,
                    data_format=data_format)

  return tf.identity(inputs, name)



def new_fc_layer(input, num_inputs,num_outputs): 

  # new weights and biases for the layer
  weights = new_weights(shape = [num_inputs, num_outputs])
  biases = new_biases(length = num_outputs)

  # calculate the layer as the matrix multiplication of the input and weights, and then add the bias-values.
  layer = tf.matmul(input, weights) + biases

  layer = tf.nn.sigmoid(layer,name = "FULLY_CONNECTED_WITH_SIGMOID")

  return layer






 ## INPUT
#**Image** of shape `num_channels` by`fingerprint_size` by `fingerprint_size`

inputs = tf.transpose(drug_image, [0, 3, 1, 2])


conv1 = conv2d_fixed_padding(inputs=inputs, filters= 64 , kernel_size= 7,strides= 2 , data_format=data_format)

conv1 = batch_norm_relu(conv1, is_training=is_training, data_format='channels_first')

pooling_layer = tf.layers.max_pooling2d(inputs=conv1, pool_size=3,strides=2, padding='SAME',data_format='channels_first')


### BLOCK 1
#layers 6x[(3x3 con,64,)] -> 6 conv layers with:    ***filter_size***=3, ***out_channels***=64, 
#or 3 times residual unit

block1= block_group(inputs=pooling_layer, filters=block_1_filters , blocks=num_blocks_1,strides=1, is_training=is_training,
                      name='BLOCK_1',data_format=data_format)


### BLOCK 2
#layers 8x[(3x3 con,128,)] -> 8 conv layers with:    ***filter_size***=3, ***out_channels***=128, 
#or 4 times residual unit

block2 = block_group(inputs=block1, filters=block_2_filters , blocks=num_blocks_2, strides=2, is_training=is_training,
                      name='BLOCK_2',data_format=data_format)


### BLOCK 3
#layers 12 x[(3x3 con,256,)] -> 12 conv layers with:    ***filter_size***=3, ***out_channels***=256, 
#or 6 times residual unit

block3 = block_group(inputs=block2, filters=block_3_filters , blocks=num_blocks_3, strides=2, is_training=is_training,
                      name='BLOCK_3',data_format=data_format)


### BLOCK 4 
#layers 6x[(3x3 con,512,)] -> 6 conv layers with:    ***filter_size***=3, ***out_channels***=512, 
#or  times residual unit

block4 = block_group(inputs=block3, filters=block_4_filters , blocks=num_blocks_4, strides=2, is_training=is_training,
                      name='BLOCK_4',data_format=data_format)

## AVE POOLING
pool_size = (1, 1)
output_ave_pooling = tf.layers.average_pooling2d(
    inputs=block4 , pool_size=pool_size, strides=1, padding='VALID',
    data_format=data_format)

layer_flat= tf.reshape(output_ave_pooling, [-1, 512])


fc_layer1 = new_fc_layer(input = layer_flat, num_inputs = 512, num_outputs = num_classes )


output = tf.round(fc_layer1)

# Cost Function to Optimize
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = fc_layer1,
                                                        labels = y_true)

### Multiply logistic loss with weights (ELEMENT-WISE) 
# sum of cost for all labels with weight 1
cost_sum = tf.reduce_sum(tf.multiply(cross_entropy_weights,cross_entropy))

# number of labels with weight 1
num_nonzero_weights = tf.count_nonzero(input_tensor=cross_entropy_weights,dtype = tf.float32)

# average cost
cost = tf.divide(cost_sum, num_nonzero_weights, name= "COST")

### Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
accuracy, accuracy_ops =tf.metrics.accuracy(labels=y_true,predictions=output, weights = cross_entropy_weights)
# Local variables need to show updated accuracy on each iteration 
stream_vars = [i for i in tf.local_variables()]

# Create TensorFlow session
session = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
session.run(init)
saver = tf.train.Saver()
train_batch_size = 50


def fetch_batch(batch_size, available_indexes):
    chosen = np.random.choice(available_indexes,batch_size, replace=False)
    available_indexes = set(available_indexes) - set(chosen)
    X_batch = [drug_fingerprints[i] for i in chosen]
    y_batch = [drug_targets[i] for i in chosen]
    cross_entropy_weights = [drug_weights[i] for i in chosen]
    return X_batch,y_batch,cross_entropy_weights, list(available_indexes)


# counter for total number of epochs
total_epochs = 0

def optimize(num_epochs):
    
    # update the global variable rather than a local copy.
    global total_epochs

    # start-time 
    start_time = time.time()

    for i in range(total_epochs, total_epochs + num_epochs):

        for j in range(int(len(drug_targets)/train_batch_size)):
            if j == 0:
                available_indexes = list(range(len(drug_targets)))                         
            x_batch,y_true_batch, weights_batch, available_indexes = fetch_batch(train_batch_size, available_indexes)

            # put the batch into a dict with the proper names for placeholder variables
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch,
                              cross_entropy_weights: weights_batch}

            # run the optimizer with the btch training data
            session.run(optimizer, feed_dict=feed_dict_train)
            # save the model's weights at the end of each epoch
            saver.save(session, "./temp/my_model_ResNet.ckpt")

            # print update every 10 iterations
            if j % 20 == 0:

                # calculate the accuracy on the training-set.
                acc_ops = session.run(accuracy_ops, feed_dict=feed_dict_train)

                # print update
                print('[Total correct, Total count]:',session.run(stream_vars)) 
                print("Epoch: {}, Optimization Iteration (batch #): {}, Training Accuracy: {} \n".format(i+1,j+1,acc_ops))                        

        # update the total number of iterations
    total_epochs += num_epochs

    # end time
    end_time = time.time()

    # difference between start and end-times.
    time_dif = end_time - start_time

    #time-usage
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


optimize(num_epochs = 1)


writer = tf.summary.FileWriter("./logs/ResNet", session.graph)

save_path= saver.save(session, "./temp/my_model_ResNet_final.ckpt")
