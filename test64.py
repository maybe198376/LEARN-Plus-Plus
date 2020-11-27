import numpy as np
import tensorflow as tf
import scipy.io as scio
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Convert sparse matrix to sparse tensor in TensorFlow
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose().astype(np.float32)
    return tf.SparseTensor(indices, coo.data, coo.shape)

# Hyperparameter
batch_size = 1
img_size = 256
img_sizef = 512
img_sizef1 = 64

# Data Input
in_mat =scio.loadmat("/home/omnisky/imdb.mat")["input"]
in_mat = np.squeeze(in_mat,2)
in_mat = np.transpose(in_mat, [2, 0, 1])
in_mat = np.expand_dims(in_mat,3)

inf_mat =scio.loadmat("/home/omnisky/imdb.mat")["inputf"]
inf_mat = np.squeeze(inf_mat,2)
inf_mat = np.transpose(inf_mat,[2,0,1])
inf_mat = np.expand_dims(inf_mat,3)

label_mat =scio.loadmat("/home/omnisky/imdb.mat")["label"]
label_mat = np.squeeze(label_mat,2)
label_mat = np.transpose(label_mat,[2,0,1])
label_mat = np.expand_dims(label_mat,3)

Mask_M =scio.loadmat("/home/omnisky/mask.mat")["Masks"]
Mask_M = tf.cast(Mask_M, dtype=tf.float32, name=None)

A = scio.loadmat("/home/omnisky/proMatrix_512.mat")["systemMatrix"]
A = convert_sparse_matrix_to_sparse_tensor(A)
A = tf.cast(A, dtype=tf.float32, name=None)
AT = tf.sparse_transpose(A)

X = tf.placeholder(dtype=tf.float32, shape=[batch_size, img_size, img_size, 1])
X_f = tf.placeholder(dtype=tf.float32, shape=[batch_size, img_sizef,img_sizef1, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, img_size, img_size, 1])
F = tf.placeholder(dtype=tf.float32, shape=[batch_size, img_sizef, img_sizef, 1])
index = [0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,264,272,280,288,296,304,312,320,328,
            336,344,352,360,368,376,384,392,400,408,416,424,432,440,448,456,464,472,480,488,496,504]

def CT_Pro(x):
    x = tf.transpose(x, [2, 1, 0])
    x = tf.reshape(x, [img_size*img_size,batch_size])
    x = tf.sparse_tensor_dense_matmul(A,x)
    res = tf.reshape(x, [img_sizef, img_sizef, batch_size])
    return res

def CT_MASK(x,y):
    x = tf.squeeze(x, 3)
    y = tf.squeeze(y, 3)
    x_temp = x[0:1,:,:]
    x_temp = tf.multiply(Mask_M, x_temp)
    y_temp = tf.multiply(1- Mask_M, y)
    res = x_temp
    res = res + y_temp
    return res

def CT_mapping(x, y,mean=0, stddev=0.01):
    lmb = tf.Variable(initial_value=0, trainable=True, dtype=tf.float32)
    x = tf.squeeze(x, 3)
    x1 = CT_Pro(x)
    x1 = tf.squeeze(x1, 2)
    x1 = tf.gather(x1,index)
    x1 = tf.expand_dims(x1, 2)
    x1 = tf.transpose(x1, [2, 1, 0])
    y1 = tf.squeeze(y, 3)    
    x1 = x1 - y1
    x1 = tf.transpose(x1, [1, 2, 0])
    x_temp = x1[:,:,0:1]
    x_temp = tf.image.resize_images(x_temp,[512, 512], method = 0)
    fimg_o = tf.expand_dims(x_temp, 0)
   
    fimg = tf.layers.conv2d(fimg_o, filters=48, kernel_size=5, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
    fimg = tf.nn.relu(fimg)
    fimg = tf.layers.conv2d(fimg, filters=48, kernel_size=5, padding='same',
                                kernel_initializer=tf.contrib.layers.xavier_initializer())   
    fimg = tf.nn.relu(fimg)
    fimg = tf.layers.conv2d(fimg, filters=1, kernel_size=5, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
    in_f = CT_MASK(fimg,fimg_o)
    in_f = tf.transpose(in_f, [2, 1, 0])
    in_f = tf.reshape(in_f, [img_sizef*img_sizef, batch_size])
    res = tf.sparse_tensor_dense_matmul(AT, in_f)
    res = tf.reshape(res, [img_size , img_size, batch_size])
    res = tf.transpose(res, [2, 1, 0])
    res = x - lmb * res
    res = tf.expand_dims(res, 3)
    return res

def iteration_block(in_img, y, mean=0, stddev=0.01):

    outputs_1 = CT_mapping(in_img, y)
    outputs = tf.layers.conv2d(in_img, filters=48, kernel_size=5, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    outputs = tf.nn.relu(outputs)
    outputs = tf.layers.conv2d(outputs, filters=48, kernel_size=5, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    outputs = tf.nn.relu(outputs)
    outputs = tf.layers.conv2d(outputs, filters=1, kernel_size=5, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
    outputs = outputs + outputs_1
    return outputs

def LEARNModel(in_img, in_f, name="LEARN", iters=50):
    cur_img = in_img
    for _ in range(iters):
        cur_img = iteration_block(cur_img, in_f)
    in_f1 = tf.squeeze(cur_img, 3)
    in_f1 = CT_Pro(in_f1)
    in_f1 = tf.transpose(in_f1, [2, 1, 0])
    in_f1 = tf.expand_dims(in_f1,3)
    return cur_img, in_f1

with tf.variable_scope('LEARN_Model') as scope:
    Y_, F_ = LEARNModel(X, X_f,iters=50)
learn_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='LEARN_Model')

mse_cost = tf.reduce_sum(tf.squared_difference(Y_, Y)) / (batch_size)
saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver.restore(sess,'/home/omnisky/n8.ckpt')
print("Start training ... ")
i = 0
while i < 10:
    batch_in = in_mat[i:(i+batch_size)]
    batch_f = inf_mat[i:(i+batch_size)]
    batch_lb = label_mat[i:(i+batch_size)]
    _mse_cost,outdata= sess.run([mse_cost, Y_], feed_dict={X:batch_in, X_f:batch_f, Y:batch_lb})
    print('Epoch: %d  - mse: %.6f' % (i+1, _mse_cost))
    scio.savemat('/home/omnisky/outimg' + repr(i) + '.mat', {'outimg': outdata, 'label': batch_lb})
    i = i + 1
sess.close()
