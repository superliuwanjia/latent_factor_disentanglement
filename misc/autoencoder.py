# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    #if len(shape) == 4:
    #W = tf.truncated_normal(shape,stddev=0.0001)
    #else:
    W = tf.random_normal(shape,stddev=1)
    #W = tf.zeros(shape)
    return tf.Variable(W)

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    b = tf.random_normal(shape=shape)
    return tf.Variable(b)

def conv2d(x, W, b):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], "SAME")
    return tf.nn.bias_add(h_conv, b)

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
    return h_max
def relu(x):
    '''
    Apply ReLU on input
    '''
    relu_out = tf.nn.relu(x)
    return relu_out

def reshape(x, shape):
    return tf.reshape(x, shape=shape)

def resize(x, size):
    return tf.image.resize_images(x, size)
    
def mult(x, W):
    return tf.matmul(x, W)
    
def softmax(x):
    return tf.nn.softmax(x)


# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 50
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 96 # 1st layer num features
n_hidden_2 = n_hidden_1 / 2 # 2nd layer num features
n_hidden_3 = n_hidden_2 / 2 # 2nd layer num features
z_size = 240

im_size = 150 # width and height of imput image
e_filter_size = 5
d_filter_size = e_filter_size

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, im_size, im_size, 1], name="input")

weights = {
    'encoder_h1': weight_variable([5, 5, 1 ,n_hidden_1]), # 146 / 2= 73
    'encoder_h2': weight_variable([4, 4, n_hidden_1 ,n_hidden_2]), # 70 / 2 = 35
    'encoder_h3': weight_variable([4, 4, n_hidden_2 ,n_hidden_3]), # 32 / 2 = 16, total 24 * 16 * 16
    'encoder_h4': weight_variable([n_hidden_3 * 16 * 16, n_hidden_3 * 16 * 16]), # 24 * 16 * 16
    'encoder_h5': weight_variable([n_hidden_3 * 16 * 16, z_size]),
    
    'decoder_h5': weight_variable([7, 7, n_hidden_1, 1]), # 146 / 2= 73
    'decoder_h4': weight_variable([5, 5, n_hidden_2 ,n_hidden_1]), # 70 / 2 = 35
    'decoder_h3': weight_variable([5, 5, n_hidden_3 ,n_hidden_2]), # 32 / 2 = 16, total 24 * 16 * 16
    'decoder_h2': weight_variable([n_hidden_3 * 16 * 16, n_hidden_3 * 16 * 16]), # 24 * 16 * 16
    'decoder_h1': weight_variable([z_size, n_hidden_3 * 16 * 16]),
}

biases = {
    'encoder_b1': bias_variable([n_hidden_1]),
    'encoder_b2': bias_variable([n_hidden_2]),
    'encoder_b3': bias_variable([n_hidden_3]),
    'encoder_b4': bias_variable([1]),
    'encoder_b5': bias_variable([1]),

    'decoder_b5': bias_variable([n_hidden_1]),
    'decoder_b4': bias_variable([n_hidden_2]),
    'decoder_b3': bias_variable([n_hidden_3]),
    'decoder_b2': bias_variable([1]),
    'decoder_b1': bias_variable([1]),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = max_pool_2x2(relu(conv2d(x, weights['encoder_h1'], biases['encoder_b1'])))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = max_pool_2x2(relu(conv2d(layer_1, weights['encoder_h2'], biases['encoder_b2'])))
    # Encoder Hidden layer with sigmoid activation #3
    layer_3 = reshape( \
            max_pool_2x2(relu(conv2d(layer_2, weights['encoder_h3'], biases['encoder_b3']))), \
            [-1, n_hidden_3 * 16 * 16])
    # FC
    layer_4 = conv2d(layer_3, weights['encoder_h4'], biases['encoder_h4'])
    # FC to Z , TODO: Do we need Relu here?
    z = conv2d(layer_4, weights['encoder_h5'], biases['encoder_h5'])
    return z


# Building the decoder
def decoder(z):
    layer_1 = conv2d(z, weights['decoder_h1'], biases['decoder_h1'])
    layer_2 = reshape(conv2d(layers_1, weights['decoder_h2'], biases['decoder_h2']), \
            [-1, 16, 16, n_hidden_3])
    layer_3 = relu(conv2d(resize(layer_2, [32, 32]), weights['decoder_h3'], biases['decoder_h3']))
    layer_4 = relu(conv2d(resize(layer_3, [56, 56]), weights['decoder_h4'], biases['decoder_h4']))
    x_p = relu(conv2d(resize(layer_4, [im_size, im_size]), weights['decoder_h5'], biases['decoder_h5']))
    return x_p

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
rc_coeff = tf.constant(1)
kl_coeff = tf.constant(1)

l_rc = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
l_kl = tf.contrib.distributions.kl()

cost = tf.add(tf.scalar_mul(rc_coeff, l_rc), tf.scalar_mul(kl_coeff, l_kl))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
