import random
import numpy as np
import tensorflow as tf
import PIL.Image
import matplotlib.pyplot as plt

from tensorflow.python.framework import function

from clamping_ops import *
# import matplotlib.pyplot as plt
import os

np.random.seed(0)
tf.set_random_seed(0)
np.set_printoptions(threshold='nan')

data_folder = "/home/ubuntu/data/statue1_rot_100_light_100/"
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

def flatten(x, shape):
    return tf.reshape(x, shape=shape)

    
def mult(x, W):
    return tf.matmul(x, W)
    
def softmax(x):
    return tf.nn.softmax(x)


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
    W = tf.random_normal(shape,stddev=0.01)
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


def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
#    return tf.ones((fan_in, fan_out), dtype=tf.float32)
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, sess, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=100, controlled_z=3, rc_loss=1, kl_loss=1):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.controlled_z = controlled_z
        self.rc_loss=rc_loss
        self.kl_loss=kl_loss
         
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [batch_size, network_architecture["n_input"]])
        
        self.n_z = network_architecture["n_z"]
        
        self.clamp_ops = []

        # create forward and backward clampping operator for all controlled zs
        for i in range(controlled_z):
            # print "Creating clamp ops, ", i 
            self.clamp_ops.append(get_clamp([i, i], self.batch_size, self.n_z))

        if self.controlled_z < self.n_z:
            # print "Creating last clamp op"
            self.clamp_ops.append(get_clamp([self.controlled_z, \
                    self.n_z - 1], self.batch_size, self.n_z))

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss()

        self.summaries = tf.summary.merge_all()
                
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = sess
        self.sess.run(init)
    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        n_z = self.network_architecture["n_z"]
        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"],
                                      self.x)

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        
        # Use generator to determine mean of reconstruction
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"],
                                    self.z)

        # forward and backward clamped z vector for different z index
        self.z_clamped = []
        for i in range(len(self.clamp_ops)):
            print "Creating clamped zs, ", i 
            self.z_clamped.append(self.clamp_ops[i](self.z))

        # reconstruction using forward and backward clamped z vector
        self.x_reconstr_mean_clamped = []
        for z in self.z_clamped:
            self.x_reconstr_mean_clamped.append(self._generator_network( \
                    network_weights["weights_gener"],
                    network_weights["biases_gener"],
                    z))
 
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, n_hidden_recog_3, 
                            n_hidden_gener_1,  n_hidden_gener_2, n_hidden_gener_3,
                            n_input, n_z):
        self.n_z = n_z
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': weight_variable([5, 5, 1, 96]), # 146/2 = 73
            'h2': weight_variable([6, 6, 32, 64]), # 68/2 = 34
            'h3': weight_variable([5, 5, 64, 32]), # 30/2 = 15
            'out_mean': weight_variable([15 * 15 * 32, n_z]),
            'out_log_sigma': weight_variable([32, n_z])}
        all_weights['biases_recog'] = {
            'b1': bias_variable([96]),
            'b2': bias_variable([64]),
            'b3': bias_variable([32]),
            'out_mean': bias_variable([n_z]),
            'out_log_sigma': bias_variable([n_z])}
        all_weights['weights_gener'] = {
            'h1': weight_variable([n_z, 7200]),
            'h2': weight_variable([7, 7, 1, 32]),
            'h3': weight_variable([7, 7, 32, 64]),
            'h4': weight_variable([7, 7, 64, 96]),
            'out_mean': weight_variable([7, 7, 96, 1]),
        all_weights['biases_gener'] = {
            'b1': bias_variables([7200]),
            'b2': bias_variables([32]),
            'b3': bias_variables([64]),
            'b4': bias_variables([96]),
            'out_mean': bias_variables([1]),
        return all_weights
            
    def _recognition_network(self, weights, biases, x):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = max_pool_2x2(relu(conv2d(x, weights['h1'], biases['h1'])))
        layer_2 = max_pool_2x2(relu(conv2d(layer_1, weights['h2'], biases['h2'])))
        layer_3 = max_pool_2x2(relu(conv2d(layer_2, weights['h3'], biases['h3'])))
        layer_4 = tf.nn.bias_add(mult(flatten(layer_3, [self.batch_size, 15*15*128]),weights['h4']), biases['h4']) 
        layer_5 = tf.nn.bias_add(mult(layer_4,weights['h5']), biases['h5']) 
        z_mean = tf.nn.bias_add(mult(layer_5, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.nn.bias_add(mult(layer_5, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases, z):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = tf.nn.bias_add(mult(z, weights['h1']), biases['h1'])
        layer_2 = relu(conv2d(tf.image.resize_images(flatten(layer_1, [self.batch_size, 15, 15,1]), [30, 30], method=1), weights['h2'], biases['h2']))
        layer_3 = relu(conv2d(tf.image.resize_images(layer_2, [48, 48], method=1), weights['h3'], biases['h3']))
        layer_4 = relu(conv2d(tf.image.resize_images(layer_3, [84, 84], method=1), weights['h4'], biases['h4']))
        x_reconstr_mean = relu(conv2d(tf.image.resize_images(layer_3, [156, 156], method=1), weights['out_mean'], biases['out_mean']))
        
        return x_reconstr_mean
            
    def _create_loss(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        self.reconstr_loss = \
            - tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) + \
                     (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1)
        #tf.summary.scalar('reconstr_loss', self.reconstr_loss)    
        self.reconstr_loss_clamped = []
        
        for i in range(len(self.x_reconstr_mean_clamped)):
            reconstr = self.x_reconstr_mean_clamped[i]
            self.reconstr_loss_clamped.append( \
                - tf.reduce_sum(self.x * tf.log(1e-10 + reconstr) + \
                     (1-self.x) * tf.log(1e-10 + 1 - reconstr), 1))
 
            #tf.summary.scalar('reconstr_loss_' + str(i), self.reconstr_loss_clamped[i])    
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        self.latent_loss = - 0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        #tf.summary.scalar('latent_loss', self.latent_loss)
        self.cost = tf.reduce_mean(self.rc_loss * self.reconstr_loss + \
                                   self.kl_loss * self.latent_loss)   # average over batch
        tf.summary.scalar('total_loss', self.cost)
        self.optimizer = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
 

        # clamped reconstruction optimizer
        
        self.cost_clamped = []
        for i in range(len(self.reconstr_loss_clamped)):
            loss = self.reconstr_loss_clamped[i]
            self.cost_clamped.append( \
                tf.reduce_mean(self.rc_loss * loss + \
                               self.kl_loss * self.latent_loss))
            #tf.summary.scalar('total_loss_'+str(i), self.cost_clamped[i])

        self.optimizer_clamped = []
        for cost in self.cost_clamped:
            self.optimizer_clamped.append( \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost))
        
    def partial_fit(self, X, z_index=-1):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        if z_index == -1:
            opt, cost, kl, rc, smr = self.sess.run((self.optimizer, self.cost, \
                                       self.latent_loss, self.reconstr_loss, self.summaries), 
                                      feed_dict={self.x: X})
        else:
            opt, cost, kl, rc, smr = self.sess.run((self.optimizer_clamped[z_index], \
                                       self.cost_clamped[z_index], \
                                       self.latent_loss, self.reconstr_loss_clamped[z_index], \
                                       self.summaries),\
                                      feed_dict={self.x: X})

        return cost, smr, np.mean(kl), np.mean(rc)
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=[self.batch_size, self.network_architecture["n_z"]])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})
def train(sess, network_architecture, inputs, input_configs, cur_path, clamped_train=False,\
          learning_rate=0.00001,
          batch_size=50, training_epochs=10, display_step=100,
          controlled_z=2, rc_loss=1, kl_loss=1, loaded=False, continuous=False):

    train_writer = tf.summary.FileWriter(os.path.join(cur_path, 'tb/train/'))
    vae = VariationalAutoencoder(sess, network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size,
                                 controlled_z=controlled_z,
                                 rc_loss=rc_loss,
                                 kl_loss=kl_loss)
    total_iters = training_epochs * inputs.shape[0] / batch_size
    # Training cycle
    for iters in range(total_iters):
        if not clamped_train: 
            if not continuous:
                batch_xs = inputs[np.random.choice(range(inputs.shape[0]), batch_size, replace=False),:]
            else:
                type_to_train = random.choice([[t] * input_configs[t]["ratio"] \
                for t in input_configs.keys()])      
                type_to_train = type_to_train[0]
                same_config_inputs = random.choice(input_configs[type_to_train]["index"])
                inputs_index_to_use = np.random.choice(same_config_inputs, batch_size, replace=False)
                batch_xs = inputs[inputs_index_to_use, :]
            # Fit training using batch data
            cost, smr, kl, rc = vae.partial_fit(batch_xs, -1)
        else:
            type_to_train = random.choice([[t] * input_configs[t]["ratio"] \
            for t in input_configs.keys()])       
            type_to_train = type_to_train[0]
            same_config_inputs = random.choice(input_configs[type_to_train]["index"])
            inputs_index_to_use = np.random.choice(same_config_inputs, batch_size, replace=False)
            batch_xs = inputs[inputs_index_to_use, :]
        # Fit training using batch data
            cost, smr, kl, rc = vae.partial_fit(batch_xs, input_configs[type_to_train]["z"])
  
        train_writer.add_summary(smr, iters)
        # Display logs per epoch step
        if iters % display_step == 0:
            print "iters:", '%04d' % (iters+1), \
                  "total_loss=", "{:.9f}".format(cost),\
                  "kl_loss=", "{:.9f}".format(kl),\
                  "rc_loss=", "{:.9f}".format(rc)
    return vae

def visualize(matrix, output_name, scale=255, offset=0, img_shape=(1, 150, 150)):
    matrix = matrix.reshape(img_shape)  * scale + offset
    matrix = matrix.clip(0,255)
    
    # h*w*c
    image = np.transpose(matrix, (1,2,0))

    image = image.astype('uint8')
    if img_shape[0] == 3:
        mode = "RGB"
    else:
        mode = "L"
        image = image[:,:,0]
    img = PIL.Image.fromarray(image, mode=mode)
    img.save(output_name)

    
def visualize_reconstruction(vae, folder, inputs):
    if not os.path.exists(folder):
        os.mkdir(folder)

    samples = np.concatenate(\
        [vae.reconstruct(inputs[i*vae.batch_size:(i+1) * vae.batch_size,]) for i in \
            range(0, inputs.shape[0]/vae.batch_size)])

    for i in range(samples.shape[0]):
        visualize(samples[i,:], os.path.join(folder, str(i)+".png"))

def visualize_latent_space(vae, folder, inputs, fn):
    color = [int((i%vae.batch_size)/10) for i in range(vae.batch_size)]
    markers = ["o", "x", "+"]
    if not os.path.exists(folder):
        os.mkdir(folder)

    plt.figure(figsize=(8, 6))
    plt.grid()
    for index in range(len(inputs)):  
        inp = inputs[index]
        z_mu = np.concatenate(\
            [vae.transform(inp[i*vae.batch_size:(i+1) * vae.batch_size,:]) for i in \
            range(0, (inp.shape[0])/vae.batch_size)])
        plt.scatter(z_mu[:, 0], z_mu[:, 1], c=color, marker=markers[index]) 
    plt.colorbar()
    plt.savefig(os.path.join(folder, fn+'.png'))

network_architecture = \
    dict(n_hidden_recog_1=1000, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_recog_3=100, # 2nd layer encoder neurons
         n_hidden_gener_1=100, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_hidden_gener_3=1000, # 2nd layer decoder neurons
         n_input=22500, # NIST data input (img shape: 28*28)
         n_z=2)  # dimensionality of latent space

inputs = np.reshape(np.load("/home/ubuntu/data/statue1_rot_100_light_100/X_train.npy"), [10000, 150, 150, 1])
cur_path = "/home/ubuntu/Documents/dc_ign/"
# index contains a list of index lists where each index lists is a 
# specfic configuration of a given variation (e.g "rotation"),
# index in a specific index list are index of inputs having that specific
# variation (e.g "rotation" = 36)
input_configs = {
    "rotation": {
        "index": [range(x, 10000, 100) for x in range(100)],
        "ratio": 1,
        "z": 0 
    },
    "light": {
        "index": [range(x*100, (x+1)*100, 1) for x in range(100)],
        "ratio": 1, 
        "z": 1     
    }
}

# saver = tf.train.Saver()
blist=[100]
tag="_clamp"
tag="_continuous"
tag=""
epoches=100
clamp=False
for b in blist:
    sess = tf.InteractiveSession()

    vae = train(sess, network_architecture, inputs, input_configs, cur_path, batch_size=100, training_epochs=epoches, kl_loss=b, clamped_train=clamp, continuous=False)
    visualize_reconstruction(vae, os.path.join(cur_path, "reconstruction_vae_light_b_"+str(b)+tag), inputs[5000:5100,:])
    visualize_reconstruction(vae, os.path.join(cur_path, "reconstruction_vae_rot_b_"+str(b)+tag), inputs[range(50,10000,100),:])
# saver.save(sess, os.path.join(cur_path, "save.ckpt"))
    visualize_latent_space(vae, os.path.join(cur_path, "latent_viz_b_"+str(b)+tag), \
        [inputs[5000:5100,:],inputs[0:100,:],inputs[9900:10000,:]],  "light")
    visualize_latent_space(vae, os.path.join(cur_path, "latent_viz_b_"+str(b)+tag), \
        [inputs[range(0,10000,100),:],inputs[range(50,10000,100),:],inputs[range(99,10000,100),:]], "rotation")
