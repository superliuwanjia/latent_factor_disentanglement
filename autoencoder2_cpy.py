import random
import numpy as np
import tensorflow as tf
import PIL.Image
from tensorflow.python.framework import function

from clamping_ops import *
# import matplotlib.pyplot as plt
import os

np.random.seed(0)
tf.set_random_seed(0)
np.set_printoptions(threshold='nan')

data_folder = "/home/ubuntu/data/statue1_rot_100_light_100/"

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
        
        self.act_clamp_ops = []
        self.grad_clamp_ops = []

        # create forward and backward clampping operator for all controlled zs
        for i in range(controlled_z):
            # print "Creating clamp ops, ", i 
            self.act_clamp_ops.append(get_act_clamp([i, i], self.batch_size, self.n_z))
            self.grad_clamp_ops.append(get_grad_clamp([i, i], self.batch_size, self.n_z))

        if self.controlled_z < self.n_z:
            # print "Creating last clamp op"
            self.act_clamp_ops.append(get_act_clamp([self.controlled_z, \
                    self.n_z - 1], self.batch_size, self.n_z))
            self.grad_clamp_ops.append(get_grad_clamp([self.controlled_z, \
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

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
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
        for i in range(len(self.act_clamp_ops)):
            print "Creating clamped zs, ", i 
            self.z_clamped.append(self.grad_clamp_ops[i](self.act_clamp_ops[i](self.z)))

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
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'h3': tf.Variable(xavier_init(n_hidden_recog_2, n_hidden_recog_3)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_3, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_3, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'b3': tf.Variable(tf.zeros([n_hidden_recog_3], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'h3': tf.Variable(xavier_init(n_hidden_gener_2, n_hidden_gener_3)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_3, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_3, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'b3': tf.Variable(tf.zeros([n_hidden_gener_3], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        layer_3 = self.transfer_fct(tf.add(tf.matmul(layer_2, weights['h3']), 
                                           biases['b3'])) 
        z_mean = tf.add(tf.matmul(layer_3, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_3, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases, z):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        layer_3 = self.transfer_fct(tf.add(tf.matmul(layer_2, weights['h3']), 
                                           biases['b3'])) 
 
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['out_mean']), 
                                 biases['out_mean']))
        
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
          batch_size=50, training_epochs=10, display_step=1,
          controlled_z=2, rc_loss=1, kl_loss=1, loaded=False):

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
            batch_xs = inputs[np.random.choice(range(inputs.shape[0]), batch_size, replace=False),:]

            # Fit training using batch data
            cost, smr, kl, rc = vae.partial_fit(batch_xs, -1)
        else:
            type_to_train = random.choice([[t] * input_configs[t]["ratio"] \
            for t in input_configs.keys()])       
            same_config_inputs = random.choice(input_configs[type_to_train]["index"])
            inputs_index_to_use = np.choice(same_config_inputs, batch_size, replace=False)
            batch_xs = all_inputs[inputs_index_to_use, :]
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

network_architecture = \
    dict(n_hidden_recog_1=1000, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_recog_3=100, # 2nd layer encoder neurons
         n_hidden_gener_1=100, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_hidden_gener_3=1000, # 2nd layer decoder neurons
         n_input=22500, # NIST data input (img shape: 28*28)
         n_z=2)  # dimensionality of latent space

inputs = np.load("/home/ubuntu/data/statue1_rot_100_light_100/X_train.npy")
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
        "index": [range(x*100, (x+1)*100) for x in range(100)],
        "ratio": 1, 
        "z": 1     
    }
}

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



# saver = tf.train.Saver()
sess = tf.InteractiveSession()

vae = train(sess, network_architecture, inputs, input_configs, cur_path, batch_size=100, training_epochs=40, kl_loss=1)
# saver.save(sess, os.path.join(cur_path, "save.ckpt"))

#samples = vae.generate()
samples = vae.reconstruct(inputs[0:100,:])
samples2 = vae.reconstruct(inputs[range(50,10000,100),:])
#print sess.run(vae.z, feed_dict={vae.x:inputs[0:50]})
for i in range(100):
    #visualize(samples[i,:], os.path.join(cur_path, "reconstruction_vae_light_2/"+str(i)+".png"))
    #visualize(samples2[i,:], os.path.join(cur_path, "reconstruction_vae_rot_2/"+str(i)+".png"))
#saver.restore(sess, os.path.join(cur_path, "save.ckpt"))
