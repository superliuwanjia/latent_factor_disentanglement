import tensorflow as tf
from tensorflow.python.framework import function
import numpy as np

def get_clamp(active_z_range, batch_size,n_z, invariance_factor=100):
    """
    concatonation of both act_clamp and grad_clamp operator
    """
    active_z_start = active_z_range[0]
    active_z_end = active_z_range[1]
    active_z_width = active_z_end - active_z_start + 1

    print "start",active_z_start
    print "end", active_z_end
    print "width", active_z_width
    @function.Defun(tf.float32, tf.float32)
    def MyGrad_Clamp(z_act, grad):
        z_act_mean = tf.tile(tf.reduce_mean(z_act, axis=0,keep_dims=True), [batch_size,1])
 
        first_mean_slice = tf.slice(z_act_mean, [0, 0], [batch_size, active_z_start])
        second_mean_slice = tf.slice(z_act_mean, [0, active_z_end + 1], \
                [batch_size, n_z - active_z_end - 1])

        first_act_slice = tf.slice(z_act, [0, 0], [batch_size, active_z_start])
        second_act_slice = tf.slice(z_act, [0, active_z_end + 1], \
                [batch_size, n_z - active_z_end - 1])
       
        grad_to_keep = tf.slice(grad, [0, active_z_start], [batch_size, active_z_width])

        # only keep gradient (grad) of a specific z to pass, rest are act - mean(act)
        return tf.concat(1,[(first_act_slice - first_mean_slice) / invariance_factor, \
                            grad_to_keep, \
                            (second_act_slice - second_mean_slice) / invariance_factor])


    @function.Defun(tf.float32, grad_func=MyGrad_Clamp)
    def MyOp_Clamp(z_act):
        z_act_mean = tf.tile(tf.reduce_mean(z_act, axis=0,keep_dims=True), [batch_size,1])
        
        first_mean_slice = tf.slice(z_act_mean, [0, 0], [batch_size, active_z_start])
        second_mean_slice = tf.slice(z_act_mean, [0, active_z_end + 1], \
                [batch_size, n_z - active_z_end - 1])
        
        act_to_keep = tf.slice(z_act, [0, active_z_start], [batch_size, active_z_width])
        z_act_clamped = tf.concat(1, [first_mean_slice, act_to_keep, second_mean_slice])
        return z_act_clamped
    return MyOp_Clamp 

# activation -> ActClamp(activation) -> GradClamp(ActClamp(activation)) ->
def get_act_clamp(active_z_range, n_z, batch_size):
    """
    Function that returns a tensorflow operator that implements forward clamping
    Gradient is preserved, active z activation are kept, rest of the z are clamped
    """
    active_z_start = active_z_range[0]
    active_z_end = active_z_range[1]
    active_z_width = active_z_end - active_z_start + 1

    @function.Defun(tf.float32, tf.float32)
    def MyGrad_ActClamp(act, grad):
        # this step shouldn't affect gradient
        return grad 

    @function.Defun(tf.float32, grad_func=MyGrad_ActClamp)
    def MyOp_ActClamp(z_act):
        z_act_mean = tf.tile(tf.reduce_mean(z_act, axis=0,keep_dims=True), [batch_size,1])
        
        first_mean_slice = tf.slice(z_act_mean, [0, 0], [batch_size, active_z_start])
        second_mean_slice = tf.slice(z_act_mean, [0, active_z_end + 1], \
                [batch_size, n_z - active_z_end - 1])
        
        act_to_keep = tf.slice(z_act, [0, active_z_start], [batch_size, active_z_width])
        z_act_clamped = tf.concat(1, [first_mean_slice, act_to_keep, second_mean_slice])
        # put clamped act and original together, will be used in gradient clamp
        return tf.concat(0, [z_act_clamped, z_act])
    return MyOp_ActClamp

def get_grad_clamp(active_z_range, n_z, batch_size, invariance_factor=100):
    """
    Function that returns a tensorflow operator that implements backward clamping
    Activation is preserved, active z gradient are kept, rest of z are clamped
    """
    active_z_start = active_z_range[0]
    active_z_end = active_z_range[1]
    active_z_width = active_z_end - active_z_start + 1
 
    @function.Defun(tf.float32, tf.float32)
    def MyGrad_GradClamp(clamped_original_combined, grad):
        # original z activation are saved in lower half of the matrix
        z_act = tf.slice(clamped_original_combined, [batch_size, 0], [batch_size, n_z])
        z_act_mean = tf.tile(tf.reduce_mean(z_act, axis=0,keep_dims=True), [batch_size,1])
 
        first_mean_slice = tf.slice(z_act_mean, [0, 0], [batch_size, active_z_start])
        second_mean_slice = tf.slice(z_act_mean, [0, active_z_end + 1], \
                [batch_size, n_z - active_z_end - 1])

        first_act_slice = tf.slice(z_act, [0, 0], [batch_size, active_z_start])
        second_act_slice = tf.slice(z_act, [0, active_z_end + 1], \
                [batch_size, n_z - active_z_end - 1])
       
        grad_to_keep = tf.slice(grad, [0, active_z_start], [batch_size, active_z_width])

        # only keep gradient (grad) of a specific z to pass, rest are act - mean(act)
        return tf.concat(1,[(first_act_slice - first_mean_slice) / invariance_factor, \
                            grad_to_keep, \
                            (second_act_slice - second_mean_slice) / invariance_factor])

    @function.Defun(tf.float32, grad_func=MyGrad_GradClamp)
    def MyOp_GradClamp(clamped_original_combined):
        # we only want clamped output 
        return tf.slice(clamped_original_combined, [0, 0], [batch_size, n_z])
    return MyOp_GradClamp


