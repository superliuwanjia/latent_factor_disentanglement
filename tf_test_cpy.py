import tensorflow as tf
from tensorflow.python.framework import function
import numpy as np


# activation -> ActClamp(activation) -> GradClamp(ActClamp(activation)) ->
def get_act_clamp(active_z_range, n_z, batch_size):
    """
    Function that returns a tensorflow operator that implements forward clamping
    Gradient is preserved, active z activation are kept, rest of the z are clamped
    """
    active_z_start = active_z_range[0]
    active_z_end = active_z_range[1]
    active_z_width = active_z_start - active_z_end + 1

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
    active_z_width = active_z_start - active_z_end + 1
 
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

act_clamp = get_act_clamp([0, 0], 2, 2)
grad_clamp = get_grad_clamp([0, 0], 2, 2)
        
sess = tf.Session()
x = tf.placeholder(tf.float32, [None, 2])
a = tf.Variable(tf.ones([2, 1], tf.float32), name="a")
x_after_op = grad_clamp(act_clamp(x))
y = tf.matmul(x_after_op, a)

dyda = tf.gradients(y, a)
dydx = tf.gradients(y, x)

# initialize all variables
init = tf.initialize_all_variables()

sess.run(init)

print "value for a before feeding: ", sess.run(a)
print "value for a during feeding: ", sess.run(a, feed_dict={a:[[2], [3]]})
print "value for a after feeding: ", sess.run(a)

a_assigned_op = tf.assign(a, [[2],[3]])
sess.run(a_assigned_op)

print "value for a after assigning: ", sess.run(a)

# Try out regular differentiation
print "value for y:", sess.run(y, feed_dict={x:np.array([[1,2],[2,3]])})
print "dy/da:", sess.run(dyda, feed_dict={x:np.array([[1,2],[2,3]])})
print "dy/dx:", sess.run(dydx, feed_dict={x:np.array([[1,2],[2,3]])})


# doesnt work
# dyada = tf.gradients(y_assigned, a)
# dyadx = tf.gradients(y_assigned, x)

# print "dya/da:", sess.run(dyada, feed_dict={x:np.array([[3],[4]])})
# print "dya/dx:", sess.run(dyadx, feed_dict={x:np.array([[3],[4]])})

   
print "value for y_after_op:", sess.run(x_after_op, feed_dict={x:np.array([[1,2],[2,3]])})

dyopda = tf.gradients(x_after_op, a) 
dyopdx = tf.gradients(x_after_op, x)

# Try out regular differentiation
print "dyop/dx:", sess.run(dyopdx, feed_dict={x:np.array([[1,2],[2,3]])})


