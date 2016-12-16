import tensorflow as tf
from tensorflow.python.framework import function
import numpy as np


# activation -> ActClamp(activation) -> GradClamp(ActClamp(activation)) ->

@function.Defun(tf.float32, tf.float32)
def MyGrad_ActClamp(act, grad):
    # this step shouldn't affect gradient
    return grad 

@function.Defun(tf.float32, grad_func=MyGrad_ActClamp)
def MyOp_ActClamp(x):
    #return x
    mean = tf.tile(tf.reduce_mean(x, axis=0,keep_dims=True), [2,1])
    act = tf.concat(1,[tf.slice(mean,[0,0],[2,1]), tf.slice(x,[0,1],[2,1])])
    # put clamped act and original together, will be used in gradient clamp
    return tf.concat(0, [act, x])

@function.Defun(tf.float32, tf.float32)
def MyGrad_GradClamp(clamped_original_combined, grad):
    # factor is used to scale down grad
    factor = 10
    original = tf.slice(clamped_original_combined, [2,0], [2,2])
    mean = tf.tile(tf.reduce_mean(original, axis=0,keep_dims=True), [2,1])
    # only keep gradient (grad) of a specific z to pass, rest are act - mean(act)
    return tf.concat(1,[(tf.slice(original,[0,0],[2,1]) - tf.slice(mean,[0,0],[2,1]))/factor, tf.slice(grad,[0,1],[2,1])])

@function.Defun(tf.float32, grad_func=MyGrad_GradClamp)
def MyOp_GradClamp(clamped_original_combined):
    # we only want clamped output 
    return tf.slice(clamped_original_combined, [0,0], [2,2])

sess = tf.Session()
x = tf.placeholder(tf.float32, [None, 2])
a = tf.Variable(tf.ones([2, 1], tf.float32), name="a")
x_after_op = MyOp_GradClamp(MyOp_ActClamp(x))
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


