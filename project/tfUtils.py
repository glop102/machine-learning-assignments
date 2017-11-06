import tensorflow as tf

def denseUnit(node,shape):
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32))
	b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
	x = tf.nn.relu(tf.matmul(node,w)+b)
	appendWeightNorm(w)
	return x
def denseUnit_noActivation(node,shape):
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32))
	b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
	x = tf.matmul(node,w)+b
	appendWeightNorm(w)
	return x

global weight_squared
weight_squared = 0.0
def appendWeightNorm(w):
	global weight_squared
	# weight_norm = weight_norm+tf.nn.l2_loss(w)
	# weight_norm = weight_norm+tf.reduce_sum(tf.nn.l2_normalize(w,0))
	weight_squared = weight_squared + tf.sqrt(tf.reduce_sum(w**2))

def convolutionUnit(node,shape): #kernel size x&y, features in&out
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32))
	b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
	conv = tf.nn.convolution(node,w,"SAME") + b
	#appendWeightNorm(w)
	return conv
def convolutionUnit_2x2step(node,shape): #kernel size x&y, features in&out
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32))
	b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
	conv = tf.nn.convolution(node,w,"SAME",strides=[2,2]) + b
	appendWeightNorm(w)
	return conv