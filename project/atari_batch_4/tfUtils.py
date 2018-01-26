import tensorflow as tf

def denseUnit(node,shape):
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32))
	b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
	mean = tf.Variable(tf.constant(1.0, shape=[1]))
	variance = tf.Variable(tf.constant(1.0, shape=[1]))
	offset = tf.Variable(tf.constant(1.0, shape=[1]))
	scale = tf.Variable(tf.constant(1.0, shape=[1]))

	mults = tf.matmul(node,w)
	norm = tf.nn.batch_normalization(mults,mean,variance,offset,scale,variance_epsilon=0.00001)
	x = tf.nn.relu(norm+b)
	appendWeightNorm(w)
	x.weights = w
	x.biases = b
	x.mean = mean
	x.variance = variance
	x.offset = offset
	x.scale = scale
	x.variables = [w,b,mean,variance,offset,scale]
	return x
def denseUnit_noActivation(node,shape):
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32))
	b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
	mean = tf.Variable(tf.constant(1.0, shape=[1]))
	variance = tf.Variable(tf.constant(1.0, shape=[1]))
	offset = tf.Variable(tf.constant(1.0, shape=[1]))
	scale = tf.Variable(tf.constant(1.0, shape=[1]))

	mults = tf.matmul(node,w)
	norm = tf.nn.batch_normalization(mults,mean,variance,offset,scale,variance_epsilon=0.00001)
	x = norm+b
	appendWeightNorm(w)
	x.weights = w
	x.biases = b
	x.mean = mean
	x.variance = variance
	x.offset = offset
	x.scale = scale
	x.variables = [w,b,mean,variance,offset,scale]
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
	mean = tf.Variable(tf.constant(1.0, shape=[1]))
	variance = tf.Variable(tf.constant(1.0, shape=[1]))
	offset = tf.Variable(tf.constant(1.0, shape=[1]))
	scale = tf.Variable(tf.constant(1.0, shape=[1]))

	conv = tf.nn.convolution(node,w,"SAME")
	norm = tf.nn.batch_normalization(conv,mean,variance,offset,scale,variance_epsilon=0.00001)
	x = norm+b
	#appendWeightNorm(w)
	x.weights = w
	x.biases = b
	x.mean = mean
	x.variance = variance
	x.offset = offset
	x.scale = scale
	x.variables = [w,b,mean,variance,offset,scale]
	return x
def convolutionUnit_2x2step(node,shape): #kernel size x&y, features in&out
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32))
	b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
	mean = tf.Variable(tf.constant(1.0, shape=[1]))
	variance = tf.Variable(tf.constant(1.0, shape=[1]))
	offset = tf.Variable(tf.constant(1.0, shape=[1]))
	scale = tf.Variable(tf.constant(1.0, shape=[1]))

	conv = tf.nn.convolution(node,w,"SAME",strides=[2,2])
	norm = tf.nn.batch_normalization(conv,mean,variance,offset,scale,variance_epsilon=0.00001)
	x = norm+b
	#appendWeightNorm(w)
	x.weights = w
	x.biases = b
	x.mean = mean
	x.variance = variance
	x.offset = offset
	x.scale = scale
	x.variables = [w,b,mean,variance,offset,scale]
	return x

def batchNorm(node):
	mean = tf.Variable(tf.constant(1.0, shape=[1]))
	variance = tf.Variable(tf.constant(1.0, shape=[1]))
	offset = tf.Variable(tf.constant(1.0, shape=[1]))
	scale = tf.Variable(tf.constant(1.0, shape=[1]))
	layer = tf.nn.batch_normalization(node,mean,variance,offset,scale,variance_epsilon=0.00001)
	layer.mean = mean
	layer.variance = variance
	layer.offset = offset
	layer.scale = scale
	layer.variables = [mean,variance,offset,scale]
	return layer