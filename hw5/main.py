import tensorflow as tf
from math import log
from numpy import *
from numpy.random import choice,shuffle,permutation

def makeOneHot(a):
	out = []
	for l in range(a.shape[0]):
		aa = array([0]*10,dtype=float32)
		aa[int(a[l])]=1
		out.append(aa)
	return array(out,dtype=float32)

def getData():
	import pickle
	names = ["cifar-10-batches-py/data_batch_1",
		"cifar-10-batches-py/data_batch_2",
		"cifar-10-batches-py/data_batch_3",
		"cifar-10-batches-py/data_batch_4",
		"cifar-10-batches-py/data_batch_5"
		];
	labels = array([],dtype=float32)
	data = empty([0,32*32*3],dtype=float32)
	for name in names:
		print(name)
		f = open(name,"rb")
		dic = pickle.load(f, encoding='bytes')
		labels = concatenate((labels,dic[b"labels"]))
		data = concatenate((data,dic[b"data"]))
	#print(data.shape)
	#print(data[b"labels"])
	#print(data[b"data"])
	return labels,data

labels,data = getData()
shuffled = permutation(labels.shape[0])
labels = labels[shuffled]
data = data[shuffled]
data = reshape(data,[-1,3,32,32]) #axies are image,color,y,x
data = transpose(data,[0,2,3,1]) #move axies to image,y,x,color
data /= 256.0
del shuffled
labels = makeOneHot(labels)

def getTest():
	import pickle
	name = "cifar-10-batches-py/test_batch"
	f = open(name,"rb")
	dic = pickle.load(f,encoding="bytes")
	return dic[b"labels"],dic[b"data"]

test_labels,test_data=getTest()
test_labels = makeOneHot(array(test_labels))
test_data = reshape(test_data,[-1,3,32,32])
test_data = transpose(test_data,[0,2,3,1]) #move axies to image,y,x,color
test_data = test_data.astype(float32)
test_data /= 256.0

# nn=100
# def testImage():
# 	from PIL import Image
# 	im = Image.new("RGB",(32,32))
# 	for x in range(32):
# 		for y in range(32):
# 			im.putpixel((x,y),
# 					(
# 						data[nn][y][x][0],
# 						data[nn][y][x][1],
# 						data[nn][y][x][2]
# 					)
# 				)
# 	im.save("test.jpg")
# print(labels[nn])
# testImage()
# exit()

#======================================================================
label_placeholder = tf.placeholder(tf.float32,shape=[None,10])
input_node = tf.placeholder(tf.float32,shape=[None,32,32,3])
dropout_keeprate = tf.placeholder(tf.float32)

def convolutionUnit(node,shape): #kernel size x&y, features in&out
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
	#w = tf.nn.l2_normalize(w,0)
	b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
	conv = tf.nn.convolution(node,w,"SAME") + b
	return conv

# def residualSection(node,features1,features2):
# 	n1 = convolutionUnit(tf.nn.relu(node),[1,1,features2,features1]) #1x1 convoluion
# 	n1 = tf.nn.dropout(n1,dropout_keeprate)

# 	n2a = convolutionUnit(tf.nn.relu(n1),[1,3,features1,features1]) #vertical convolution
# 	n2b = convolutionUnit(tf.nn.relu(n1),[3,1,features1,features1]) #horizontal convolution
# 	n2 = tf.concat((n2a,n2b),axis=3)
# 	#n2 = tf.nn.dropout(n2,dropout_keeprate)

# 	n3 = convolutionUnit(tf.nn.relu(n2),[1,1,features1*2,features2]) #1x1 convoluion
# 	#n3 = tf.nn.dropout(n3,dropout_keeprate)

# 	#pool = tf.nn.max_pool(n3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 	return tf.nn.l2_normalize(n3 + node , 3)
# def residualSection(node,features1,features2):
# 	n1 = convolutionUnit(tf.nn.relu(node),[1,1,features2,features1]) #1x1 convoluion
# 	n1 = tf.nn.dropout(n1,dropout_keeprate)
# 	#n1 = tf.nn.l2_normalize(n1,0)

# 	n2 = convolutionUnit(tf.nn.relu(n1),[3,3,features1,features1]) #vertical convolution
# 	#n2 = tf.nn.dropout(n2,dropout_keeprate)
# 	#n2 = tf.nn.l2_normalize(n2,0)

# 	n3 = convolutionUnit(tf.nn.relu(n2),[1,1,features1,features2]) #1x1 convoluion
# 	n3 = tf.nn.dropout(n3,dropout_keeprate)
# 	#n3 = tf.nn.l2_normalize(n3,0)

# 	return n3 + node
def residualSection(node,features1,features2):
	n1 = convolutionUnit(tf.nn.relu(node),[3,3,features2,features1]) #1x1 convoluion
	n2 = convolutionUnit(tf.nn.relu(n1),[3,3,features1,features1]) #vertical convolution

	return tf.nn.l2_normalize(n2 + node , 0)

def denseUnit(node,shape):
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
	b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
	x = tf.nn.relu(tf.matmul(node,w)+b)
	#x = tf.nn.l2_normalize(x,0)
	return x

l1 = tf.nn.relu(convolutionUnit(input_node,[3,3,3,32]))
l1 = tf.nn.relu(convolutionUnit(l1,[3,3,32,64]))
for _ in range(2):
	l1 = residualSection(l1,64,64)
l2 = tf.nn.max_pool( convolutionUnit(l1,[3,3,64,128]) , [1,2,2,1],[1,2,2,1],"SAME") #16x16
# l2 = tf.nn.l2_normalize(l2,0)
for _ in range(2):
	l2 = residualSection(l2,128,128)
l3 = tf.nn.max_pool( convolutionUnit(l2,[3,3,128,256]) , [1,2,2,1],[1,2,2,1],"SAME") #8x8
# l3 = tf.nn.l2_normalize(l3,0)
for _ in range(2):
	l3 = residualSection(l3,256,256)
l4 = convolutionUnit(l3,[3,3,256,512])
# l4 = tf.nn.l2_normalize(l4,0)
for _ in range(2):
	l4 = residualSection(l4,512,512)
l5 = tf.nn.max_pool(l4,[1,2,2,1],[1,2,2,1],"SAME") #4x4
l5 = tf.reshape(l5,[-1,4*4*512])
l5 = tf.nn.dropout(l5,dropout_keeprate)
# l5 = tf.nn.l2_normalize(l5,0)

l6 = denseUnit(l5,[4*4*512,1024])
#l6 = tf.nn.dropout(l6,dropout_keeprate)
l6 = denseUnit(l6,[1024,512])
l6 = tf.nn.dropout(l6,dropout_keeprate)

w = tf.Variable(tf.truncated_normal([512,10], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))
output = tf.matmul(l6,w)+b

#======================================================================

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_placeholder,logits=output)
cross_entropy = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(label_placeholder, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 75
max_epochs = 10000
drop_keep = 0.75
for epoch in range(max_epochs+1):
	if(epoch%100 == 0):
		acc = 0
		for x in range(int(max_epochs/batch_size)):
			xx = batch_size*x
			acc += sess.run(accuracy,feed_dict={
					input_node:test_data[xx:xx+batch_size],
					label_placeholder:test_labels[xx:xx+batch_size],
					dropout_keeprate:1
				})
		print(epoch,acc/(max_epochs/batch_size))
	indicies = choice(data.shape[0] , batch_size)
	batch_labels = labels[indicies]
	batch_data = data[indicies]
	sess.run(optimizer,feed_dict={
			input_node:batch_data,
			label_placeholder:batch_labels,
			dropout_keeprate:drop_keep
		})