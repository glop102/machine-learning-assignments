import tensorflow as tf
from math import log
#from random import shuffle
from numpy import *
from numpy.random import choice,shuffle
import gc

class empty(): pass
nn = empty()

def make_onehot_dict(options_string):
	#returns a dictionary of prebuilt arrays
	#you pass in a comma seperated list of keys and we make buckets for them
	# "a,b,c" -> {"a":[1,0,0] , "b":[0,1,0] , "c":[0,0,1]}
	keys = options_string.split(",")
	temp = {"?":[0]*len(keys)}
	for k,i in zip(keys,range(len(keys))):
		k = k.strip()
		ar = [0]*len(keys)
		ar[i] = 1
		temp[k] = ar
	return temp
def scale(val,minn,maxx):
	#scales val to be from -1 to 1
	rangee = maxx - minn
	val -= minn
	val /= float(rangee) / 2.0
	return val - 1.0
def make_onehot_bucket_from_range(val,minn,maxx,num_buckets):
	#so if mapping 10 from 1 to 20 with 5 buckets
	#you get [0,0,1,0,0]
	rangee = maxx - minn
	if val<maxx: val -= minn
	else: val = rangee-1
	if val<0: val=0

	bucket_size = rangee/num_buckets
	ar = [0]*num_buckets
	ar[int(val//bucket_size)] = 1
	return ar

def parse_data_file(filename):
	f = open(filename,"r")
	age = lambda x: [scale(int(x),0,80)]
	age_buckets = lambda x: make_onehot_bucket_from_range(int(x),0,80,16)
	workclass = make_onehot_dict("Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked")
	fnlwgt = lambda x : [scale(int(x),0,500000)]
	fnlwgt_buckets = lambda x : make_onehot_bucket_from_range(int(x),0,1000000,100)
	education = make_onehot_dict("Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool")
	education_num = lambda x : [scale(int(x),0,20)]
	education_num_buckets = lambda x : make_onehot_bucket_from_range(int(x),0,20,10)
	marital_status = make_onehot_dict("Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse")
	occupation = make_onehot_dict("Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces")
	relationship = make_onehot_dict("Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried")
	race = make_onehot_dict("White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black")
	sex = make_onehot_dict("Female, Male")
	capital_gain = lambda x: [scale(int(x),0,50000)]
	capital_gain_buckets = lambda x: make_onehot_bucket_from_range(int(x),0,50000,50)
	capital_gain_buckets_log = lambda x: make_onehot_bucket_from_range(int(math.log(int(x)+1,10)),0,9,10)
	capital_loss = lambda x: [scale(int(x),0,50000)]
	capital_loss_buckets = lambda x: make_onehot_bucket_from_range(int(x),0,50000,50)
	capital_loss_buckets_log = lambda x: make_onehot_bucket_from_range(int(math.log(int(x)+1,10)),0,9,10)
	hours_per_week = lambda x: [scale(int(x),0,80)] # assuming max number is 80 hours
	hours_per_week_buckets = lambda x: make_onehot_bucket_from_range(int(x),0,120,12)
	native_country = make_onehot_dict("United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands")

	income = make_onehot_dict("<=50K, >50K")

	data = []
	answers = []
	for line in f:
		line = line.split(",")
		line_data = array([],dtype=float32)
		try:
			line_data = concatenate(( line_data , age(line[0].strip()) ))
			line_data = concatenate(( line_data , age_buckets(line[0].strip()) ))
			line_data = concatenate(( line_data , workclass[line[1].strip()] ))
			line_data = concatenate(( line_data , fnlwgt(line[2].strip()) ))
			line_data = concatenate(( line_data , fnlwgt_buckets(line[2].strip()) ))
			line_data = concatenate(( line_data , education[line[3].strip()] ))
			line_data = concatenate(( line_data , education_num(line[4].strip()) ))
			line_data = concatenate(( line_data , education_num_buckets(line[4].strip()) ))
			line_data = concatenate(( line_data , marital_status[line[5].strip()] ))
			line_data = concatenate(( line_data , occupation[line[6].strip()] ))
			line_data = concatenate(( line_data , relationship[line[7].strip()] ))
			line_data = concatenate(( line_data , race[line[8].strip()] ))
			line_data = concatenate(( line_data , sex[line[9].strip()] ))
			line_data = concatenate(( line_data , capital_gain(line[10].strip()) ))
			line_data = concatenate(( line_data , capital_gain_buckets(line[10].strip()) ))
			line_data = concatenate(( line_data , capital_gain_buckets_log(line[10].strip()) ))
			line_data = concatenate(( line_data , capital_loss(line[11].strip()) ))
			line_data = concatenate(( line_data , capital_loss_buckets(line[11].strip()) ))
			line_data = concatenate(( line_data , capital_loss_buckets_log(line[11].strip()) ))
			line_data = concatenate(( line_data , hours_per_week(line[12].strip()) ))
			line_data = concatenate(( line_data , hours_per_week_buckets(line[12].strip()) ))
			line_data = concatenate(( line_data , native_country[line[13].strip()] ))
		except e:
			print("Skipping line",e)
			continue

		answers.append(array(income[line[14].strip()],dtype=float32))
		data.append(array(line_data,dtype=float32))
		# print(len(line_data))
		# print(line_data)
		# break

	data = array(data)
	answers = array(answers)
	return data,answers

#===============================================================================================================================================================================

def setup_layers(deep_sizes=[100,50,10],wide_size=100):
	full_layer = lambda i,x,y: tf.nn.relu(
								tf.matmul(i , tf.Variable(tf.truncated_normal([x,y],stddev=0.1)))
								+tf.Variable(tf.constant(0.1,shape=[y]))
							 )

	input_size = nn.training_data.shape[1]
	output_size = nn.training_answers.shape[1]

	nn.input_= tf.placeholder(tf.float32,[None,input_size])
	nn.labels =tf.placeholder(tf.float32,[None,output_size])
	nn.dropout_rate_placeholder = tf.placeholder(tf.float32)

	deep = full_layer( nn.input_, input_size,deep_sizes[0] )
	for index in range(1,len(deep_sizes)):
		deep = full_layer( deep, deep_sizes[index-1],deep_sizes[index] )
	wide = full_layer( nn.input_, input_size,wide_size )
	total= tf.concat([deep,wide],axis=1)
	total= tf.nn.dropout(total,nn.dropout_rate_placeholder)

	nn.output_ = tf.matmul(total , tf.Variable(tf.truncated_normal([deep_sizes[-1]+wide_size,output_size],stddev=0.1)))

	loss = tf.losses.sigmoid_cross_entropy(nn.labels,nn.output_)
	nn.optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
	nn.accuracy = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(nn.output_, 1), tf.argmax(nn.labels, 1)) , tf.float32))


#===============================================================================================================================================================================

def train_nn():
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	training_loss=[]
	for epoch in range(nn.epochs+1):
		indexes = choice(nn.training_data.shape[0],250)
		batch_data = nn.training_data[indexes]
		batch_answers = nn.training_answers[indexes]

		sess.run(nn.optimizer,feed_dict = {nn.input_:batch_data,nn.labels:batch_answers,nn.dropout_rate_placeholder:nn.dropout_rate})
		if epoch%50 == 0:
		#	print(epoch,sess.run(nn.accuracy,feed_dict = {nn.input_:nn.validation_data[0],nn.labels:nn.validation_data[1],nn.dropout_rate_placeholder:1}))
			training_loss.append(sess.run(nn.accuracy,feed_dict = {nn.input_:nn.validation_data,nn.labels:nn.validation_answers,nn.dropout_rate_placeholder:1}))


	#and now finally the test data
	#acc = sess.run(nn.accuracy,feed_dict = {nn.input_:nn.test_data[0],nn.labels:nn.test_data[1],nn.dropout_rate_placeholder:1})
	#print("\nfinal accuracy",acc)
	tf.reset_default_graph()
	sess.close()
	return training_loss

#===============================================================================================================================================================================

def run_test():
	nn.training_data,nn.training_answers = parse_data_file("adult.data")
	nn.training_data,nn.training_answers = ill_just_shuffle_it_myself(nn.training_data,nn.training_answers)

	validation_size       = int( nn.training_data.shape[0] * 0.15 )
	nn.validation_data    = nn.training_data[:validation_size]
	nn.validation_answers = nn.training_answers[:validation_size]
	nn.training_data      = nn.training_data[validation_size:]
	nn.training_answers   = nn.training_answers[validation_size:]
	del validation_size

	#nn.test_data = parse_data_file("adult.test")
	#nn.test_data = array(zip(*nn.test_data))
	nn.epochs = 2000

	f = open("results.csv","w")
	f.write("Depth,Depth_Decay,Depth_Start,Dropout_Rate,Width,"+numList_string(nn.epochs,50)+"\n")
	f.flush()

	count=0
	for depth in range(3,7):
		for depth_decay in [x/10.0 for x in range(4,6)]: #range(0.1,0.8,.1)
			for depth_start in range(50,400,50):
				deep_sizes = [int(depth_start*(depth_decay**x)) for x in range(depth)]
				for dropout_rate in [x/100.0 for x in range(15,30,5)]: #range(0.05,0.5,0.05)
					nn.dropout_rate = dropout_rate
					for width in range(50,400,50):
						setup_layers(deep_sizes,width)
						training_loss = train_nn()
						count+=1
						print("did round",count)
						f.write(make_csv_line(depth,depth_decay,depth_start,dropout_rate,width,training_loss)+"\n")
						f.flush()
						gc.collect()
	print("done")

def numList_string(size,step):
	s=""
	for x in range(0,size+1,step):
		s+="epoch"
		s+=str(x)
		s+=","
	return s[:-1] #remove the last comma

def make_csv_line(depth,depth_decay,depth_start,dropout_rate,width,training_loss):
	s=""
	s+=str(depth)+","
	s+=str(depth_decay)+","
	s+=str(depth_start)+","
	s+=str(dropout_rate)+","
	s+=str(width)+","
	for x in training_loss:
		s+=str(x)+","
	return s[:-1] #remove that last comma

def ill_just_shuffle_it_myself(a1,a2):
	length = a1.shape[0]
	ind=arange(length)
	shuffle(ind)
	a1 = a1[ind]
	a2 = a2[ind]
	return a1,a2


if(__name__ == "__main__"):
	run_test()