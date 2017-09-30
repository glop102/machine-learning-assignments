import tensorflow as tf
from math import log
from random import shuffle,sample

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
	capital_gain_buckets_log = lambda x: make_onehot_bucket_from_range(log(int(x)+1,10),0,9,10)
	capital_loss = lambda x: [scale(int(x),0,50000)]
	capital_loss_buckets = lambda x: make_onehot_bucket_from_range(int(x),0,50000,50)
	capital_loss_buckets_log = lambda x: make_onehot_bucket_from_range(log(int(x)+1,10),0,9,10)
	hours_per_week = lambda x: [scale(int(x),0,80)] # assuming max number is 80 hours
	hours_per_week_buckets = lambda x: make_onehot_bucket_from_range(int(x),0,120,12)
	native_country = make_onehot_dict("United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands")

	income = make_onehot_dict("<=50K, >50K")

	data = []
	answers = []
	for line in f:
		line = line.split(",")
		line_data = []
		try:
			line_data += age(line[0].strip())
			line_data += age_buckets(line[0].strip())
			line_data += workclass[line[1].strip()]
			line_data += fnlwgt(line[2].strip())
			line_data += fnlwgt_buckets(line[2].strip())
			line_data += education[line[3].strip()]
			line_data += education_num(line[4].strip())
			line_data += education_num_buckets(line[4].strip())
			line_data += marital_status[line[5].strip()]
			line_data += occupation[line[6].strip()]
			line_data += relationship[line[7].strip()]
			line_data += race[line[8].strip()]
			line_data += sex[line[9].strip()]
			line_data += capital_gain(line[10].strip())
			line_data += capital_gain_buckets(line[10].strip())
			line_data += capital_gain_buckets_log(line[10].strip())
			line_data += capital_loss(line[11].strip())
			line_data += capital_loss_buckets(line[11].strip())
			line_data += capital_loss_buckets_log(line[11].strip())
			line_data += hours_per_week(line[12].strip())
			line_data += hours_per_week_buckets(line[12].strip())
			line_data += native_country[line[13].strip()]
		except:
			continue

		answers.append(income[line[14].strip()])
		data.append(line_data)
		# print(len(line_data))
		# print(line_data)
		# break
	return list(zip(data,answers))

training_data = parse_data_file("adult.data")
shuffle(training_data)
print("Num of entries : {}".format(len(training_data)))
print("Len of entries : {}".format(len(training_data[0][0])))
print("Len of answers : {}".format(len(training_data[0][1])))
print()

test_data = parse_data_file("adult.test")
test_data = list(zip(*test_data))

#===============================================================================================================================================================================

full_layer = lambda i,x,y: tf.nn.relu(
							tf.matmul(i , tf.Variable(tf.truncated_normal([x,y],stddev=0.1)))
							+tf.Variable(tf.constant(0.1,shape=[y]))
						 )

input_size = len(training_data[0][0])
output_size = len(training_data[0][1])

input_ = tf.placeholder(tf.float32,[None,input_size])
labels =tf.placeholder(tf.float32,[None,output_size])
dropout_rate = tf.placeholder(tf.float32)

deep = full_layer( input_ , input_size,500 )
deep = full_layer( deep, 500,250 )
deep = full_layer( deep, 250,125 )
deep = full_layer( deep, 125,70 )
wide = full_layer( input_ , input_size,500 )
total= tf.concat([deep,wide],axis=1)
total= tf.nn.dropout(total,dropout_rate)

output_ = tf.matmul(total , tf.Variable(tf.truncated_normal([570,output_size],stddev=0.1)))

loss = tf.losses.sigmoid_cross_entropy(labels,output_)
optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
accuracy = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(output_, 1), tf.argmax(labels, 1)) , tf.float32))

#===============================================================================================================================================================================

validation_size = int( len(training_data) * 0.15 )
validation_data = training_data[:validation_size]
validation_data = list(zip(*validation_data))
training_data   = training_data[validation_size:]
del validation_size

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(2000):
		batch = sample(training_data,150)
		batch = list(zip(*batch))
		sess.run(optimizer,feed_dict = {input_:batch[0],labels:batch[1],dropout_rate:0.15})
		if epoch%100 == 0:
			print(epoch,sess.run(accuracy,feed_dict = {input_:validation_data[0],labels:validation_data[1],dropout_rate:1}))


	#and now finally the test data
	print("\nfinal accuracy",sess.run(accuracy,feed_dict = {input_:test_data[0],labels:test_data[1],dropout_rate:1}))