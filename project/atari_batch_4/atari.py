#https://github.com/keon/deep-q-learning
import gym
import tensorflow as tf
import numpy as np
from tfUtils import *
from collections import deque
from PriorityQueue import PriorityQueue
# print(gym.envs.registry.all()) #available learning enviroments
# exit()

# env = gym.make("CartPole-v0")
# env = gym.make("CartPole-v1")
env = gym.make("SpaceInvaders-ram-v0")
# env = gym.make("SpaceInvaders-ramNoFrameskip-v0")
env.reset()

# memory_size = 50000
# memory = deque(maxlen=memory_size)

gamma = 0.98 #discount rate for future gains - higher makes it favor future more
e = 0.4 #chance of forcing a random action
delta_e = -0.001
min_e = 0.0
e_reset=0.4
batch_size = 100
num_training_loops = 9

input_size = env.reset().shape[0]
num_actions_available = env.action_space.n

# values_file = open("estimated_values.csv","w")
loss_file = open("loss_values.csv","w")
# actions_file = open("actions_taken.csv","w")

# print(input_size,num_actions_available)

class BestEffortsQueue:
	"""
	holds a sorted list of items, each item having a value
	the first item is the most valuable
	the last item is the least valuable
	"""
	def __init__(self,maxlen):
		self.maxlen=maxlen
		self.memory=[]
	def __len__(self):
		return len(self.memory)
	def append(self,item,value):
		if len(self.memory)==self.maxlen and value<self.memory[-1]['value']:
			return
		i=0;
		while i<len(self.memory):
			if self.memory[i]['value']<value:
				break
			i+=1
		self.memory.insert(i,{'value':value,'item':item})
		if len(self.memory)>self.maxlen:
			del self.memory[-1]
	def get_samples(self,num):
		if num>len(self.memory):
			return []
		seq = np.random.choice(self.memory,size=num)
		seqq=[]
		for x in seq:
			seqq.append(np.random.choice(x["item"],size=1)[0])
		return seqq
	def clear(self):
		self.memory=[]
class NormalQueue:
	"""
	holds a list of items, each item a single step of an episode
	the first item is the oldest
	the last item is the newest
	"""
	def __init__(self,maxlen):
		self.maxlen=maxlen
		self.memory=[]
	def __len__(self):
		return len(self.memory)
	def append(self,items):
		for item in items:
			self.memory.append(item)
		if len(self.memory)>self.maxlen:
			del self.memory[:-self.maxlen] #keep length of only maxlen
	def get_samples(self,num):
		if num > len(self.memory):
			return []
		seq = np.random.choice(self.memory,size=num)
		return seq
	def clear(self):
		self.memory=[]

bestEffort_memory = BestEffortsQueue(40)
normal_memory = NormalQueue(250000)
priority_memory = PriorityQueue(250000)

#=====================================================================
class Network:
	def __init__(self):
		self.dropout_rate = tf.placeholder(tf.float32)
		#make the network
		self.input_state = tf.placeholder(tf.float32,shape=[None,input_size])
		self.input_state_norm = batchNorm(self.input_state)
		# self.input_state_dropout = tf.nn.dropout(self.input_state_norm,self.dropout_rate)

		self.l1 = denseUnit(self.input_state_norm,[input_size,200])
		# self.l1_dropout = tf.nn.dropout(self.l1,self.dropout_rate)

		self.l2 = denseUnit(self.l1,[200,200])
		# self.l2_dropout = tf.nn.dropout(self.l2,self.dropout_rate)

		# self.l3 = denseUnit(self.l2_norm,[200,200])
		# self.l3_norm = batchNorm(self.l3)
		# self.l3_dropout = tf.nn.dropout(self.l3_norm,self.dropout_rate)

		self.output_values = denseUnit_noActivation(self.l2,[200,num_actions_available])
		self.output_action = tf.argmax(self.output_values,axis=1) #the index of the highest value

		self.wanted_output = tf.placeholder(tf.float32,shape=[None])
		self.action_used = tf.placeholder(tf.int32,shape=[None])
		self.masked_outputs = tf.reduce_sum( self.output_values * tf.one_hot(self.action_used,num_actions_available), axis=1)
		self.deltas = tf.square(self.wanted_output - self.masked_outputs)
		# self.loss = tf.reduce_sum(self.deltas)
		self.loss = tf.reduce_sum(self.deltas) + (0.001 * weight_squared)
	def setup_copy(self, other):
		self.copyImean     = self.input_state_norm.mean.assign(     other.input_state_norm.mean     )
		self.copyIvariance = self.input_state_norm.variance.assign( other.input_state_norm.variance )
		self.copyIoffset   = self.input_state_norm.offset.assign(   other.input_state_norm.offset   )
		self.copyIscale    = self.input_state_norm.scale.assign(    other.input_state_norm.scale    )

		self.copy1w        = self.l1.weights.assign(  other.l1.weights  )
		self.copy1b        = self.l1.biases.assign(   other.l1.biases   )
		self.copy1mean     = self.l1.mean.assign(     other.l1.mean     )
		self.copy1variance = self.l1.variance.assign( other.l1.variance )
		self.copy1offset   = self.l1.offset.assign(   other.l1.offset   )
		self.copy1scale    = self.l1.scale.assign(    other.l1.scale    )

		self.copy2w        = self.l2.weights.assign(  other.l2.weights  )
		self.copy2b        = self.l2.biases.assign(   other.l2.biases   )
		self.copy2mean     = self.l2.mean.assign(     other.l2.mean     )
		self.copy2variance = self.l2.variance.assign( other.l2.variance )
		self.copy2offset   = self.l2.offset.assign(   other.l2.offset   )
		self.copy2scale    = self.l2.scale.assign(    other.l2.scale    )

		self.copyOw        = self.output_values.weights.assign(  other.output_values.weights  )
		self.copyOb        = self.output_values.biases.assign(   other.output_values.biases   )
		self.copyOmean     = self.output_values.mean.assign(     other.output_values.mean     )
		self.copyOvariance = self.output_values.variance.assign( other.output_values.variance )
		self.copyOoffset   = self.output_values.offset.assign(   other.output_values.offset   )
		self.copyOscale    = self.output_values.scale.assign(    other.output_values.scale    )
	def perform_copy(self,sess):
		sess.run(self.copyImean)
		sess.run(self.copyIvariance)
		sess.run(self.copyIoffset)
		sess.run(self.copyIscale)

		sess.run(self.copy1w)
		sess.run(self.copy1b)
		sess.run(self.copy1mean)
		sess.run(self.copy1variance)
		sess.run(self.copy1offset)
		sess.run(self.copy1scale)

		sess.run(self.copy2w)
		sess.run(self.copy2b)
		sess.run(self.copy2mean)
		sess.run(self.copy2variance)
		sess.run(self.copy2offset)
		sess.run(self.copy2scale)

		sess.run(self.copyOw)
		sess.run(self.copyOb)
		sess.run(self.copyOmean)
		sess.run(self.copyOvariance)
		sess.run(self.copyOoffset)
		sess.run(self.copyOscale)
	def get_weights(self):
		return self.input_state_norm.variables+self.l1.variables+self.l2.variables+self.output_values.variables

student = Network()
teacher = Network()
teacher.setup_copy(student)

#=====================================================================
#do the learning
sess = tf.Session()
curt_state = env.reset()
step_counter = 0
actions_taken=[]
epoch_counter = 0
episode_memory = []
teacher_available = False

trainer = tf.train.AdamOptimizer()
saver = tf.train.Saver(var_list=student.get_weights())
# trainer = tf.train.GradientDescentOptimizer(0.0001)
optimize = trainer.minimize(student.loss)
sess.run(tf.global_variables_initializer())

def trainNet():
	# seq = list(bestEffort_memory.get_samples(batch_size//3)) + list(normal_memory.get_samples(batch_size))
	seq = priority_memory.sample(batch_size)
	if len(seq)==0:
		return
	states=[]
	wanted=[]
	deltas=[]
	actions=[]
	for step in seq:
		curt_state = step[1]["curt_state"]
		next_state = step[1]["next_state"]
		reward     = step[1]["reward"]
		action     = step[1]["action"]
		done       = step[1]["done"]
		if not teacher_available:
			outputs_next = sess.run(student.output_values,feed_dict={student.input_state:[next_state],student.dropout_rate:1.0})
		else:
			outputs_next = sess.run(teacher.output_values,feed_dict={teacher.input_state:[next_state],teacher.dropout_rate:1.0})

		delta=0
		if not done:
			delta = reward + ( gamma*np.max(outputs_next))
		else:
			delta = reward

		states.append(curt_state)
		deltas.append(delta)
		actions.append(action)

	# if epoch_counter %250 == 0:
	# 	print(deltas)
	# 	print()
	# 	# print()
	# 	# print(sess.run(student.deltas,feed_dict={student.input_state:states,student.wanted_output:deltas}))
	# 	# exit()
	# 	input()
	# print( sess.run([student.masked_outputs,student.deltas],feed_dict={student.input_state:states,student.wanted_output:deltas,student.action_used:actions,student.dropout_rate:0.5}) )
	# exit()
	
	_,loss = sess.run([optimize,student.loss],feed_dict={student.input_state:states,student.wanted_output:deltas,student.action_used:actions,student.dropout_rate:0.5})
	loss_file.write(str(loss)+"\n")

	#update our error for the states we just trained on
	for step in seq:
		curt_state = step[1]["curt_state"]
		next_state = step[1]["next_state"]
		reward     = step[1]["reward"]
		done       = step[1]["done"]
		outputs,choice = sess.run([student.output_values,student.output_action],feed_dict={student.input_state:[curt_state],student.dropout_rate:1.0})
		outputs_next   = sess.run(student.output_values,feed_dict={student.input_state:[next_state],student.dropout_rate:1.0})
		if not done:
			delta = reward + ( gamma*np.max(outputs_next))
		else:
			delta = reward
		priority_memory.update(step[0],abs(delta-outputs[0][choice[0]]))

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return np.array(discounted_r)

prev_expected_value=0
episode_errors = []
while e>min_e:
	step_counter += 1
	#Chose an action for the current state
	action,predictions = sess.run([student.output_action,student.output_values],feed_dict={student.input_state:[curt_state],student.dropout_rate:1.0})
	if np.random.rand(1) < e:
		action[0] = env.action_space.sample()
	actions_taken.append(action[0])
	current_expected_value = predictions[0][action[0]]

	#see what the action does
	try:
		next_state, reward, done, info = env.step(action[0])
		next_state = next_state
	except:
		print("error when taking action ",action)
		curt_state = env.reset()
		continue

	episode_memory.append([curt_state.copy(),next_state.copy(),reward,action[0],done])
	if not len(episode_memory) == 1:

		episode_errors.append(abs(prev_expected_value - (reward+gamma*current_expected_value)))
		
	prev_expected_value = current_expected_value

	curt_state = next_state
	if done == True:
		curt_state = env.reset()
		# Reduce chance of random action as we train the model.
		if e>0:
			e += delta_e

		if e<min_e+0.001 and num_training_loops>0:
			num_training_loops-=1
			e=e_reset
			bestEffort_memory.clear()
			teacher.perform_copy(sess)
			teacher_available = True

		#pretty little progres bar of how long it runs before being "done"
		line = ""
		for x in actions_taken : line += str(x)
		print(epoch_counter,"{:.4f}".format(e),line)
		actions_taken=[]
		step_counter = 0
		epoch_counter += 1

		#train the net
		# rewards = discount_rewards(np.array(episode_memory)[:,2])
		rewards = np.array(episode_memory)[:,2]
		reward_total = sum(np.array(episode_memory)[:,2])
		episode_errors.append(abs(current_expected_value-reward))
		print(reward_total)
		mem = []
		for i,x in enumerate(episode_memory):
			mem.append({
				"curt_state":x[0],
				"next_state":x[1],
				"reward":rewards[i],
				"action":x[3],
				"done":x[4],
				})
			priority_memory.add(episode_errors[i],mem[-1])
		trainNet()
		if epoch_counter%500 == 0:
			saver.save(sess,"./model",global_step=epoch_counter)
		episode_memory=[]
		episode_errors=[]
saver.save(sess,"./model",global_step=epoch_counter)
