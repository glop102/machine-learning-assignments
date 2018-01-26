#https://github.com/keon/deep-q-learning
import gym
import tensorflow as tf
import numpy as np
from tfUtils import *
# print(gym.envs.registry.all()) #available learning enviroments
# exit()

# env = gym.make("CartPole-v0")
# env = gym.make("CartPole-v1")
# env = gym.make("SpaceInvaders-ram-v0")
env = gym.make("SpaceInvaders-ram-v4")
# env = gym.make("SpaceInvaders-ramNoFrameskip-v0")
# env = gym.make("Pitfall-ram-v0")

input_size = env.reset().shape[0]
num_actions_available = env.action_space.n

values_file = open("estimated_values.csv","w")
actions_file = open("actions_taken.csv","w")

#=====================================================================
class Network:
	def __init__(self):
		self.dropout_rate = tf.placeholder(tf.float32)
		#make the network
		self.input_state = tf.placeholder(tf.float32,shape=[None,input_size])
		self.input_state_norm = batchNorm(self.input_state)
		self.input_state_dropout = tf.nn.dropout(self.input_state_norm,self.dropout_rate)

		self.l1 = denseUnit(self.input_state_dropout,[input_size,200])
		self.l1_norm = batchNorm(self.l1)
		self.l1_dropout = tf.nn.dropout(self.l1_norm,self.dropout_rate)

		self.l2 = denseUnit(self.l1_dropout,[200,250])
		self.l2_norm = batchNorm(self.l2)
		self.l2_dropout = tf.nn.dropout(self.l2_norm,self.dropout_rate)

		self.l3 = denseUnit(self.l2_dropout,[250,200])
		self.l3_norm = batchNorm(self.l3)
		self.l3_dropout = tf.nn.dropout(self.l3_norm,self.dropout_rate)

		self.l4 = denseUnit(self.l3_dropout,[200,150])
		self.l4_norm = batchNorm(self.l4)
		self.l4_dropout = tf.nn.dropout(self.l4_norm,self.dropout_rate)

		self.wide = denseUnit(self.input_state_dropout,[input_size,200])
		self.wide_norm = batchNorm(self.wide)
		self.wide_dropout = tf.nn.dropout(self.wide_norm,self.dropout_rate)

		self.combined = tf.concat([self.wide_dropout,self.l4_dropout],axis=1)

		self.output_values = denseUnit_noActivation(self.combined,[350,num_actions_available])
		self.output_action = tf.argmax(self.output_values,axis=1) #the index of the highest value

		self.wanted_output = tf.placeholder(tf.float32,shape=[None])
		self.action_used = tf.placeholder(tf.int32,shape=[None])
		self.masked_outputs = self.output_values * tf.one_hot(self.action_used,num_actions_available)
		self.deltas = tf.square(self.wanted_output - tf.reduce_sum(self.masked_outputs))
		# self.loss = tf.reduce_sum(self.deltas)
		self.loss = tf.reduce_sum(self.deltas) + (0.001 * weight_squared)
	def get_weights(self):
		return [
			self.input_state_norm.mean,
			self.input_state_norm.variance,
			self.input_state_norm.offset,
			self.input_state_norm.scale,

			self.l1.weights,
			self.l1.biases,
			self.l1_norm.mean,
			self.l1_norm.variance,
			self.l1_norm.offset,
			self.l1_norm.scale,

			self.l2.weights,
			self.l2.biases,
			self.l2_norm.mean,
			self.l2_norm.variance,
			self.l2_norm.offset,
			self.l2_norm.scale,

			self.l3.weights,
			self.l3.biases,
			self.l3_norm.mean,
			self.l3_norm.variance,
			self.l3_norm.offset,
			self.l3_norm.scale,

			self.l4.weights,
			self.l4.biases,
			self.l4_norm.mean,
			self.l4_norm.variance,
			self.l4_norm.offset,
			self.l4_norm.scale,

			self.wide.weights,
			self.wide.biases,
			self.wide_norm.mean,
			self.wide_norm.variance,
			self.wide_norm.offset,
			self.wide_norm.scale,

			self.output_values.weights,
			self.output_values.biases
		]
student = Network()
# teacher = Network()
# teacher.setup_copy(student)

#=====================================================================
#do the learning
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333) #ratio of how much vram it allocates
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
curt_state = env.reset()
actions_taken=[]
action_values=[]
epoch_counter = 0
reward_total = 0

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(var_list=student.get_weights())
# saver.restore(sess,"./model-3991")
saver.restore(sess,tf.train.latest_checkpoint("."))

#just checking the values of the batch vars
varss = sess.run(student.get_weights())
for var in varss:
	if len(var) == 1:
		print(var[0])

while True:
	env.render()
	#Chose an action for the current state
	action,values = sess.run([student.output_action,student.output_values],feed_dict={student.input_state:[curt_state],student.dropout_rate:1.0})
	actions_taken.append(action[0])
	action_values.append(values[0][action[0]])
	# print(values[0])

	#see what the action does
	try:
		next_state, reward, done, info = env.step(action[0])
		next_state = next_state
		reward_total += reward
	except:
		print(action)
		curt_state = env.reset()
		continue

	curt_state = next_state
	if done == True:
		for x in range(len(actions_taken)):
			values_file.write(str(action_values[x])+",")
			actions_file.write(str(actions_taken[x])+",")
		values_file.write("\n")
		actions_file.write("\n")

		curt_state = env.reset()

		#pretty little progres bar
		line = ""
		for x in actions_taken : line += str(x)
		print(reward_total,line)
		actions_taken=[]
		action_values=[]
		reward_total = 0
