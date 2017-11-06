#https://github.com/keon/deep-q-learning
import gym
import tensorflow as tf
import numpy as np
from tfUtils import *
from collections import deque
#print(gym.envs.registry.all()) #available learning enviroments

env = gym.make("CartPole-v0")
env.reset()

memory_size = 1000
memory = deque(maxlen=memory_size)

y = 0.95 #discount rate for future gains - higher makes it favor future more
e = 0.1 #chance of forcing a random action

def denseUnit_custom(node,shape):
	w = tf.Variable(tf.random_normal(shape, stddev=0.1,dtype=tf.float32))
	b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
	x = tf.matmul(node,w)+b
	appendWeightNorm(w)
	return x

#=====================================================================
#make the network
input_state = tf.placeholder(tf.float32,shape=[None,4])
# normalised_input = tf.nn.l2_normalize(input_state,dim=0)
# l1 = denseUnit(input_state,[4,10])
l1 = denseUnit(input_state,[4,10])
output_values = denseUnit(l1,[10,2])
# output_values = tf.nn.tanh(denseUnit_noActivation(l1,[10,2]))
# output_values = tf.nn.sigmoid(denseUnit_noActivation(l1,[10,2]))
output_action = tf.argmax(output_values,axis=1) #the index of the highest value

wanted_output = tf.placeholder(tf.float32,shape=[None,2])
loss = tf.reduce_sum(tf.square(wanted_output - output_values),axis=1)
# loss = tf.reduce_sum(tf.square(wanted_output - output_values) + (0.001 * weight_squared),axis=1)
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=wanted_output,logits=output_values)

#=====================================================================
#do the learning
sess = tf.Session()
curt_state = env.reset()
step_counter = 0
reward_total = 0
actions_taken=[]
prevAction = 0
numPrevAction = 0

# trainer = tf.train.AdamOptimizer()
trainer = tf.train.GradientDescentOptimizer(0.0001)
optimize = trainer.minimize(loss)
sess.run(tf.global_variables_initializer())
sequence = []

def predictWholeSequence(sess,e):
	sequence = []
	actions_taken = []
	reward_total = 0
	curt_state = env.reset()
	step_counter = 0
	while True:
		env.render()
		step_counter += 1
		#Chose an action for the current state
		action,outputs = sess.run([output_action,output_values],feed_dict={input_state:[curt_state]})
		if np.random.rand(1) < e:
			action[0] = env.action_space.sample()
		actions_taken.append(action[0])

		#see what the action does
		next_state, reward, done, info = env.step(action[0])
		reward_total += reward
		#and what the network wants to do for the new state
		outputs_next = sess.run(output_values,feed_dict={input_state:[next_state]})

		sequence.append({
			"curt_state":curt_state,
			"next_state":next_state,
			"reward":reward,
			"reward_total":reward_total,
			"action":action[0],
			"done":done
			})
		curt_state = next_state
		print(outputs[0])
		if done == True:
			# Reduce chance of random action as we train the model.
			# e = 1./((step_counter/50) + 10)
			line = ""
			for x in actions_taken : line += str(x)
			print(line)
			actions_taken=[]
			step_counter = 0
			return sequence
def trainNet(sess,seq):
	max_reward = seq[-1]["reward_total"] / 10
	prevAction = 0
	numPrevAction = 0
	for step in seq:
		curt_state = step["curt_state"]
		reward = step["reward"]
		action = step["action"]
		if action != prevAction:
			prevAction = action
			numPrevAction = 0
		outputs      = sess.run(output_values,feed_dict={input_state:[curt_state]})
		outputs_next = sess.run(output_values,feed_dict={input_state:[step["next_state"]]})

		# delta = reward + ( y*np.max(outputs_next))
		# delta = max_reward + ( y*np.max(outputs_next)) - numPrevAction
		# delta = max_reward + y*( np.max(outputs_next) ) - (numPrevAction)
		# target_outputs = np.array( [[0]*len(outputs_next[0])] ,dtype=np.float32)
		target_outputs = outputs.copy()
		target_outputs[0,action] = delta

		sess.run(optimize,feed_dict={input_state:[curt_state],wanted_output:target_outputs})

# while True:
# 	seq = predictWholeSequence(sess,e)
# 	trainNet(sess,seq)

while True:
	env.render()
	step_counter += 1
	#Chose an action for the current state
	action,outputs = sess.run([output_action,output_values],feed_dict={input_state:[curt_state]})
	if np.random.rand(1) < e:
		action[0] = env.action_space.sample()
	actions_taken.append(action[0])
	if action[0] != prevAction:
		prevAction = action[0]
		numPrevAction = 0

	#see what the action does
	next_state, reward, done, info = env.step(action[0])
	reward_total += reward
	#and what the network wants to do for the new state
	outputs_next = sess.run(output_values,feed_dict={input_state:[next_state]})

	# how much do we want to reinforce the action taken?
	delta = reward + ( y*np.max(outputs_next))
	# delta = reward + y*( np.max(outputs_next)-np.min(outputs_next) ) - (numPrevAction**2)
	# delta = reward + y*( np.max(outputs_next)-np.min(outputs_next) ) - (numPrevAction)
	# delta = reward + ( y*np.max(outputs_next)) - ( y*np.min(outputs_next)) - (numPrevAction**2)
	# delta = reward + ( y*np.max(outputs_next)) - (numPrevAction**2)
	# delta = reward_total + y*np.max(outputs_next)
	# delta = y*np.max(outputs_next)
	# delta = outputs[0,action[0]]
	# if done:
	# 	delta = reward_total
	# put the amount to reinforce as the only delta of the output to have it train
	target_outputs = outputs.copy()
	# target_outputs = np.array( [[0]*len(outputs[0])] ,dtype=np.float32)
	# target_outputs[0,action[0]] = delta
	target_outputs[0,action[0]] += delta
	# target_outputs[0,action[0]] = 1
	print(action,outputs,target_outputs)

	curt_state = next_state
	if done == True:
		curt_state = env.reset()
		reward_total = 0
		# Reduce chance of random action as we train the model.
		e = 1./((step_counter/50) + 10)
		# e -= 0.001
		# e *= 0.999

		#pretty little progres bar of how long it runs before being "done"
		# print((" "*int(step_counter)) + "#")
		line = ""
		for x in actions_taken : line += str(x)
		print(line)
		actions_taken=[]
		step_counter = 0
		memory.append(sequence)
		# break

	sess.run(optimize,feed_dict={input_state:[curt_state],wanted_output:target_outputs})

	# print(next_state)
	# print(env.action_space)
	# print(reward)



#TODO LIST
#little NN stuff above for DeepQ
#clean room - mostly bags and bottles
#clean room - move food around
#NN stuff again
#laundry
#take check to bank