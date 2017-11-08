#https://github.com/keon/deep-q-learning
import gym
import tensorflow as tf
import numpy as np
from tfUtils import *
from collections import deque
#print(gym.envs.registry.all()) #available learning enviroments

env = gym.make("CartPole-v0")
env.reset()

memory_size = 10000
memory = deque(maxlen=memory_size)

y = 0.95 #discount rate for future gains - higher makes it favor future more
e = 1.0 #chance of forcing a random action

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
actions_taken=[]
epoch_counter = 0

# trainer = tf.train.AdamOptimizer()
trainer = tf.train.GradientDescentOptimizer(0.0001)
optimize = trainer.minimize(loss)
sess.run(tf.global_variables_initializer())

def trainNet():
	seq = np.random.choice(memory,size=100)
	states=[]
	actions=[]
	wanted=[]
	for step in seq:
		curt_state = step["curt_state"]
		next_state = step["next_state"]
		reward     = step["reward"]
		action     = step["action"]
		outputs      = sess.run(output_values,feed_dict={input_state:[curt_state]})
		outputs_next = sess.run(output_values,feed_dict={input_state:[next_state]})

		delta = reward + ( y*np.max(outputs_next))
		target_outputs = outputs.copy()
		target_outputs[0,action] = delta

		states.append(curt_state)
		wanted.append(target_outputs[0])

	sess.run(optimize,feed_dict={input_state:states,wanted_output:wanted})

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

	memory.append({
		"curt_state":curt_state,
		"next_state":next_state,
		"reward":reward,
		"action":action,
		"done":done
		})

	trainNet()

	curt_state = next_state
	if done == True:
		curt_state = env.reset()
		reward_total = 0
		# Reduce chance of random action as we train the model.
		# e = 1./((step_counter/50) + 10)
		# e -= 0.001
		e *= 0.998

		#pretty little progres bar of how long it runs before being "done"
		# print((" "*int(step_counter)) + "#")
		line = ""
		for x in actions_taken : line += str(x)
		print(epoch_counter,"{:.4f}".format(e),line)
		actions_taken=[]
		step_counter = 0
		epoch_counter += 1
		# break

	# sess.run(optimize,feed_dict={input_state:[curt_state],wanted_output:target_outputs})

	# print(next_state)
	# print(env.action_space)
	# print(reward)