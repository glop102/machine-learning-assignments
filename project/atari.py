#https://github.com/keon/deep-q-learning
import gym
import tensorflow as tf
import numpy as np
from tfUtils import *
from collections import deque
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
e = 1.0 #chance of forcing a random action
batch_size = 50

input_size = env.reset().shape[0]
num_actions_available = env.action_space.n

# print(input_size,num_actions_available)

class PriorityQueue:
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
		if len(self.memory)>0 and value<self.memory[-1]['value']:
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
		seq = np.random.choice(self.memory,size=num)
		seqq=[]
		for x in seq:
			seqq.append(np.random.choice(x["item"],size=1)[0])
		return seqq

memory = PriorityQueue(1000)

#=====================================================================
#make the network
input_state = tf.placeholder(tf.float32,shape=[None,input_size])
# normalised_input = tf.nn.l2_normalize(input_state,dim=0)
# l1 = denseUnit(input_state,[input_size,10])
l1 = denseUnit(input_state,[input_size,100])
l2 = denseUnit(l1,[100,150])
l3 = denseUnit(l2,[150,150])
l4 = denseUnit(l3,[150,100])
output_values = denseUnit_noActivation(l4,[100,num_actions_available])
# output_values = tf.nn.tanh(denseUnit_noActivation(l1,[10,num_actions_available]))
# output_values = tf.nn.sigmoid(denseUnit_noActivation(l1,[10,num_actions_available]))
output_action = tf.argmax(output_values,axis=1) #the index of the highest value

wanted_output = tf.placeholder(tf.float32,shape=[None,num_actions_available])
loss = tf.reduce_sum(tf.square(wanted_output - output_values),axis=1)
# loss = tf.reduce_sum(tf.square(wanted_output - output_values),axis=1) + (0.001 * weight_squared)

#=====================================================================
#do the learning
sess = tf.Session()
curt_state = env.reset() / 256.0
step_counter = 0
actions_taken=[]
epoch_counter = 0
episode_memory = []
prev_rewards=deque(maxlen=50)

trainer = tf.train.AdamOptimizer()
# trainer = tf.train.GradientDescentOptimizer(0.0001)
optimize = trainer.minimize(loss)
sess.run(tf.global_variables_initializer())

def trainNet():
	# if(len(memory)<batch_size): return
	# seq = np.random.choice(memory,size=batch_size)
	seq = memory.get_samples(batch_size)
	states=[]
	wanted=[]
	for step in seq:
		curt_state = step["curt_state"]
		next_state = step["next_state"]
		reward     = step["reward"]
		action     = step["action"]
		done       = step["done"]
		outputs      = sess.run(output_values,feed_dict={input_state:[curt_state]})
		outputs_next = sess.run(output_values,feed_dict={input_state:[next_state]})

		delta=0
		if not done:
			# delta = reward + ( gamma*outputs_next[0,action])
			delta = reward + ( gamma*np.max(outputs_next))
		else:
			delta = reward
		target_outputs = outputs[0].copy()
		target_outputs[action] = delta

		states.append(curt_state)
		wanted.append(target_outputs)

	sess.run(optimize,feed_dict={input_state:states,wanted_output:wanted})

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return np.array(discounted_r)

while e>0.15:
	step_counter += 1
	#Chose an action for the current state
	action = sess.run(output_action,feed_dict={input_state:[curt_state]})
	if np.random.rand(1) < e:
		action[0] = env.action_space.sample()
	actions_taken.append(action[0])

	#see what the action does
	try:
		next_state, reward, done, info = env.step(action[0])
		next_state = next_state / 256.0
	except:
		print("error when taking action ",action)
		curt_state = env.reset() / 256.0
		continue

	episode_memory.append([curt_state.copy(),next_state.copy(),reward,action[0],done])

	curt_state = next_state
	if done == True:
		curt_state = env.reset() / 256.0
		# Reduce chance of random action as we train the model.
		# e = 1./((step_counter/50) + 10)
		# e -= 0.001
		if e>0:
			e -= 0.001
			# e -= 0.00005
			# e *= 0.998
		#pretty little progres bar of how long it runs before being "done"
		# print((" "*int(step_counter)) + "#")
		line = ""
		for x in actions_taken : line += str(x)
		print(epoch_counter,"{:.4f}".format(e),line)
		actions_taken=[]
		step_counter = 0
		epoch_counter += 1

		#train the net
		# reward = discount_rewards(np.array(episode_memory)[:,2])
		reward = np.array(episode_memory)[:,2]
		reward_total = sum(reward)
		mem = []
		for i,x in enumerate(episode_memory):
			mem.append({
				"curt_state":x[0],
				"next_state":x[1],
				"reward":reward[i],
				"action":x[3],
				"done":x[4]
				})
		memory.append(mem,reward_total)
		trainNet()
		episode_memory=[]



while True:
	env.render()
	#Chose an action for the current state
	action = sess.run(output_action,feed_dict={input_state:[curt_state]})
	actions_taken.append(action[0])

	#see what the action does
	try:
		next_state, reward, done, info = env.step(action[0])
	except:
		print(action)
		curt_state = env.reset()
		continue

	curt_state = next_state
	if done == True:
		curt_state = env.reset()

		#pretty little progres bar
		line = ""
		for x in actions_taken : line += str(x)
		print("{:.4f}".format(e),line)
		actions_taken=[]


"""
HISTORY of changes

trying to get atari to work
chose to do invaders (ram)

started with agent learning to only stay still
changed discount rate - 0.95, 0.98, 0.99, 0.9
changed the memory/batch size - 10k/25 10k/50 25k/50
changed reward system to use the known end use of the episode instead of the estimated reward
changed it back to use the estimated reward

changed to Adam optimizer - started working to some degree
	-> learned to move slightly and fire with these
changed discount rate to 0.99 - worked better but not enough
made network 3 hidden deep, memory 25k, batch 50 - *slightly* better

changed the e-greedy reduction from -0.001 to -0.0002, effectivly changing the training from 1000 episodes to 5000
	-> agent moves to middle of screen and fires semi-effectivly with ~50% hit rate
	-> agent seems to move towards middle if it is far outside of the area the enemies exist
	-> game1.png and game2.png
second run with no changes to params
	-> learned to either not move or to only move right while firing
	-> made me think that i should have it add more random if the max average score has platued

tried at 50k training episodes (20k memory and 100 batch size)
	-> ran overnight because would take so long
	-> unfortunantly it also only moves to the right while firing

changed to proritized learning queue (more likely to chose better episodes)
	-> implemented to remember top 1000 episodes ever played
	-> yes i know it is not a "priority" queue but it is fine for a first step
"""