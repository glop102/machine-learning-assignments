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
# env = gym.make("SpaceInvaders-ram-v0")
env = gym.make("SpaceInvaders-ram-v4")
# env = gym.make("SpaceInvaders-ramNoFrameskip-v0")
# env = gym.make("Pitfall-ram-v0")
# env = gym.make("MsPacman-ram-v0")
env.reset()

# memory_size = 50000
# memory = deque(maxlen=memory_size)

gamma = 0.99 #discount rate for future gains - higher makes it favor future more
e = 0.4 #chance of forcing a random action
delta_e = -0.001
min_e = 0.0
e_reset=0.4
batch_size = 100
num_training_loops = 9
learning_dropout_rate = 0.8

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
		self.input_state_dropout = tf.nn.dropout(self.input_state_norm,self.dropout_rate)

		self.l1 = denseUnit(self.input_state_dropout,[input_size,200])
		self.l1_norm = batchNorm(self.l1)
		# self.l1_dropout = tf.nn.dropout(self.l1_norm,self.dropout_rate)

		self.l2 = denseUnit(self.l1_norm,[200,250])
		self.l2_norm = batchNorm(self.l2)
		self.l2_dropout = tf.nn.dropout(self.l2_norm,self.dropout_rate)

		self.l3 = denseUnit(self.l2_dropout,[250,200])
		self.l3_norm = batchNorm(self.l3)
		# self.l3_dropout = tf.nn.dropout(self.l3_norm,self.dropout_rate)

		self.l4 = denseUnit(self.l3_norm,[200,150])
		self.l4_norm = batchNorm(self.l4)
		self.l4_dropout = tf.nn.dropout(self.l4_norm,self.dropout_rate)

		self.wide = denseUnit(self.input_state_dropout,[input_size,200])
		self.wide_norm = batchNorm(self.wide)
		# self.wide_dropout = tf.nn.dropout(self.wide_norm,self.dropout_rate)

		self.combined = tf.concat([self.wide_norm,self.l4_dropout],axis=1)

		self.output_values = denseUnit_noActivation(self.combined,[350,num_actions_available])
		self.output_action = tf.argmax(self.output_values,axis=1) #the index of the highest value

		self.wanted_output = tf.placeholder(tf.float32,shape=[None])
		self.action_used = tf.placeholder(tf.int32,shape=[None])
		self.masked_outputs = self.output_values * tf.one_hot(self.action_used,num_actions_available)
		self.deltas = tf.square(self.wanted_output - tf.reduce_sum(self.masked_outputs))
		# self.loss = tf.reduce_sum(self.deltas)
		self.loss = tf.reduce_sum(self.deltas) + (0.001 * weight_squared)
	def setup_copy(self, other):
		self.copyImean     = self.input_state_norm.mean.assign(     other.input_state_norm.mean     )
		self.copyIvariance = self.input_state_norm.variance.assign( other.input_state_norm.variance )
		self.copyIoffset   = self.input_state_norm.offset.assign(   other.input_state_norm.offset   )
		self.copyIscale    = self.input_state_norm.scale.assign(    other.input_state_norm.scale    )

		self.copy1w        = self.l1.weights.assign(       other.l1.weights       )
		self.copy1b        = self.l1.biases.assign(        other.l1.biases        )
		self.copy1mean     = self.l1_norm.mean.assign(     other.l1_norm.mean     )
		self.copy1variance = self.l1_norm.variance.assign( other.l1_norm.variance )
		self.copy1offset   = self.l1_norm.offset.assign(   other.l1_norm.offset   )
		self.copy1scale    = self.l1_norm.scale.assign(    other.l1_norm.scale    )

		self.copy2w        = self.l2.weights.assign(       other.l2.weights       )
		self.copy2b        = self.l2.biases.assign(        other.l2.biases        )
		self.copy2mean     = self.l2_norm.mean.assign(     other.l2_norm.mean     )
		self.copy2variance = self.l2_norm.variance.assign( other.l2_norm.variance )
		self.copy2offset   = self.l2_norm.offset.assign(   other.l2_norm.offset   )
		self.copy2scale    = self.l2_norm.scale.assign(    other.l2_norm.scale    )

		self.copy3w        = self.l3.weights.assign(       other.l3.weights       )
		self.copy3b        = self.l3.biases.assign(        other.l3.biases        )
		self.copy3mean     = self.l3_norm.mean.assign(     other.l3_norm.mean     )
		self.copy3variance = self.l3_norm.variance.assign( other.l3_norm.variance )
		self.copy3offset   = self.l3_norm.offset.assign(   other.l3_norm.offset   )
		self.copy3scale    = self.l3_norm.scale.assign(    other.l3_norm.scale    )

		self.copy4w        = self.l4.weights.assign(       other.l4.weights       )
		self.copy4b        = self.l4.biases.assign(        other.l4.biases        )
		self.copy4mean     = self.l4_norm.mean.assign(     other.l4_norm.mean     )
		self.copy4variance = self.l4_norm.variance.assign( other.l4_norm.variance )
		self.copy4offset   = self.l4_norm.offset.assign(   other.l4_norm.offset   )
		self.copy4scale    = self.l4_norm.scale.assign(    other.l4_norm.scale    )

		self.copyWw        = self.wide.weights.assign(       other.wide.weights       )
		self.copyWb        = self.wide.biases.assign(        other.wide.biases        )
		self.copyWmean     = self.wide_norm.mean.assign(     other.wide_norm.mean     )
		self.copyWvariance = self.wide_norm.variance.assign( other.wide_norm.variance )
		self.copyWoffset   = self.wide_norm.offset.assign(   other.wide_norm.offset   )
		self.copyWscale    = self.wide_norm.scale.assign(    other.wide_norm.scale    )

		self.copyOw = self.output_values.weights.assign( other.output_values.weights )
		self.copyOb = self.output_values.biases.assign(  other.output_values.biases  )
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

		sess.run(self.copy3w)
		sess.run(self.copy3b)
		sess.run(self.copy3mean)
		sess.run(self.copy3variance)
		sess.run(self.copy3offset)
		sess.run(self.copy3scale)

		sess.run(self.copy4w)
		sess.run(self.copy4b)
		sess.run(self.copy4mean)
		sess.run(self.copy4variance)
		sess.run(self.copy4offset)
		sess.run(self.copy4scale)

		sess.run(self.copyWw)
		sess.run(self.copyWb)
		sess.run(self.copyWmean)
		sess.run(self.copyWvariance)
		sess.run(self.copyWoffset)
		sess.run(self.copyWscale)

		sess.run(self.copyOw)
		sess.run(self.copyOb)
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
teacher = Network()
teacher.setup_copy(student)

#=====================================================================
#do the learning
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333) #ratio of how much vram it allocates
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
	# print( sess.run([student.masked_outputs,student.deltas],feed_dict={student.input_state:states,student.wanted_output:deltas,student.action_used:actions,student.dropout_rate:0.8}) )
	# exit()
	
	_,loss = sess.run([optimize,student.loss],feed_dict={student.input_state:states,student.wanted_output:deltas,student.action_used:actions,student.dropout_rate:learning_dropout_rate})
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
		# e = 1./((step_counter/50) + 10)
		# e -= 0.001
		if e>0:
			e += delta_e

		if e<min_e+0.001 and num_training_loops>0:
			num_training_loops-=1
			e=e_reset
			bestEffort_memory.clear()
			teacher.perform_copy(sess)
			teacher_available = True

		#pretty little progres bar of how long it runs before being "done"
		# print((" "*int(step_counter)) + "#")
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
		# bestEffort_memory.append(mem,reward_total)
		# normal_memory.append(mem)
		trainNet()
		if epoch_counter%500 == 0:
			saver.save(sess,"./model",global_step=epoch_counter)
		episode_memory=[]
		episode_errors=[]
saver.save(sess,"./model",global_step=epoch_counter)


# while True:
# 	env.render()
# 	#Chose an action for the current state
# 	action,values = sess.run([student.output_action,student.output_values],feed_dict={student.input_state:[curt_state]})
# 	actions_taken.append(action[0])
# 	# print(values[0])
# 	values_file.write(str(values[0][action[0]])+",")
# 	actions_file.write(str(action[0])+",")

# 	#see what the action does
# 	try:
# 		next_state, reward, done, info = env.step(action[0])
# 		next_state = next_state / 256.0
# 	except:
# 		print(action)
# 		curt_state = env.reset() / 256.0
# 		continue

# 	curt_state = next_state
# 	if done == True:
# 		values_file.write("\n")
# 		actions_file.write("\n")
# 		curt_state = env.reset() / 256.0

# 		#pretty little progres bar
# 		line = ""
# 		for x in actions_taken : line += str(x)
# 		print("{:.4f}".format(e),line)
# 		actions_taken=[]


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
	-> as it it remember the top plays ever, and equaly choses from them

changed epsilon system to loop from .75 -> .16 for a number of times and queue only have 250 slots
	-> does not seem to help

delta_e = -0.001, 2 training loops, 250 in the priority queue, .001 weight normalisation, only 3 hidden layers (input_size,100,150)
	-> moves around L/R while shooting. Does not seem to have much pattern to the movements
	-> video1 & video2

========================================== Problems below
delta_e = -0.001, 2 training loops, 150 in the priority queue, 2500 in normal queue, .001 weight normalisation, only 3 hidden layers (input_size,100,150)
	-> Only went to the right while firing. Perhaps the last one was a fluke, or perhaps i need more training
	-> should figure out way to remove old best plays. thought being that it may have randomly done well and not serious strategy so better to get a good strat instead of noise
Same settings except 6 training loops and the priority queue is reset every time epsilon is set back to 0.75
	-> learned to only move right while firing
Same settings but 26 loops,300 priority, 4000 regular
	-> only choses the single action of moving right
4 loops, priority 100, normal 2500
	-> only right while firing
	-> did notice the prev tests were invalid as the training loop would terminate on hard-coded min_e and not using variable
So after fixing the prev problem, running the test again
	-> moves around in slightly jerky motions, generally prefering the right side
changed loops to 16
	-> OOM error!
changed to 6 loops (and fixed mem leak)
	-> fixed normal memory not removing old entries and so grew mem to stupid size
	-> sat still while firing
noticed the normal memory was not returning actual data to train on so prev tests are invalid
priority memmory not returning any values
	-> bug of it not adding entries into memory
=========================================== Problems Fixed

renamed priority_memory to bestEffort_memory to make the name reflect what it really does
made it copy the network to a second one every training loop to help stabilize the learning (i hope) (aka double-q)
	-> student does the training while the teacher provides the expected next reward
student-teacher thing with 5 loops batch size 40, memory of 2500 and 50 (norm, bestEffort)
	-> weird thing where it will die twice to finish killing the very left column and then move in a jerky fashion to the right
	-> this is doing better than anything else i have tried so far
	-> video3
	-> more training loops next
found that outputs are arund 15k to 19k for estimated value when should be 100 to 500 range
	-> probably something with the "wanted_outputs" in my training
	-> did not fix

Trained for 20 loops
	-> corner peeking around the shields really often
	-> not smart enough to dodge enemy fire
	-> video4
trained for 100 loops (10 freaking hours!!!) 
	-> same corner peeking but stays on left side of the screen
	-> much better at hitting the aliens
	-> larger variation of controls it uses (not move, left/right while firing, firing)
		- does not seem to know how to shoot without moving, interesting
	-> video5

trying with smaller bestEffort queue and longer normal memory (40,5000) , 10 loops
	-> moves to the right but seemed to be starting to understand moving left
	-> will make a proper priority queue tomorrow
much bigger normal memory (250K) and 20 loops
	-> nothing
	-> expected values are too high still, 8k-9k

stole code from some guy to do a priority queue (probably) , 6 loops, batch of 60, only priority queue
	-> man this thing just keeps getting slower and slower, wish i could do it in c++ so i can audit what operations it is actually doing so i can make it fast again
	-> ONLY 46 MINUTES AND IT IS NOT BAD!!!
	-> video6

had it play catpole gamma=0.98
	-> used this to try faster itterations through fixing my code
	-> did many different fiddleing bits that would be insanly tedious to list here
	-> cartpole works amazingly now!!
	-> video7
	-> really weird though how expected values are negative most of the time, but sometimes is positive
back to space invaders
	-> e of 0.4 to 0.0 with delta -0.001
	-> gamma = 0.98
	-> batch size 60
	-> memory size 250k
	-> weight norm 0.001
	-> loops 10
	-> moves right only
lets try again
	-> e of 0.4 to 0.0 with delta -0.001
	-> gamma = 0.995
	-> batch size 60
	-> memory size 250k
	-> weight norm 0.001
	-> loops 10
	-> sits still while shooting
	-> outputs are freaking huge, 4-5 million
lets try again
	-> e of 0.4 to 0.0 with delta -0.001
	-> gamma = 0.99
	-> batch size 60
	-> memory size 500k
	-> weight norm 0.001
	-> loops 10
	-> sits still while shooting
	-> outputs are around 800k-1mill
lets try again
	-> e of 0.3 to 0.0 with delta -0.001
	-> gamma = 0.99
	-> batch size 60
	-> memory size 250k
	-> loops 5
	-> sit still and shoots
	-> output around 18k - 20k
lets try again - changed loss function for reduce_sum (removed axis=1)
	-> e of 0.4 to 0.0 with delta -0.001
	-> gamma = 0.99
	-> batch size 60
	-> memory size 250k
	-> loops 5
	-> gave up because looked like it wasn't working

trying to figure out why the estimated values are so high and just keep getting higher the longer it trains
	-> it does seem to eventually learn and kindof do the right thing, but it should be much smaller numbers
	-> every 250 episodes, it pauses and prints what it is trying to train the values to be
	-> 250 - chosen outputs 50-90 range - gradients all have only 1 worthwhile amount (more than 0.001) per state to train
	-> 500 - chosen outputs 70-100 range - graients are similar
	-> 750 - chosen outputs 50-100 range with single 37
	-> 1000- chosen outputs 40-100 range
	-> 1250- chosen outputs 60-100 range with single 14
	-> 1500- chosen outputs 60-100 range
	-> 1750- chosen outputs 60-100 range
	-> 2000- chosen outputs 50-120 range with single 45,21
	-> 2250- chosen outputs 50-110 range with single 36,44
	-> 2500- chosen outputs 60-120 range with single 21,43,14
	-> 2750- chosen outputs 20-110 range
	-> 3000- chosen outputs 60-110 range with single 26,41,13
	-> 3250- chosen outputs 50-110 range 17,120
	-> 3500- chosen outputs 55-110 range 30,26
	-> 3750- chosen outputs 60-110 range 44,145
	-> 4000- chosen outputs 30-110 range 293
	-> 4250- chosen outputs 60-110 range 48,32,38
	-> seems to do randomish stuff, but does seem to be shooting through the shield to fire at aliens
	-> video8
	-> in non-training and just running mode, the outputs change to 3-30k mostly, but sometimes is in the hundreds, idk what it is doing

lets try again - changed loss function for reduce_sum (removed axis=1)
	-> e of 0.4 to 0.0 with delta -0.001
	-> gamma = 0.99
	-> batch size 60
	-> memory size 250k
	-> loops 11
	-> sat still while shooting
lets try again - changed loss function for reduce_sum (removed axis=1)
	-> e of 0.4 to 0.0 with delta -0.001
	-> gamma = 0.98
	-> batch size 60
	-> memory size 250k
	-> loops 20
	-> 2 hours
	-> mostly jsut sat around
lets try again - 2 hidden layer (down from 3)
	-> e of 0.4 to 0.0 with delta -0.001
	-> gamma = 0.98
	-> batch size 60
	-> memory size 250k
	-> loops 20
	-> 

tried batch norm
	-> got things broken
fixed my variables that i am saving to file
	-> HOLY COW IT IS GOOD AND ONLY 4000 EPISODES!!!
	-> video10
	-> atari_batch folder
i guess overtraining is a thing
	-> sits still after 40 training loops
second try just to see what it will look like
	-> it is getting good
	-> only 4000 episodes (10 loops)
	-> moves around ant routinely get 250-300 score
	-> atari_batch_2 folder
	-> video11
added dropout (only 50% for student in training, but 100% all other times like teacher)
	-> atari_batch_3 folder
	-> 10 loops - which i think is all i need because the loss levels out mostly by 1000 episodes, but is suspicious that it does so because that is the time that the second teacher comes around?
		-> i mean, loss being crazy when challenging itself makes sense, and getting extreme gains on the first teacher makes sense
	-> broken. I think it is because the same dropout is not being used for the training from the first and last runs
tried with a different loss system that should fix the dropout issues
	-> atari_batch_3 folder
	-> 10 loops
	-> only moves right while shooting. Maybe 50% loss is too agressive?

trying batch norm in a different location in the layer (right after weight multiply instead of after activation)
	-> no dropout because it wasn't working perfectly before so am isolating changes
	-> atari_batch_4 folder
	-> no-go, maybe soemthing is wrong because my loss function stops getting smaller after ~500 episodes but is still huge
		-> 100K averate for batch of 100
		-> 1k off per item (wanted-output squared)
		-> 31.6 off per item (wanted-output)

Back to dropout version
	-> 0.8 keep rate instead of 0.5
	-> 20 loops
	-> added third hidden layer and a "wide" layer
	-> meh results
try again with gamma of 0.99
	-> not bad, jumps around an doesn't just hold the shoot button
try with 4 hidden layers
	-> only 4000 episodes and is pretty good
	-> video12
try with 5 hidden layers
	-> 10 loops
	-> not sure if training enough
5 layers
	-> 30 loops
	-> only did 6000 episodes because i got bored
	-> still not very good

screw the 5 layers, lets use 4 again but on a different game
	-> 10 loops
	-> pitfall
	-> HAHAHAHHAHAHAHAHHAHAHAHA ... no
"""