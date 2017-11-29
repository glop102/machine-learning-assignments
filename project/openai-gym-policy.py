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

gamma = 0.975 #discount rate for future gains - higher makes it favor future more
e = 1.0
batch_size = 25

input_size = env.reset().shape[0]
num_actions_available = env.action_space.n

#=====================================================================
#make the network
input_state = tf.placeholder(tf.float32,shape=[None,input_size])
# normalised_input = tf.nn.l2_normalize(input_state,dim=0)
# l1 = denseUnit(input_state,[input_size,10])
l1 = denseUnit(input_state,[input_size,20])
l2 = denseUnit(l1,[20,40])
l3 = denseUnit(l2,[40,20])
output_values = tf.nn.sigmoid( denseUnit_noActivation(l3,[20,num_actions_available]) )
# output_values = tf.nn.tanh(denseUnit_noActivation(l1,[10,num_actions_available]))
# output_values = tf.nn.sigmoid(denseUnit_noActivation(l1,[10,num_actions_available]))
output_action = tf.argmax(output_values,axis=1) #the index of the highest value

action_taken = tf.placeholder(tf.int32,shape=[None])
reward_given = tf.placeholder(tf.float32,shape=[None])
deltas = tf.one_hot(action_taken,num_actions_available) - output_values
loss = tf.reduce_sum(deltas)*reward_given

# action_taken = tf.placeholder(tf.int32,shape=[None])
# reward_given = tf.placeholder(tf.float32,shape=[None])
# indexes = tf.range(0, tf.shape(output_values)[0]) * tf.shape(output_values)[1] + action_taken
# responsible_outputs = tf.gather(tf.reshape(output_values, [-1]), indexes)
# loss = -tf.reduce_mean(tf.log(responsible_outputs)*reward_given)
#loss = tf.reduce_sum(tf.square(wanted_output - output_values),axis=1) + (0.001 * weight_squared)

#=====================================================================
#do the learning
sess = tf.Session()
curt_state = env.reset()
epoch_counter = 0

# trainer = tf.train.AdamOptimizer()
trainer = tf.train.GradientDescentOptimizer(0.0001)
optimize = trainer.minimize(loss)
sess.run(tf.global_variables_initializer())

def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return np.array(discounted_r)

def trainNet(episode_memory):
	#print(episode_memory[:,0])
	# states=[s for s in episode_memory[:,0]]
	# sess.run(optimize,feed_dict={
	# 		input_state:states,
	# 		action_taken:episode_memory[:,1],
	# 		reward_given:episode_memory[:,2]
	# 	})

	if len(memory)<batch_size: return
	seq = np.array([s.state for s in np.random.choice(memory,size=batch_size) ])
	states=[s for s in seq[:,0]]
	sess.run(optimize,feed_dict={
			input_state:states,
			action_taken:seq[:,1],
			reward_given:seq[:,2]
		})

class holder:
	def __init__(self,s):
		self.state = s

episode_memory=[]
while True:
	env.render()
	#Chose an action for the current state
	action = sess.run(output_action,feed_dict={input_state:[curt_state]})
	if np.random.rand(1) < e:
		action[0] = env.action_space.sample()
	action = action[0]

	#see what the action does
	next_state, reward, done, info = env.step(action)
	episode_memory.append([curt_state,action,reward,next_state])

	curt_state = next_state
	if done == True:
		curt_state = env.reset()
		# Reduce chance of random action as we train the model.
		# e = 1./((step_counter/50) + 10)
		# e -= 0.001
		if e>0:
			e -= 0.001
			# e *= 0.998

		#pretty little progres bar of how long it runs before being "done"
		# print((" "*int(step_counter)) + "#")
		episode_memory = np.array(episode_memory)
		line = ""
		for x in episode_memory[:,1] : line += str(x)
		print(epoch_counter,"{:.4f}".format(e),line)

		episode_memory[:,2] = discount_rewards(episode_memory[:,2])
		for s in episode_memory:
			memory.append(holder(s))
		trainNet(episode_memory)

		episode_memory=[]
		step_counter = 0
		epoch_counter += 1
		# break

	# sess.run(optimize,feed_dict={input_state:[curt_state],wanted_output:target_outputs})

	# print(next_state)
	# print(env.action_space)
	# print(reward)




















exit()

class agent():
	def __init__(self, lr, s_size,a_size,h_size):
		#These lines established the feed-forward part of the network. The agent takes a state and produces an action.
		self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
		hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
		self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
		self.chosen_action = tf.argmax(self.output,1)

		#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
		#to compute the loss, and use it to update the network.
		self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
		self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
		
		self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
		self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

		self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
		
		tvars = tf.trainable_variables()
		self.gradient_holders = []
		for idx,var in enumerate(tvars):
			placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
			self.gradient_holders.append(placeholder)
		
		self.gradients = tf.gradients(self.loss,tvars)
		
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
	sess.run(init)
	i = 0
	total_reward = []
	total_lenght = []
		
	gradBuffer = sess.run(tf.trainable_variables())
	for ix,grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0
		
	while i < total_episodes:
		s = env.reset()
		running_reward = 0
		ep_history = []
		for j in range(max_ep):
			#Probabilistically pick an action given our network outputs.
			a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
			a = np.random.choice(a_dist[0],p=a_dist[0])
			a = np.argmax(a_dist == a)

			s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.
			ep_history.append([s,a,r,s1])
			s = s1
			running_reward += r
			if d == True:
				#Update the network.
				ep_history = np.array(ep_history)
				ep_history[:,2] = discount_rewards(ep_history[:,2])
				feed_dict={myAgent.reward_holder:ep_history[:,2],
						myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
				grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
				for idx,grad in enumerate(grads):
					gradBuffer[idx] += grad

				if i % update_frequency == 0 and i != 0:
					feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
					_ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
					for ix,grad in enumerate(gradBuffer):
						gradBuffer[ix] = grad * 0
				
				total_reward.append(running_reward)
				total_lenght.append(j)
				break

		
			#Update our running tally of scores.
		if i % 100 == 0:
			print(np.mean(total_reward[-100:]))
		i += 1