#https://github.com/keon/deep-q-learning
import gym
import tensorflow as tf
import numpy as np
from tfUtils import *
#print(gym.envs.registry.all()) #available learning enviroments

env = gym.make("CartPole-v0")
env.reset()

memory_size = 1000
memory = []

y = 0.99 #discount rate for future gains - higher makes it favor future more
e = 0.1 #chance of forcing a random action

#=====================================================================
#make the network
input_state = tf.placeholder(tf.float32,shape=[None,4])
l1 = denseUnit(input_state,[4,10])
output_values = denseUnit(l1,[10,2])
output_action = tf.argmax(output_values) #the index of the highest value

wanted_output = tf.placeholder(tf.float32,shape=[None,2])
loss = tf.reduce_sum(tf.square(wanted_output - output_values))

#=====================================================================
#do the learning
sess = tf.Session()
curt_state = env.reset()
step_counter = 0
reward_total = 0

# trainer = tf.train.AdamOptimizer()
trainer = tf.train.GradientDescentOptimizer(0.1)
optimize = trainer.minimize(loss)
sess.run(tf.global_variables_initializer())
while True:
	env.render()
	step_counter += 1
	#Chose an action for the current state
	action,outputs = sess.run([output_action,output_values],feed_dict={input_state:[curt_state]})
	if np.random.rand(1) < e:
		action[0] = env.action_space.sample()

	#see what the action does
	next_state, reward, done, info = env.step(action[0])
	reward_total += reward
	#and what the network wants to do for the new state
	outputs_next = sess.run(output_values,feed_dict={input_state:[next_state]})

	# how much do we want to reinforce the action taken?
	delta = reward + y*np.max(outputs_next)
	# delta = reward_total + y*np.max(outputs_next)
	# delta = y*np.max(outputs_next)
	# delta = outputs[0,action[0]]
	if done:
		delta = reward_total
	# put the amount to reinforce as the only delta of the output to have it train
	target_outputs = outputs
	target_outputs[0,action] = delta

	curt_state = next_state
	if done == True:
		curt_state = env.reset()
		reward_total = 0
		# Reduce chance of random action as we train the model.
		e = 1./((step_counter/50) + 10)
		# e -= 0.001

		#pretty little progres bar of how long it runs before being "done"
		print((" "*int(step_counter/4)) + "#")
		step_counter = 0

	sess.run(optimize,feed_dict={input_state:[curt_state],wanted_output:target_outputs})

	# print(next_state)
	# print(env.action_space)
	# print(reward)



#TODO LIST
#little NN stuff above for DeepQ
#clean room - mostly bags and bottles
#clean room - move food around
#school for loan stuff
#NN stuff again
#laundry






# import gym
# import numpy as np
# import random
# import tensorflow as tf
# import matplotlib.pyplot as plt


# env = gym.make('FrozenLake-v0')

#These lines establish the feed-forward part of the network used to choose actions
# inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
# W = tf.Variable(tf.random_uniform([16,4],0,0.01))
# Qout = tf.matmul(inputs1,W)
# predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
# nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
# loss = tf.reduce_sum(tf.square(nextQ - Qout))
# trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# updateModel = trainer.minimize(loss)

# Set learning parameters
# y = .99
# e = 0.1
# num_episodes = 2000
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for i in range(num_episodes):
    #Reset environment and get first new observation
    # s = env.reset()
    # done = False
    # step = 0
    #The Q-Network
    # while step < 99:
    #     step+=1
        #Choose an action by greedily (with e chance of random action) from the Q-network
        # a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
        # if np.random.rand(1) < e:
        #     a[0] = env.action_space.sample()
        #Get new state and reward from environment
        # s1,r,done,_ = env.step(a[0])
        # env.render()
        #Obtain the Q' values by feeding the new state through our network
        # Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
        #Obtain maxQ' and set our target value for chosen action.
        # maxQ1 = np.max(Q1)
        # targetQ = allQ
        # targetQ[0,a[0]] = r + y*maxQ1
        #Train our network using target and predicted Q values
        # sess.run(updateModel,feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
        # s = s1
        # if done == True:
        #     #Reduce chance of random action as we train the model.
        #     e = 1./((i/50) + 10)
        #     break

