

import gym

env = gym.make("CartPole-v0")
env.reset()

counter=0
while True:
	env.render()
	action = env.action_space.sample()
	next_state, reward, done, info = env.step(action)
	counter += 1
	if done and counter > 20:
		env.reset()
		counter = 0

