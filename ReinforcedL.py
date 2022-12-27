import gym
import numpy as np
import time
import os
import matplotlib as plt
from matplotlib import pyplot

''' Reinforced learning with the gym library '''

env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()
#env.render()

done = False
best_reward = -1
episodes = 25000
step_limit = 100

#Parameters for Q-table
lr = 0.5 #alpha
gamma = 0.9
max_eps = 1.0
min_eps = 0.01
decay_rate = 0.001

rewards = []
eps = max_eps #Epsilon starting point

def animation(q_table, state=env.reset()):
    for step in range(step_limit):
        os.system('cls')
        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)
        state = new_state
        env.render()
        time.sleep(.5)

        if done:
            break

#Why not dense horizon rewards? :/

q_table = np.zeros((env.observation_space.n,env.action_space.n)) # Qtable empty at first
#Attempt iteration
for episode in range(episodes):
    state = env.reset()
    step = 0
    done = False
    reward_tot = 0
    
    frames = []

    for step in range(step_limit):
        rnd_for_egreedy = np.random.rand(1,1)

        if rnd_for_egreedy[0][0] > eps: #Introducing decreasing chance for random action
            action = np.argmax(q_table[state,:])

        else:
            action = env.action_space.sample() #Random action

        new_state, reward, done, info = env.step(action)

        #updating Q value
        #q_table[state, action] = reward + gamma * np.argmax(q_table[new_state,:]) #A and B sections
        q_table[state, action] = q_table[state, action] + lr*(reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]) #C section

        '''
        frames.append({
            'frame' : env.render(mode='ansi'),
            'state' : state,
            'action' : action,
            'reward' : reward
        })
        '''

        reward_tot += reward
        state = new_state

        if done:
            rewards.append(reward_tot)
            if episode % 10 == 0:
                print(f'Episode {episode+1}: Reward: {reward} with {step} steps')
                #time.sleep(0.5)

            eps = min_eps + (max_eps-min_eps)*np.exp(-decay_rate*episode) #Updating the epsilon value
            break
'''
    if reward_tot > best_reward:
        animation(frames, episode)

        print(f"new personal best of {reward_tot} points!")
        best_reward = reward_tot
        best_q_table = q_table
'''

print("0/left 1/down 2/right 3/up")
print("== Optimal Q-table ==")
print(q_table)

#Utilizing a reward plot
moving_avg_reward = []
window = 1000
for i in range(window, episodes):
    moving_avg_reward.append(100*sum(rewards[i-window:i])/window)

fig, axes = pyplot.subplots(figsize=(8, 8))
pyplot.plot(range(window, episodes), moving_avg_reward)
axes.set(xlabel='Episode Idx', ylabel='Success Rate', title='Expected reward with a moving average with window size = {}'.format(window))

#animation(q_table)
pyplot.show()
