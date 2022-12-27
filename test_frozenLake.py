import gym
import time
import numpy as np
import matplotlib.pyplot as plt

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()
env.render()

done = False
env.reset()

def eval_policy(qtable_, num_of_episodes_, max_steps_):
    rewards = []
    for episode in range(num_of_episodes_):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        for step in range(max_steps_):
            action = np.argmax(qtable_[state,:])
            new_state, reward, done, info = env.step(action)
            total_rewards += reward
            if done:
                rewards.append(total_rewards)
                break

        state = new_state
    
    env.close()
    avg_reward = sum(rewards)/num_of_episodes_
    return avg_reward

print(f'Average reward: {eval_policy(np.random.rand(env.observation_space.n,env.action_space.n),1000,100)}')

reward_best = -1000
total_episodes = 1000
max_steps = 100
qtable_best = []

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    qtable = np.random.rand(env.observation_space.n,env.action_space.n) # Random Q-table
    reward_tot = 0
    for step in range(max_steps):
        action = np.argmax(qtable[state,:])
        new_state, reward, done, info = env.step(action)
        reward_tot += reward
        state = new_state
    if done == True:
        break
    
    if reward_tot > reward_best:
        reward_best = reward_tot
        qtable_best = qtable
        print(f'Better found - reward: {reward_best}')

    if episode % 100 == 0:
        print(f'Best reward after episode {episode+1} is {eval_policy(qtable_best)}')

print (f'Tot reward of the found policy: {eval_policy(qtable_best,1000,100)}')
print(qtable_best)