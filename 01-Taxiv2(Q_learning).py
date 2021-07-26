# %% importing the Libraries
import numpy as np
import gym
import random

# %%Making the evnironment
# %%Making the evnironment
env = gym.make('Taxi-v2')
env.render()
# %% Creating the Q table
env.reset()
action_size = env.action_space.n
state_size = env.observation_space.n
# Creating the Q table
q_table = np.zeros((state_size, action_size))
q_table

env.action_space.sample()
# %% creating the hyper parameters
total_episode = 1000        # Total episodes
total_test_episode = 100
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon_greedy = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01

# %% Training the  Agent
for episode in range(total_episode):

    state = env.reset()
    done = False
    step = 0

    for steps in range(max_steps):
        # Applying the epsilon_greedy startegy
        # env.render()
        exploit_exploration_tradeoff = random.uniform(0, 1)

        if exploit_exploration_tradeoff > epsilon_greedy:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        # Applying Bellman equation for q learning
        # Temporal difference
        # Q(s,a) = Q(s,a) +  learning_rate(R(s,a),gamma *max(Q(s',a')) - Q(s,a))
        q_table[state, action] = q_table[state, action] + learning_rate * \
            (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
        state = new_state

        if done:
            break
    episode += 1
    epsilon_greedy = min_epsilon + \
        (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
# %% using the q_table to play Game

env.reset()
rewards = []
for episode in range(total_test_episode):

    state = env.reset()
    done = False
    total_reward = 0
    print('==' * 80)
    print('Episode : ', episode)
    for steps in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])

        new_state, reward, done, info = env.step(action)

        total_reward += reward
        if done:
            rewards.append(total_reward)
            print('Score  : ', total_reward)
            break
        state = new_state


env.close()
print("Reward over time : " + str(sum(rewards)/total_test_episode))
