# %% importing the Libraries
import numpy as np
import gym
import random
from keras.layers import Dense, Flatten
from keras.models import Sequential
from collections import deque
# %%Making the evnironment
env = gym.make('MountainCar-v0')
# env.render()
env.action_space.sample()
env.observation_space.shape
env.action_space.n
# %% Creating the Q table

model = Sequential()
model.add(Dense(units=50, input_shape=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# %%
total_episode = 1000        # Total episodes
total_test_episode = 100
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon_greedy = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum explorat ion probability
decay_rate = 0.01
# %% Experience Replay


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)


# %%
memory = Memory(10000)
# %%
observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
state.shape
done = False

for episode in range(5000):
    # Applying the epsilon_greedy startegy
    # env.render()
    exploit_exploration_tradeoff = random.uniform(0, 1)

    if exploit_exploration_tradeoff > 0.5:
        Q = model.predict(state)
        print('Q_value : ', Q)
        action = np.argmax(Q)
        print('NN : ', action)
    else:
        action = env.action_space.sample()
        print('Sample : ', action)

    new_observation, reward, done, info = env.step(action)
    obs = np.expand_dims(new_observation, axis=0)
    new_state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
    memory.add((state, action, reward, new_state, done))
    state = new_state
    if done:
        observation = env.reset()
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs, obs), axis=1)
# %% Learning from the observations
sample = random.sample(memory.buffer, 5000)
state.shape
# creating q_table
inputs_shape = (5000,) + state.shape[1:]
input = np.zeros(inputs_shape)
target = np.zeros(shape=(5000, env.action_space.n))
# %%
for i in range(5000):
    state = sample[i][0]
    action = sample[i][1]
    reward = sample[i][2]
    new_state = sample[i][3]
    done = sample[i][4]

    input[i:i+1] = np.expand_dims(state, axis=0)
    target[i] = model.predict(state)
    Q_sa = model.predict(new_state)

    if done:
        target[i, action] = reward
    else:
        target[i, action] = reward + gamma * np.argmax(Q_sa)

    model.train_on_batch(input, target)
    if  i  % 1000 == 0:
        model.save(f'model{i}.h5')
target
# %% Playing
observation = env.reset()
obs = np.expand_dims(observation, axis=0)
state = np.stack((obs, obs), axis=1)
done = False
tot_reward = 0.0
while not done:
    env.render()                    # Uncomment to see game running
    Q = model.predict(state)
    action = np.argmax(Q)
    observation, reward, done, info = env.step(action)

    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
    tot_reward += reward
print('Game ended! Total reward: {}'.format(reward))
