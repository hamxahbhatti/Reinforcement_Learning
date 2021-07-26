# %%
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from keras.layers import Dense, Conv1D, Flatten
import keras
import numpy as np
from skimage.color import rgb2gray
from collections import deque
import random
import matplotlib.pyplot as plt
# %%

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
env.action_space.n
# input_shape = env.observation_space.shape
# input_shape
# rgb2gray(np.array(input_shape))
input_shape=rgb2gray(env.reset()).shape
# plt.imshow(rgb2gray(env.reset()))
# %%
model = keras.models.Sequential()
model.add(Conv1D(filters=32, kernel_size=4, strides=2,
                 activation='relu', input_shape=input_shape))
model.add(Conv1D(filters=64, kernel_size=3, strides=1,
                 activation='relu'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=env.action_space.n))
model.compile(loss='mse', optimizer='adam')
model.summary()
# %%


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)



# %%
# def get_samples(data):
#     for i in range(10):
#         yield random.sample(data.buffer,500)    


# %%
data = Memory(max_size=8000)
done = True
for step in range(5000):
    if done:
        state = env.reset()
        state = state/255.0
        state = rgb2gray(state)
    epsilon = random.uniform(0, 1)
    state = np.expand_dims(state, axis=0)
    if epsilon < 0.5:
        Q = model.predict(state)
        # print(Q)
        action = np.argmax(Q)
        print("NN Action :", action)
    else:
        action = env.action_space.sample()
        # print("Random : ", action)
    new_state, reward, done, info = env.step(action)
    new_state = new_state/255.0
    data.add((state, action, reward, rgb2gray(new_state), done))
    env.render()
    state = rgb2gray(new_state)

env.close()
# %% Learning from the observations
gamma = 0.95
#samples =get_samples(data)
samples = random.sample(data.buffer,5000)
inputs_shape = (50,) + state.shape
inputs = np.zeros(inputs_shape)
inputs.shape
targets = np.zeros((50, env.action_space.n))
# %%
# for i in range(5000):
#     state = samples[i][0]
#     action = samples[i][1]
#     reward = samples[i][2]
#     new_state = samples[i][3]
#     done = samples[i][4]

#     inputs[i:i+1] = np.expand_dims(state, axis=0)
#     targets[i] = model.predict(state)
#     Q_sa = model.predict(np.expand_dims(new_state, axis=0))

#     if done:
#         targets[i, action] = reward
#     else:
#         targets[i, action] = reward + gamma * np.argmax(Q_sa)
#     if  i % 500 == 0:
#         samples =get_samples(data)
#     model.train_on_batch(inputs, targets)
#     if  i  % 1000 == 0:
#         model.save(f'model{i}.h5')

# %% playing the Game
done = False
state = rgb2gray(env.reset())
total_reward = 0
while done:
    env.render()
    action = model.predict(np.expand_dims(state, axis=0))
    action = np.argmax(action)
    new_state, reward, done, info = env.step(action)
    state = new_state / 255.0
    total_reward += reward
print('Total Reward :  ', total_reward)
