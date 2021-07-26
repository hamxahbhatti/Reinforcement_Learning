# Creating  Class for 
import keras
from keras.layers import  Dense ,Conv2D, Flatten, Dropout
import numpy as np
from collections import deque
import random

class LinearDecaySchedule(object):
    def __init__(self,initial_value,final_value,max_steps):
        assert initial_value > final_value, "initial_value should be > final_value"
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_factor = (initial_value - final_value) / max_steps


    def __call__(self,step_num):
        curent_value = self.initial_value - self.decay_factor *  step_num
        if current_value <  self.final_value:
            current_value = self.final_value
        return current_value



# NOTE : I have to improve this class
class ExperienceReplay(object):

      def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

class SLP(object):
    def __init__(self,input_shape,output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = keras.models.Sequential()
        # self.model.add(Dense(50,activation='relu',input_shape=input_shape[1]))
        # self.model.add(Dense(self.output_shape[1]))
        # self.model.compile(optimizer='adam',loss=)

    def __call__(self):
        self.model.add(Dense(50,activation='relu',input_shape=input_shape[1]))
        self.model.add(Dense(self.output_shape[1]))

class CNN(object):
    def __init__(self,input_shape,output_shape):
        self.input_shape = input_shape
        self,output_shape = output_shape
        self.model = keras.models.Sequential()
        #  self.model.add(Conv2D(input_shape=input_shape[1],32,kernel_size=8,stride=4,padding=0,activation='relu'))
        # self.model.add(Conv2D(units =32,filters=65,kernel_size=3,stride=2,padding=0,activation='relu'))
        # self.model.add(Conv2D(units=64,filters=64,kernel_size=3,stride=1,padding=0))
        # self.model.add(Flatten())
        # self.model.add(Dense(units=self.output_shape[1]))

    def __call__(self):
        self.model.add(Conv2D(input_shape=input_shape[1],32,kernel_size=8,stride=4,padding=0,activation='relu'))
        self.model.add(Conv2D(units =32,filters=65,kernel_size=3,stride=2,padding=0,activation='relu'))
        self.model.add(Conv2D(units=64,filters=64,kernel_size=3,stride=1,padding=0))
        self.model.add(Flatten())
        self.model.add(Dense(units=self.output_shape[1]))

class Q_learner(object):
    def __init__(self,state,shape,action_shape,gamma,learning_rate):
        self.state_shape = state_shape
        self.action_shape  = action_shape
        self.learning_rate = learning_rate
        self.gamma = gamma

        if len(self.state_shape)== 1 : # Single Dimension observation
            self.DQN = SLP
        if  len(self.state_shape) == 3 : #  3D image state shape
            self.DQN = CNN
        self.Q = self.DQN(state_shape,action_shape)

        self.policy = self.epsilon_greedy_Q
         # NOTE : i have to implement this
        self.epsilon_max = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = LinearDecaySchedule(initial_value=self.epsilon_max,
                                            final_value=self.epsilon_min,
                                            max_steps=)

    def get_action(self,observation):
        observation = np.array(observation)
        observation = observation / 255.0 # Scaling the image
        if len(observation.shape) == 3 : # Single Image
            observation = np.expand_dims(observation,axis=0)
            # NOTE  : I have to implement  Epsilon Greedy Policy
        return self.policy(observation) 
    
    def epsilon_greedy_Q(self,observation):
        if random.random()  < self.epsilon_decay(self.step_num):
            action = random.choice([i for i in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(observation).numpy()) # NOTE : i may have to fix this
        
        return action


    def learn()

    