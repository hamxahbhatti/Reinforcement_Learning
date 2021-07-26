# Implementing A Shallow Q-Network Using Pytorch

import torch
class SLP(torch.nn.Module):
    "A Single Layer Preceptron class to approximate functions"
    def __init__(self,input_shape,output_shape,device = torch.device('cpu')):
        super(SLP,self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.linear1 = torch.nn.Linear(self.input_shape,self.hidden_shape)
        self.out = torch.nn.Linear(self.hidden_shape,output_shape)

    def forwarf(self,x):
        x = torch.from_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.out(x)
        return x

MAX_NUM_EPISODES = 100000
MAX_STEPS_PER_EPISODE = 300

class Shallow_Q_learner(object):
    def __init__(self,state_shape,action_shape,learning_rate=0.005 , gamma = 0.98):
        slef.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.learning_rate = learning_rate
        # Self.Q is  the Action Value function. This agent represents Q using a Neural Network
        self.Q = SLP(state_shape,action_shape)
        self.Q_optimizer  = torch.optim.Adam(self.Q.parameters(),lr=1e-3)
        # sel


EPSILON_MIN =0.005
max_num_steps = 