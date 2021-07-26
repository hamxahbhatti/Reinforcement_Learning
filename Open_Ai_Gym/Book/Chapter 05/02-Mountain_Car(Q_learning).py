# Implementing the Q learner class
import numpy as np
import gym
Max_num_episodes = 15000
Steps_per_episode = 200
Max_num_steps = Max_num_episodes * Steps_per_episode
Epsilon_min = 0.005
Epsilon_decacy = 500 * Epsilon_min / Max_num_steps
Alpha = 0.05
Gamma = 0.98
Num_discrete_bins = 30

class Q_learner(object):
    def __init__(self,env):
        self.obs_shape = env.observation_space.shape # Getting the Evn State Shapes
        self.obs_high = env.observation_space.high # Getting the highest State space
        self.obs_low = env.observation_space.low # Getting the lowest State space
        self.obs_bins = Num_discrete_bins  # Num of bins to Discretize each observation dim
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n  # Getting Env Action space shape
        # Creating multidimensional Array (aka matrix) to represent the Q values
        self.Q = np.zeros((self.obs_bins +1 ,self.obs_bins +1 ,self.action_shape)) # (51,51,3)
        self.alpha = Alpha
        self.gamma = Gamma
        self.epsilon = 1.0

    def discretize(self,obs):
        return tuple(((obs - self.obs_low)  / self.bin_width).astype(int))

    def get_action(self,obs):
        discretized_obs = self.discretize(obs)
        # Epsilon Greedy action Selection 
        if self.epsilon > Epsilon_min:
            self.epsilon -= Epsilon_decacy
        if np.random.random() > self.epsilon:  # Search for the best action  with higest reward
            return np.argmax(self.Q[discretized_obs])
        else : # Choose the Random Action   
            return  np.random.choice([a for a in range(self.action_shape)])
    
    def  learn(self,obs,action,reward,next_obs):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        td_target = reward + self.gamma * np.argmax(self.Q[discretized_next_obs])
        td_error = td_target - self.Q[discretized_obs][action]
        self.Q[discretized_obs][action] += self.alpha * td_error

def train(agent,env):
    best_reward = - float('inf')
    for episode in range(Max_num_episodes):
        done = False
        obs = env.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs)
            next_obs,reward,done,info= env.step(action)
            agent.learn(obs,action,reward,next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode# : {} Reward {} Best Reward {} Epsilon {}".format(episode,total_reward,best_reward,agent.epsilon))
        # Return  the trained policy 
    return np.argmax(agent.Q,axis=2)


def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    agent = Q_learner(env)
    learned_policy = train(agent,env)
    # Using the Gym monitor wrapper to evalaute the agent and record video
    gym_monitor_path = "./gym_monitor_output"
    env = gym.wrappers.Monitor(env,gym_monitor_path,force=True)
    for _ in  range(1000):
        test(agent,env,learned_policy)
    env.close()