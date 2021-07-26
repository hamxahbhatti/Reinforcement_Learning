import gym

env= gym.make('MountainCar-v0')
Max_num_episode = 5000
for episodes in range(Max_num_episode):
        done = False
        obs = env.reset()
        total_reward = 0.0
        step = 0
        while not done:
            env.render()
            action = env.action_space.sample()
            next_state,reward,done,_ = env.step(action)
            total_reward += reward
            step += 1
            obs = next_state
        print('\n Episode {} ended in {} .Total reward {}'.format(episodes,step,total_reward))
env.close ()


