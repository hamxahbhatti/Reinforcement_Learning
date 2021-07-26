import gym
env = gym.make("Qbert-v0")
Max_num_episodes= 10
Max_steps_per_episode  = 500
for episode in  range(Max_num_episodes):
    obs=env.reset()
    for steps in range(Max_steps_per_episode):
        env.render()
        action = env.action_space.sample() # Get the Random Action .
        next_state,reward,done,info = env.step(action)
        obs = next_state

        if done:
            print("\n Episode {} ended in {} steps.".format(episode,steps +1))
            break