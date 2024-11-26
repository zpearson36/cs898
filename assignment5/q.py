from collections import defaultdict
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time

def t(n):
    return tuple(np.round(n[2:],1))
q = defaultdict(lambda: np.zeros(e.action_space.n))
#e = gym.make('CartPole-v1', render_mode="human")
#obs, info = e.reset()
#for _ in range(200):
#    action = int(np.argmax(q[t(obs)]))
#    obs, reward, term, trunc, info  = e.step(action)
#    if term: e.reset()
#e.close()
#print(np.random.uniform(low = 0, high = 1, size=([1,2,3,4,5])))
#exit()
e = gym.make('CartPole-v1')
obs, info = e.reset()
eps = 1
decay = .001
print("training...")
episode_reward = 0
reward_list = []
avg_list = []
for _ in range(5):
    # print random state tables
    state = np.random.rand(1,2)
    print(state)
    print(q[t(state[0])])
for ep in range(1000):
    #print('Episode', ep+1)
    term = False
    step = 0
    while not term:
        step += 1
        if np.random.random() < eps: action = e.action_space.sample()
        else: action = int(np.argmax(q[t(obs)]))
        f_obs, reward, term, trunc, info = e.step(action)
        f_q = (not term) * np.max(q[t(f_obs)])
        #if term and step < 50: reward = -75
        t_d = reward + .9 * f_q
        q[t(obs)][action] *= .999
        q[t(obs)][action] += .001 * t_d
        obs = f_obs
        episode_reward += reward
    if eps > .1: eps -= decay
    obs, info = e.reset()
    reward_list.append(episode_reward)
    avg_list.append(sum(reward_list) /  len(reward_list))
    episode_reward = 0
print("training complete")
e.close()
print(eps)
for v in range(5):
    # print random state tables
    state = np.random.rand(1,2)
    print(list(q.keys())[v])
    print(q[list(q.keys())[v]])
#plt.subplot(2,1,1)
#plt.plot(np.linspace(1, len(reward_list), num=len(reward_list)), reward_list)
#plt.title("Reward Per Episode")
#plt.subplot(2,1,2)
#plt.plot(np.linspace(1, len(reward_list), num=len(reward_list)), avg_list)
#plt.title("Average Reward per Episode")
#plt.subplots_adjust(hspace=.5)
#plt.show()
#e = gym.make('CartPole-v1', render_mode="human")
#obs, info = e.reset()
#for _ in range(200):
#    action = int(np.argmax(q[t(obs)]))
#    obs, reward, term, trunc, info  = e.step(action)
#    if term: e.reset()
#e.close()
