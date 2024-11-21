from collections import defaultdict
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time

e = gym.make('CartPole-v1')
q = defaultdict(lambda: np.zeros(e.action_space.n))
#print(np.random.uniform(low = 0, high = 1, size=([1,2,3,4,5])))
#exit()
obs, info = e.reset()
eps = 1
decay = .001
t = tuple
def t(n):
    d = np.sign(n[2:])
    return tuple(np.round(n[3:], 1))
print("training...")
episode_reward = 0
reward_list = []
for _ in range(1000):
    if np.random.random() < eps: action = e.action_space.sample()
    else: action = int(np.argmax(q[t(obs)]))
    f_obs, reward, term, trunc, info = e.step(action)
    f_q = (not term) * np.max(q[t(f_obs)])
    t_d = (reward + .2 * f_q - q[t(obs)][action])
    q[t(obs)][action] += .001 * t_d
    obs = f_obs
    episode_reward += reward
    if term:
        obs, info = e.reset()
        reward_list.append(episode_reward)
        episode_reward = 0
    if eps > .1: eps -= decay
print("training complete")
e.close()
plt.plot(np.linspace(1, len(reward_list), num=len(reward_list)), reward_list)
plt.show()
e = gym.make('CartPole-v1', render_mode="human")
obs, info = e.reset()
for _ in range(1000):
    action = int(np.argmax(q[t(obs)]))
    obs, reward, term, trunc, info  = e.step(action)
    if term: e.reset()
e.close()
