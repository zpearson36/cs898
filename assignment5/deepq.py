from collections import defaultdict
import gymnasium as gym
import numpy as np
import tensorflow as tf
import time

e = gym.make('CartPole-v1')
#q = defaultdict(lambda: np.zeros(e.action_space.n))
q = tf.keras.models.Sequential([
    tf.keras.layers.Input(e.observation_space.n)
    tf.keras.layers.Dense(24,activation="tanh"),
    tf.keras.layers.Dense(24,activation="tanh"),
    tf.keras.layers.Dense(e.action_space.n, activation="linear")
    ])
q.compile(loss="mse", optimizer=tf.keras.optimizers.Adam())
q.summary()
#print(np.random.uniform(low = 0, high = 1, size=([1,2,3,4,5])))
#exit()
obs, info = e.reset()
eps = 1
decay = .001
t = tuple
def t(n):
    d = np.sign(n[2:])
    return tuple(np.round(n[1:], 1))
print("training...")
for _ in range(1000):
    if np.random.random() < eps: action = e.action_space.sample()
    else: action = int(np.argmax(q[t(obs)]))
    f_obs, reward, term, trunc, info = e.step(action)
    f_q = (not term) * np.max(q[t(f_obs)])
    t_d = (reward + .2 * f_q - q[t(obs)][action])
    q[t(obs)][action] += .001 * t_d
    if term: e.reset()
    if eps > .1: eps -= decay
    obs = f_obs
print("training complete")
e.close()
e = gym.make('CartPole-v1', render_mode="human")
obs, info = e.reset()
for _ in range(1000):
    action = int(np.argmax(q[t(obs)]))
    obs, reward, term, trunc, info  = e.step(action)
    if term: e.reset()
e.close()
