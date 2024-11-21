from collections import defaultdict
import gymnasium as gym
import numpy as np
import tensorflow as tf
import time

def z_print(n):
    print("=============\n")
    print(n)
    print("\n=============")


class DeepQAgent:
    def __init__(self, action_space, state_shape):
        print(state_shape)
        self.action_space = action_space
        self.q = self.make_model(state_shape)
        self.t = self.make_model(state_shape)
        #self.t.set_weights(self.q.get_weights())
        self.replay = []
        self.replay_limit = 100
        self.epsilon = 1
        self.decay = .001
        self.eps_min = .1
        self.gamma = .001
        self.q.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=tf.keras.optimizers.SGD(learning_rate=.001))
        self.q.summary()

    def make_model(self, state_shape):
        model = tf.keras.models.Sequential([
                   tf.keras.layers.Input(state_shape),
                   tf.keras.layers.Dense(128,activation="tanh"),
                   tf.keras.layers.Dense(64,activation="tanh"),
                   tf.keras.layers.Dense(self.action_space.n, activation="linear")
                ])
        return model

    def choose_action(self, state):
        if np.random.random() < self.epsilon: action = np.random.randint(self.action_space.n)
        else: action = int(np.argmax(self.q.predict(state.reshape((1, state.shape[0])), verbose=0)))
        return action
    
    def add_replay(self, state, action, reward, f_state):
        self.replay.append([state, action, reward, f_state])
        if len(self.replay) > self.replay_limit: self.replay.pop(0)

    def train_q(self, batch_size=20):
        states = []
        targets = []
        linear_space = np.linspace(0, len(self.replay)-1, num=len(self.replay), dtype=int)
        for _ in range(batch_size):
            state, action, reward, f_state = self.replay[np.random.choice(linear_space)]
            states.append(state)
            targets.append(self.t.predict(f_state.reshape((1,f_state.shape[0])), verbose=0)[0])
            targets[-1] *= self.gamma
            targets[-1][action] += reward
            if self.epsilon > self.eps_min: self.epsilon -= self.decay
        states = np.array(states)
        targets = np.array(targets)
        self.q.fit(states, targets, verbose=0)
        self.update_target()

    def update_target(self):
        self.t.set_weights(self.q.get_weights())


e = gym.make('CartPole-v1')
state, info = e.reset()
agent = DeepQAgent(e.action_space, e.observation_space.shape)
z_print("Training...")
for i in range(10):
    for _ in range(100):
        action = agent.choose_action(state)
        f_state, reward, term, trunc, info = e.step(action)
        agent.add_replay(state, action, reward, f_state)
        state = f_state
        if term: state, info = e.reset()
    agent.train_q()
e.close()
z_print("Training Complete!")
e = gym.make('CartPole-v1', render_mode="human")
obs, info = e.reset()
for _ in range(1000):
    action = agent.choose_action(obs)
    obs, reward, term, trunc, info  = e.step(action)
    if term: e.reset()
e.close()
