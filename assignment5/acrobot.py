from collections import defaultdict
import gymnasium as gym
import matplotlib.pyplot as plt
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
        self.t.set_weights(self.q.get_weights())
        self.replay = []
        self.replay_limit = 50000
        self.epsilon = 1
        self.decay = .0001
        self.eps_min = .01
        self.gamma = .618
        self.q.compile(loss=tf.keras.losses.Huber(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=.001))
        self.q.summary()

    def make_model(self, state_shape):
        model = tf.keras.models.Sequential([
                   tf.keras.layers.Input(state_shape),
                   tf.keras.layers.Dense(24, activation="relu", kernel_initializer='he_normal'),
                   tf.keras.layers.Dense(12, activation="relu", kernel_initializer='he_normal'),
                   tf.keras.layers.Dense(self.action_space.n, activation="linear", kernel_initializer='he_normal')
                ])
        return model

    def choose_action(self, state, allow_random=False, update_eps=True):
        if allow_random and np.random.random() < self.epsilon: action = np.random.randint(self.action_space.n)
        else: action = int(np.argmax(self.q.predict(state.reshape((1, state.shape[0])), verbose=0)))
        if update_eps and self.epsilon > self.eps_min: self.epsilon -= self.decay
        return action
    
    def add_replay(self, state, action, reward, f_state, term):
        self.replay.append([state, action, reward, f_state, term])
        if len(self.replay) > self.replay_limit: self.replay.pop(0)

    def train_q(self, batch_size=20):
        states = []
        targets = []
        linear_space = np.linspace(0, len(self.replay)-1, num=len(self.replay), dtype=int)
        for _ in range(batch_size):
            state, action, reward, f_state, term = self.replay[np.random.choice(linear_space)]
            states.append(state)
            tmp = self.t.predict(f_state.reshape((1,f_state.shape[0])), verbose=0)[0]
            tmp2 = np.zeros(tmp.shape)
            if term: tmp2[action] = reward
            else: tmp2[action] = reward + self.gamma * max(tmp)
            targets.append(tmp2)
        states = np.array(states)
        targets = np.array(targets)
        self.q.fit(states, targets, verbose=0)

    def update_target(self):
        self.t.set_weights(self.q.get_weights())


e = gym.make('Acrobot-v1', render_mode="human")
agent = DeepQAgent(e.action_space, e.observation_space.shape)
obs, info = e.reset()
for _ in range(1):
    action = agent.choose_action(obs)
    obs, reward, term, trunc, info  = e.step(action)
    if term: e.reset()
obs, info = e.reset()
e.close()
e = gym.make('Acrobot-v1')
state, info = e.reset()
batch_size = 128
for _ in range(batch_size):
    action = agent.choose_action(state, allow_random=True, update_eps=False)
    f_state, reward, term, trunc, info = e.step(action)
    agent.add_replay(state, action, reward, f_state, term)
    state = f_state
    if term: state, info = e.reset()
obs, info = e.reset()
z_print("Training...")
reward_list  = []
avg_list = []
steps_to_train = 500
steps_to_update_target = 1000
z_print(agent.q.get_weights())
reward_sum = 0
x = 0
start = time.time()
for ep in range(1000):
    print("Episode", ep+1, '-', x, '-', agent.epsilon,'-', reward_sum, end="")
    if len(reward_list): print(" -", sum(reward_list) / len(reward_list))
    else: print()
    term = False
    reward_sum = 0
    state, info = e.reset()
    while not term:
        x += 1
        action = agent.choose_action(state, allow_random=True)
        f_state, reward, term, trunc, info = e.step(action)
        agent.add_replay(state, action, reward, f_state, term)
        state = f_state
        reward_sum += reward
        steps_to_train -= 1
        steps_to_update_target -= 1
        if not steps_to_train:
            agent.train_q(batch_size=batch_size)
            steps_to_train = 150
        if not steps_to_update_target:
            steps_to_update_target = 500
            agent.update_target()
    #agent.train_q(batch_size=batch_size)
    reward_list.append(reward_sum)
    avg_list.append(sum(reward_list) /  len(reward_list))
e.close()
z_print(agent.q.get_weights())
z_print("Training Complete in "+str(time.time() - start)+" seconds")
plt.subplot(2,1,1)
plt.plot(np.linspace(1, len(reward_list), num=len(reward_list)), reward_list)
plt.title("Reward Per Episode")
plt.subplot(2,1,2)
plt.plot(np.linspace(1, len(reward_list), num=len(reward_list)), avg_list)
plt.title("Average Reward per Episode")
plt.subplots_adjust(hspace=.5)
plt.savefig('deepq.png')
plt.show()
e = gym.make('Acrobot-v1', render_mode="human")
obs, info = e.reset()
for _ in range(1000):
    action = agent.choose_action(obs)
    obs, reward, term, trunc, info  = e.step(action)
    if term: e.reset()
e.close()
