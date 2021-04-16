import gym
import numpy as np
import matplotlib.pyplot as plt
from init import *

del sum, max

env = gym.make('CartPole-v1')

class Agent(Model):
    def __init__(self, dim_out):
        self.apply = Compose(
            Affine(24), dropout(0.4), ReLU,
            Affine(4), tanh,
            Affine(dim_out), softmax
        )
        
    def replay(self, buffer):
        for memo in buffer:
            obs = np.array(memo['obs'])
            loss = loss_func(memo, self(obs))
            optim(loss.backward())

class Memory(dict):
    replay_buffer_size = 0
    
    def __init__(self):
        self.buffer = []
        self.clear()
        
    def clear(self):
        if self:  # add memory to replay buffer
            self.buffer.append(self.copy())
            self.buffer.sort(key=lambda m: -len(m['rwd']))
            if len(self.buffer) > self.replay_buffer_size:
                self.buffer.pop()
        for key in ['obs', 'out', 'act', 'rwd']:
            self[key] = []
        
    def add(self, obs, out, act, rwd):
        for k, v in vars().items():
            if k != 'self': self[k].append(v)
            
    
def episode_gif(states, gifname=None):
    env.reset()
    frames = []
    for state in states:
        frames.append(env.render(mode='rgb_array'))
        env.env.state = state
    if gifname is not None:
        makegif(frames, gifname)


try:
    agent = Model.load('cartpole-agent')
    print('model loaded')
except FileNotFoundError:
    agent = Agent(2)

optim = Adam(lr=5e-3, reg='l2', lamb=0.01)


def loss_func(memory, outputs=None):
    if outputs is None: outputs = memory['out']
    actions, rewards = memory['act'], memory['rwd']
    return -sum(log(max(1e-8, ps[a])) * r for ps, a, r in
                zip(outputs, actions, discount_rewards(rewards))) / len(rewards)
    
def discount_rewards(rewards, discount=0.97):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(len(rewards))):
        R = discount * R + rewards[t]
        discounted_rewards[t] = R
    return standardize(discounted_rewards)

def smooth(records, k=10):
    smoothed_records = np.zeros_like(records)
    for i in range(2, len(records)):
        i0 = max(0, i - k)
        smoothed_records[i] = np.mean(records[i0:i])
    return smoothed_records


memory = Memory()
eps_rewards = []

fig, ax = plt.subplots()
fig.show()

for eps in (pb := pbar(range(1000), unit='eps')):
    obs = env.reset()
    obs_record = []
    done = False
    
    while not done:
        probabs = agent(old_obs := obs)[0]
        action = np.argmax(np.random.multinomial(1, probabs))
        obs, reward, done, info = env.step(action)
        obs_record.append(obs)
        memory.add(old_obs, probabs, action, reward)

    loss = loss_func(memory)
    params = loss.backward()
    optim(params)
    agent.replay(memory.buffer)
    
    eps_reward = sum(memory['rwd'])
    eps_rewards.append(eps_reward)
    pb.set_postfix(reward=eps_reward)

    if eps % 5 == 0:
        ax.cla()
        ax.plot(np.arange(eps+1), smooth(eps_rewards))
        plt.pause(0.01)
    
    if eps_reward >= 500:
        agent.save('cartpole-agent')
        # episode_gif(obs_record)#, f'cartpole:{eps_reward}.gif')
        
    env.close()
    memory.clear()
    