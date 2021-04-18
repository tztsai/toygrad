import torch as pt
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

import random
import itertools
from tqdm import trange
from msvcrt import getwch

import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

env = gym.make('Tennis-ram-v0')
play_mode = 'agent'

EPISODES = 20
SYNC_INTERVAL = 2


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.forward = fn


class DQNAgent(nn.Sequential):
    dim_in = 128
    dim_out = env.action_space.n
    
    gamma = 0.99
    eps_start = 1.
    eps_end = 0.001
    eps_decay = 10000
    replay_batch_size = 512
    
    class Memory(list):
        def __init__(self, capacity=10000):
            self.capacity = capacity
            self.p = 0
        
        def push(self, s0, a, r, s1):
            if len(self) < self.capacity:
                self.append(None)
            self[self.p] = s0, a, r, s1
            self.p = (self.p + 1) % self.capacity
            
        def sample(self, size):
            return random.sample(self, size)
    
    def __init__(self):
        super().__init__(
            Lambda(self.preprocess),
            nn.Linear(self.dim_in, 64), nn.LayerNorm(64),
            nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(64, 64), nn.LayerNorm(64),
            nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(64, self.dim_out)
        )
        
        self.eps = self.eps_start
        self.steps = 0
        self.memory = self.Memory()
        
    @classmethod
    def preprocess(cls, input):
        return pt.from_numpy(input.astype(np.float32))
        
    def select_action(self, state):
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps / self.eps_decay)
        self.steps += 1
        if random.random() > self.eps:
            self.eval()
            return np.argmax(self(state).detach())
        else:
            return random.randrange(self.dim_out)
        
    def sync_with(self, other):
        self.load_state_dict(other.state_dict())

    def replay(self):
        if len(self.memory) < self.replay_batch_size: return
        
        self.train()
        transitions = self.memory.sample(self.replay_batch_size)
        batch = np.array(transitions, dtype=object).T
        b_s0, b_a, b_r, b_s1 = batch
        b_s0 = np.stack(b_s0, axis=0)
        b_a, b_r = b_a.astype(np.int), b_r.astype(np.float32)
        
        mask1 = onehot(b_a, k=self.dim_out).astype(bool)
        mask2 = np.array([s is not None for s in b_s1])
        
        outputQ = self(b_s0)[mask1]  # only keep selected actions
        nextQ = pt.zeros(*outputQ.shape)
        nonfinal_next_states = np.array([s for s in b_s1 if s is not None])
        # mask3 = onehot(self(nonfinal_next_states).detach().max(dim=1)[1],
        #                k=self.dim_out).astype(bool)
        # nextQ[mask2] = past_agent(nonfinal_next_states).detach()[mask3]
        nextQ[mask2] = past_agent(nonfinal_next_states).max(dim=1)[0].detach()
        expectedQ = (nextQ * self.gamma) + b_r

        loss = F.smooth_l1_loss(outputQ, expectedQ)
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1., 1.)
        optimizer.step()
        optimizer.zero_grad()


try:
    # assert 0
    agent = pt.load('tennis-agent-torch')
    print('model loaded')
except:
    agent = DQNAgent()
agent.replay_batch_size = 512
past_agent = DQNAgent()
past_agent.sync_with(agent)

optimizer = Adam(agent.parameters())


def episode_gif(states, gifname=None):
    env.reset()
    frames = []
    for state in states:
        if state is None: break
        env.state = state
        frames.append(env.render(mode='rgb_array'))
    if interactive_plot:
        ax.axis('off')
        # plt.ion()
        im = ax.imshow(frames[0])
        for frame in frames:
            im.set_data(frame)
            fig.canvas.draw()
            fig.canvas.start_event_loop(0.01)
            # plt.pause(0.01)
        ax.axis('on')
        # plt.ioff()
    # if gifname is not None:
    #     makegif(frames, gifname)
    
def makegif(frames, filename='__tmp__.gif', blit=True, fps=60, **kwds):
    fig, ax = plt.subplots()
    ax.axis('off')
    def step(frame): return ax.imshow(frame)
    gif = matplotlib.animation.FuncAnimation(
        fig, step, frames=frames, **kwds)
    gif.save(filename=filename, fps=fps)

def onehot(x, k, *, cold=0, hot=1):
    x = np.asarray(x)
    m = np.full((*x.shape, k), cold, dtype=np.int)
    for idx in itertools.product(*map(range, x.shape)):
        m[idx][x[idx]] = hot
    return m

def smooth(records, k=100):
    smoothed_records = np.zeros_like(records)
    for i in range(2, len(records)):
        i0 = max(0, i - k)
        smoothed_records[i] = np.mean(records[i0:i])
    return smoothed_records


scores = []
interactive_plot = 1
animate_episodes = 1
frames = []  # for animating the episode afterwards

if interactive_plot:
    fig, ax = plt.subplots()
    fig.show()

for eps in (pb := trange(EPISODES, unit='eps')):
    obs = env.reset()
    frames.clear()
    done = False
    score = 0
    
    while not done:
        if play_mode == 'agent':
            action = agent.select_action(old_obs := obs)
            obs, reward, done, info = env.step(action)
            if done: obs = None
            score += reward
            agent.memory.push(old_obs, action, reward, obs)
            if animate_episodes:
                # frames.append(env.render('rgb_array'))
                env.render()
        else:
            env.render()
            action = '1234567890asdfghjk'.index(getwch())
            obs, reward, done, info = env.step(action)

    if (eps + 1) % SYNC_INTERVAL == 0:
        past_agent.sync_with(agent)
    
    if play_mode == 'agent':
        agent.replay()
    
    scores.append(score)
    pb.set_postfix(score=score, avg_score=np.mean(scores[-100:]), eps=agent.eps)

    if interactive_plot:
        ax.cla()
        ax.plot(np.arange(eps+1), smooth(scores))
        fig.canvas.draw()
        fig.canvas.start_event_loop(0.001)
    
    if eps % 5 == 0:
        pt.save(agent, 'tennis-agent-torch')
    if score > -0.1 and animate_episodes:
        # episode_gif(frames, f'tennis:{score}.gif')
        pass
        
    env.close()
    
if not interactive_plot:
    plt.plot(smooth(scores))
    plt.show()
