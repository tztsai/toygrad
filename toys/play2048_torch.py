from init import *
import torch as pt
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from gym2048 import Game2048Env
from tqdm import trange
from msvcrt import getwch
import matplotlib

del sum, max

env = Game2048Env()
play_mode = 'agent'

EPISODES = 10000
SYNC_INTERVAL = 50


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.forward = fn
        
        
class DQNAgent(nn.Sequential):
    dim_out = env.action_space.n
    
    gamma = 0.999
    eps_start = 1.
    eps_end = 0.001
    eps_decay = 30000
    replay_batch_size = 300
    
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
            Lambda(DQNAgent.preprocess),
            nn.Linear(16, 128), nn.LayerNorm(128),
            nn.LeakyReLU(), nn.Dropout(),
            nn.Linear(128, 64), nn.LayerNorm(64),
            nn.LeakyReLU(), #nn.Dropout(),
            nn.Linear(64, self.dim_out),
            nn.Softmax(-1)
        )
        
        self.eps = self.eps_start
        self.steps = 0
        self.memory = self.Memory()
        self.optimizer = Adam(self.parameters())
        
    @classmethod
    def preprocess(cls, board):
        return pt.from_numpy(np.log2(board + 1.).reshape(-1, env.h * env.w)).to(pt.float)
    
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
        nextQ = pt.full(outputQ.shape, -10.)
        nonfinal_next_states = np.array([s for s in b_s1 if s is not None])
        nextQ[mask2] = past_agent(nonfinal_next_states).max(dim=1)[0].detach()
        expectedQ = (nextQ * self.gamma) + b_r

        loss = F.smooth_l1_loss(outputQ, expectedQ)
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1., 1.)
        self.optimizer.step()
        self.optimizer.zero_grad()


try:
    agent = pt.load('2048-agent')
except:
    agent = DQNAgent()
past_agent = DQNAgent()
past_agent.sync_with(agent)

    
def episode_gif(states, gifname=None):
    env.reset()
    frames = []
    for state in states:
        if state is None: break
        env.board = state
        frames.append(env.render(mode='rgb_array'))
    if gifname is not None:
        makegif(frames, gifname)

def smooth(records, k=100):
    smoothed_records = np.zeros_like(records)
    for i in range(2, len(records)):
        i0 = max(0, i - 10)
        smoothed_records[i] = np.mean(records[i0:i])
    return smoothed_records


scores = []
interactive_score_plot = 1

if interactive_score_plot:
    fig, ax = plt.subplots()
    fig.show()

for eps in (pb := trange(EPISODES, unit='eps')):
    obs = env.reset()
    obs_record = []  # for animating the episode afterwards
    done = False
    
    while not done:
        if play_mode == 'agent':
            action = agent.select_action(old_obs := obs)
            obs, reward, done, info = env.step(action)
            if done: obs = None
            agent.memory.push(old_obs, action, reward, obs)
            obs_record.append(obs)
        else:
            env.render()
            action = 'kljh'.index(getwch())
            obs, reward, done, info = env.step(action)

    if (eps + 1) % SYNC_INTERVAL == 0:
        past_agent.sync_with(agent)
    
    if play_mode == 'agent':
        agent.replay()
    
    scores.append(env.score)
    pb.set_postfix(score=env.score, eps=agent.eps)

    if interactive_score_plot and eps % 10 == 0:
        ax.cla()
        ax.plot(np.arange(eps+1), smooth(scores))
        fig.canvas.draw()
        fig.canvas.start_event_loop(0.001)
    
    if env.score > 2000:
        pt.save(agent, '2048-agent')
        if env.score > 3000:
            episode_gif(obs_record)#, f'cartpole:{eps_reward}.gif')
        
    env.close()
    
if not interactive_score_plot:
    plt.plot(smooth(scores))
    plt.show()
