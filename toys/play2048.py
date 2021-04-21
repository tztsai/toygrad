from msvcrt import getwch
import matplotlib
import matplotlib.pyplot as plt

from .gym2048 import Game2048Env
from toych import *
del sum, max

onehot = utils.onehot

env = Game2048Env()
play_mode = 'agent'

# Param.grad_lim = 1.

EPISODES = 10000
SYNC_INTERVAL = 100

class DQNAgent(Model):
    dim_in = env.h * env.w
    dim_out = env.action_space.n
    
    gamma = 0.99
    eps_start = 1.
    eps_end = 1e-3
    eps_decay = 30000
    replay_batch_size = 256

    optim = Adam(decay='l2')
    
    class Memory(list):
        capacity = 50000
        
        def __init__(self, capacity=capacity):
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
        self.apply = Compose(
            self.preprocess,
            affine(128), normalize(),
            leakyReLU, #dropout,
            affine(64), normalize(),
            leakyReLU,
            affine(self.dim_out)
        )
        
        self.eps = self.eps_start
        self.steps = 0
        self.memory = self.Memory()
        self.past_copy = None

    def preprocess(self, board):
        board = np.log2(board + 1.).astype(np.int).reshape(-1, self.dim_in)
        return np.array([onehot(b, self.dim_in).reshape(-1) for b in board])
    
    def select_action(self, state):
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps / self.eps_decay)
        self.steps += 1
        if random.random() > self.eps:
            with Param.not_training():
                return np.argmax(self(state))
        else:
            return random.randrange(self.dim_out)

    def replay(self):
        if len(self.memory) < self.replay_batch_size: return
        
        transitions = self.memory.sample(self.replay_batch_size)
        batch = np.array(transitions, dtype=object).T
        b_s0, b_a, b_r, b_s1 = batch
        b_s0, b_s1 = np.stack(b_s0, axis=0), np.stack(b_s1, axis=0)
        b_a, b_r = b_a.astype(np.int), b_r.astype(np.float)
        
        # only keep selected actions
        outputQ = self(b_s0)[onehot(b_a, k=self.dim_out).astype(bool)]
        nextQ = self.past_copy(b_s1).max(axis=1)
        expectedQ = (nextQ * self.gamma) + b_r  # unrelated in backprop
        
        loss = outputQ.mse(expectedQ)
        self.optim(loss.backward())

try:
    agent = Model.load('2048-agent')
    print('model loaded')
    print('memory size:', len(agent.memory))
except FileNotFoundError:
    agent = DQNAgent()

    
def episode_gif(states, gifname=None):
    env.reset()
    frames = []
    for state in states:
        env.board = state
        frames.append(env.render(mode='rgb_array'))
    if gifname is not None:
        makegif(frames, gifname)

def smooth(records, k=50):
    smoothed_records = np.zeros_like(records)
    for i in range(2, len(records)):
        i0 = max(0, i - k)
        smoothed_records[i] = np.mean(records[i0:i])
    return smoothed_records


scores = []
interactive_score_plot = True

if interactive_score_plot:
    fig, ax = plt.subplots()
    fig.show()

for eps in (pb := pbar(range(EPISODES), unit='eps')):
    obs = env.reset()
    obs_record = []  # for animating the episode afterwards
    done = False
    
    while not done:
        if play_mode == 'agent':
            action = agent.select_action(old_obs := obs)
            obs, reward, done, info = env.step(action)
            if done: reward = -128
            agent.memory.push(old_obs, action, reward, obs)
            obs_record.append(obs)
        else:
            env.render()
            action = 'kljh'.index(getwch())
            obs, reward, done, info = env.step(action)

    if play_mode == 'agent':
        if eps % SYNC_INTERVAL == 0:
            agent.past_copy = Model.load(agent.state())
        agent.replay()
    
    scores.append(env.score)
    pb.set_postfix(score=env.score, eps=agent.eps)

    if interactive_score_plot and eps % 10 == 0:
        ax.cla()
        ax.plot(smooth(scores))
        fig.canvas.draw()
        fig.canvas.start_event_loop(0.001)
    
    if (eps + 1) % 500 == 0:
        agent.save('2048-agent')
    if env.score > 1000:
        episode_gif(obs_record)#, f'cartpole:{eps_reward}.gif')
        
    env.close()
    
if not interactive_score_plot:
    plt.plot(smooth(scores))
    plt.show()
