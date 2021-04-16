from init import *
from gym2048 import Game2048Env
from msvcrt import getwch
import matplotlib

env = Game2048Env()
play_mode = 'agent'

del sum, max


class DQNAgent(Model):
    board_shape = (env.h, env.w)
    dim_out = env.action_space.n
    
    conv_filters = 16
    hidden_dim1 = 64
    hidden_dim2 = 32
    
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    replay_batch_size = 128

    optim = Adam(lr=5e-3, reg='l2', lamb=0.01)
    
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
        # self.conv1 = Conv2D(self.conv_filters, size=(1, 2))
        # self.conv2 = Conv2D(self.conv_filters, size=(2, 1))
        self.apply = Compose(
            self.preprocess,
            # self.convolve_and_flatten,
            flatten,
            Affine(self.hidden_dim1), leakyReLU,
            Affine(self.hidden_dim2), leakyReLU,
            Affine(self.dim_out), leakyReLU
        )
        
        self.eps = 1.
        self.memory = self.Memory()
        
    def preprocess(self, board):
        return np.log2(board + 1.).reshape(-1, 1, *self.board_shape)
    
    def convolve_and_flatten(self, im):
        return concat(self.conv1(im).flatten(), self.conv2(im).flatten())
    
    # def clear_memory(self):
    #     for key in ['s0', 'a', 'r', 's1']:
    #         self.memory[key] = []

    # def add_memory(self, s0, a, r, s1):
    #     for k, v in vars().items():
    #         if k != 'self':
    #             self.memory[k].append(v)
    #             if len(self.memory[k]) > self.memory_size:
    #                 for _ in range(10): self.memory[k].pop(0)
                    
    def select_action(self, state):
        eps_thr = self.eps_end + (self.eps_start - self.eps_end) * self.eps
        self.eps *= np.exp(-1 / self.eps_decay)
        if random.random() > eps_thr:
            return np.argmax(self(state))
        else:
            return random.randrange(self.dim_out)
                
    def replay(self):
        if len(self.memory) < self.replay_batch_size: return
        transitions = self.memory.sample(self.replay_batch_size)
        batch = np.array(transitions).T
        b_s0, b_a, b_r, b_s1 = batch
        
        outputQ = self(b_s0)
        nonfinal_mask = np.array([s is not None for s in b_s1])
        nonfinal_next_states = np.array([s for s in b_s1 if s is not None])
        nextQ = np.zeros(b_s1.shape)
        nextQ[nonfinal_mask] = self(nonfinal_next_states)
        expectedQ = (nextQ * self.gamma) + b_r
        
        loss = self.loss(outputQ, expectedQ)
        self.optim(loss.backward())
        
    def loss(self, outputQ, expectedQ):
        return outputQ.mse(expectedQ)

    
def episode_gif(states, gifname=None):
    env.reset()
    frames = []
    for state in states:
        env.board = state
        frames.append(env.render(mode='rgb_array'))
    if gifname is not None:
        makegif(frames, gifname)


try:
    agent = Model.load('2048-agent')
    print('model loaded')
except FileNotFoundError:
    agent = Agent()



def smooth(records, k=50):
    smoothed_records = np.zeros_like(records)
    for i in range(2, len(records)):
        i0 = max(0, i - 10)
        smoothed_records[i] = np.mean(records[i0:i])
    return smoothed_records


scores = []

fig, ax = plt.subplots()
fig.show()

for eps in (pb := pbar(range(1000), unit='eps')):
    obs = env.reset()
    obs_record = []  # for animating the episode afterwards
    done = False
    
    while not done:
        if play_mode == 'agent':
            probabs = agent(old_obs := obs)[0]
            action = np.argmax(np.random.multinomial(1, probabs))
            obs, reward, done, info = env.step(action)
            memory.add(old_obs, probabs, action, reward)
            obs_record.append(obs)
        else:
            env.render()
            action = 'kljh'.index(getwch())
            obs, reward, done, info = env.step(action)

    if play_mode == 'agent':
        loss = loss_func(memory)
        params = loss.backward()
        optim(params)
        agent.replay(memory.buffer)
    
    scores.append(env.score)
    pb.set_postfix(score=env.score)

    if eps % 5 == 0:
        ax.cla()
        ax.plot(np.arange(eps+1), smooth(scores))
        fig.canvas.draw()
        fig.canvas.start_event_loop(0.001)
    
    if env.score > 1000:
        agent.save('2048-agent')
        episode_gif(obs_record)#, f'cartpole:{eps_reward}.gif')
        
    env.close()
    
