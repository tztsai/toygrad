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
    
    # conv_filters = 16
    hidden_dim1 = 64
    hidden_dim2 = 64
    
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    replay_batch_size = 128

    optim = Adam(lr=5e-3, reg='l2', lamb=1e-3)
    
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
            Affine(self.dim_out), softmax
        )
        
        self.eps = 1.
        self.memory = self.Memory()
        
    def preprocess(self, board):
        return np.log2(board + 1.).reshape(-1, 1, *self.board_shape)
    
    def convolve_and_flatten(self, im):
        return concat(self.conv1(im).flatten(), self.conv2(im).flatten())
    
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
        batch = np.array(transitions, dtype=object).T
        b_s0, b_a, b_r, b_s1 = batch
        b_s0 = np.stack(b_s0, axis=0)
        b_a, b_r = b_a.astype(np.int), b_r.astype(np.float)
        
        mask1 = onehot(b_a, k=self.dim_out).astype(bool)
        mask2 = np.array([s is not None for s in b_s1])
        
        outputQ = self(b_s0)[mask1]  # only keep selected actions
        nextQ = np.zeros(outputQ.shape)
        nonfinal_next_states = np.array([s for s in b_s1 if s is not None])
        nextQ[mask2] = self(nonfinal_next_states).max(axis=1)
        expectedQ = (nextQ * self.gamma) + b_r  # unrelated in backprop
        
        loss = self.loss(outputQ, expectedQ)
        self.optim(loss.backward())
        
        from tests.gradcheck import checkgrad
        def loss(w):
            self.apply[2].w = w
            outputQ = self(b_s0)[mask1]  # only keep selected actions
            nextQ = np.zeros(outputQ.shape)
            nonfinal_next_states = np.array([s for s in b_s1 if s is not None])
            nextQ[mask2] = self(nonfinal_next_states).max(axis=1)
            expectedQ = (nextQ * self.gamma) + b_r  # unrelated in backprop
            return self.loss(outputQ, expectedQ)
        checkgrad(self.apply[2].w, loss)
        exit(0)
        
    def loss(self, outputQ, expectedQ):
        return outputQ.mse(expectedQ)


try:
    agent = Model.load('2048-agent')
    print('model loaded')
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
        i0 = max(0, i - 10)
        smoothed_records[i] = np.mean(records[i0:i])
    return smoothed_records


scores = []
interactive_score_plot = False

if interactive_score_plot:
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
            if done: obs = None
            agent.memory.push(old_obs, action, reward, obs)
            obs_record.append(obs)
        else:
            env.render()
            action = 'kljh'.index(getwch())
            obs, reward, done, info = env.step(action)

    if play_mode == 'agent':
        agent.replay()
    
    scores.append(env.score)
    pb.set_postfix(score=env.score)

    if interactive_score_plot and eps % 5 == 0:
        ax.cla()
        ax.plot(np.arange(eps+1), smooth(scores))
        fig.canvas.draw()
        fig.canvas.start_event_loop(0.001)
    
    if env.score > 1000:
        agent.save('2048-agent')
        episode_gif(obs_record)#, f'cartpole:{eps_reward}.gif')
        
    env.close()
    
if not interactive_score_plot:
    plt.plot(smooth(scores))
    plt.show()
