# %%
import sys
import time
from collections import Counter
sys.path.append('..')

from importer import *
time.sleep(.5)

DATAPATH = 'goblet.txt' # '史记.txt'
BOOKLEN = -1

def sample_categorical(probabs):
    r = np.random.rand()
    for i, s in enumerate(np.cumsum(probabs)):
        if s > r: return i

def text2array(text):
    if isinstance(text, np.ndarray):
        text = map(lambda c: c.item(), text)
    return onehot([CHAR2INT[c] for c in text], DATA_DIM)

def softmax(x):
    return (ex := np.exp(x)) / np.sum(ex, axis=-1, keepdims=1)

def array2text(array):
    seq = [sample_categorical(softmax(x)) for x in array]
    return ''.join(INT2CHAR[i] for i in seq)

with open(DATAPATH, 'r', encoding='utf8') as f:
    book = list(f.read()[:BOOKLEN])

CHARS = Counter(book)
for i, c in enumerate(book):
    if CHARS.get(c, 0) < 5:
        if c in CHARS:
            del CHARS[c]
            CHARS['*'] += 1
        book[i] = '*'
DATA_DIM = len(CHARS)
CHAR2INT = {c: i for i, c in enumerate(CHARS)}
INT2CHAR = dict(enumerate(CHARS))

print('Data loaded.')
print('Number of characters:', DATA_DIM)

# %%
smooth_loss = None

def generate(**vars):
    global smooth_loss
    if vars['t'] % 200 == 0:
        rnn = vars['self']
        seq = rnn.generate(100, start=text2array('. '))
        print(array2text(seq))
    if smooth_loss is None:
        smooth_loss = vars['ls'].item()
    else:
        smooth_loss = vars['ls'].item() * .001 + smooth_loss * .999
    vars['pb'].set_postfix(loss=smooth_loss)

RNN = model.rnn.RNN
RNN.batch_loader.preprocess = staticmethod(text2array)

nn = RNN(256, DATA_DIM)
nn.fit(book, epochs=2, bs=5, callbacks=[generate], callback_each_batch=True)
