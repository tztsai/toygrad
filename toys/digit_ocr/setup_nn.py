#%%
from nn import *
from pathlib import Path
import requests
import pickle
import gzip

DATA_DIR = Path('.')
URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"
PATH = DATA_DIR / FILENAME

DATA_DIR.mkdir(parents=True, exist_ok=True)

if not PATH.exists():
    content = requests.get(URL + FILENAME).content
    PATH.open("wb").write(content)

with gzip.open(PATH.as_posix(), "rb") as f:
    ((x_train, y_train), (x_test, y_test), _) = pickle.load(f, encoding="latin-1")

im_size = (28, 28)
input_dim = np.prod(im_size)


#%% Evaluate the best configuration
# d_hs = [10, 15, 20]  # dims of the hidden layer
# models = [NN(input_dim, d_h, 10) for d_h in d_hs]
# accuracies = []

def accuracy(model):
    return np.mean(model.predict(x_test, argmax=True) == y_test)

# for model in models:
#     losses = model.fit(x_train, onehot(y_train, 10), epochs=10)
#     plt.plot(range(len(losses)), losses, label=str(model.shape))
#     accuracies.append(accuracy(model))
    
# plt.legend()
# plt.show()
# print(accuracies)


#%%
# nn = models[np.argmax(accuracies)]
# print('Using a neural network of shape', nn.shape)
nn = NN(input_dim, 15, 15, 10)
nn.fit(x_train, onehot(y_train, 10), epochs=15)
print('Accuracy: %.1f%%' % (accuracy(nn) * 100))

# visualize prediction
x_sample = x_test[np.random.randint(1000, size=8)]
output = nn.predict(x_sample, argmax=True)

for i in range(8):
    # plot image
    ax = plt.subplot(8, 2, 2 * i + 1)
    ax.axis('off')
    ax.imshow(x_sample[i].reshape(im_size), cmap='gray')

    # plot prediction
    ax = plt.subplot(8, 2, 2 * i + 2)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.axis('off')
    ax.text(0, 0, str(output[i]), fontsize=15)

plt.show()

nn.save('ocr.pkl')
