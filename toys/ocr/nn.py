import pickle
import numpy as np
import matplotlib.pyplot as plt


class NN:
    def __init__(self, input_dim, *layer_dims, with_bias=True):
        self.shape = [input_dim, *layer_dims]
        self.depth = len(self.shape) - 1
        self.with_bias = with_bias
        self.weights = [init_weight(self.shape[k] + with_bias, self.shape[k+1])
                        for k in range(self.depth)]
        self.grads = [None] * self.depth  # gradients
        self._inputs = [None] * self.depth
        self._outputs = [None] * self.depth

    @classmethod
    def read(cls, filename):
        nn = cls(1)
        nn.load(filename)
        return nn
        
    def fit(self, input, target, epochs=30, lr=0.01, callbacks=[]):
        """Fit the weights given the input and the target data.
        
        Args:
            input: the input matrix of size N * D_IN
            target: the target matrix of size N * D_OUT
            epochs: number of iterations to train
            lr: learning rate
        """
        losses = []
        
        for epoch in range(epochs):
            print('\nEpoch %d:' % epoch)
            loss = 0
            
            for x, t in get_batches(input, target):
                y = self.forward(x)
                self.backward(y, t)
                self.update_weights(lr)
                loss += np.sum((y - t) ** 2)
                
            # compute the mean square error
            loss /= len(input)
            print('Loss =', loss)
            losses.append(loss)
            
            for callback in callbacks:
                callback(self)

        return losses
        
    def forward(self, input):
        """Pass forward the input and compute the output.
        
        Args:
            input: a matrix in which each row is an input vector

        Returns:
            The output matrix.
        """
        for k in range(self.depth):
            if self.with_bias:  # insert a column of constant 1
                input = np.insert(input, 0, 1, axis=1)
            self._inputs[k] = input
            
            output = self.activation(input @ self.weights[k])
            self._outputs[k] = output
            input = output
        
        return output
    
    def predict(self, input, argmax=False):
        if len(np.shape(input)) == 1:
            # plt.imshow(np.reshape(input, (28, 28)))
            # plt.show()
            input = np.reshape(input, [1, -1])
            batch = False
        else:
            batch = True
        output = self.forward(input)
        if argmax:
            output = np.argmax(output, axis=-1)
        return output if batch else output[0]

    def backward(self, output, target):
        """Back propagate the error between the output and the target and compute the gradients."""
        error = output - target
        for k in reversed(range(self.depth)):
            input, output = self._inputs[k], self._outputs[k]
            error *= self.deriv_activation(output)
            grad = input.T @ error
            self.grads[k] = grad
            if k > 0:  # pass back the error
                error = error @ self.weights[k][1:].T
                
    def update_weights(self, lr):
        for k in range(self.depth):
            self.weights[k] -= lr * self.grads[k]
        
    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def deriv_activation(y):
        return y * (1 - y)
    
    def state_dict(self):
        return self.__dict__
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.state_dict(), f)
        
    def load(self, filename):
        with open(filename, 'rb') as f:
            state_dict = pickle.load(f)
        for attr in state_dict:
            setattr(self, attr, state_dict[attr])

        
def init_weight(*size, scale=None):
    if scale is None:
        scale = 1 / np.sqrt(size[0])
    return np.random.normal(scale=scale, size=size)


def get_batches(x, y, bs=32):
    """A iterator that produces batches of data.
    
    Args:
        x, y: input and target data
        bs: batch size
    """
    n = len(x)
    steps = range(0, n, bs)
    for i in steps:
        yield x[i:i+bs], y[i:i+bs]
        

def onehot(x, k, hot=1, cold=-1):
    y = np.full([len(x), k], cold)
    for i, j in enumerate(x):
        y[i, j] = hot
    return y
        
        
if __name__ == '__main__':
    data = np.concatenate([np.random.normal(0, size=[100, 2]), np.random.normal(2, size=[100, 2])])
    labels = np.array([-1]*100 + [1]*100).reshape(-1, 1)
    nn = NN(2, 3, 1)
    losses = nn.fit(data, labels)

    plt.plot(range(len(losses)), losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
