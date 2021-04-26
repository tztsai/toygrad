from importer import *
from data.mnist import *


class MLP:
    """Multi-layer Perceptron"""
    learning_rate = 1e-3
    momentum = 0.9

    def __init__(self, shape, lr=learning_rate):
        """
        Construct a dense multi-layer perceptron network

        Args:
            shape (list of int): number of nodes in each layer
        """
        self.shape = shape
        self.depth = len(shape) - 1
        self.lr = lr

        # weight matrix of each layer
        self.weights = [self.init_weight(shape[i] + 1, shape[i + 1])
                        for i in range(self.depth)]

        # values stored in each layer
        self.values = [None] * (self.depth + 1)

        # gradients of weights in each layer
        self._grads = [None] * self.depth

        # records of delta weight in the last update
        self._deltas = [0] * self.depth
        
    @staticmethod
    def init_weight(d_in, d_out):
        s = np.sqrt(2 / d_in)
        return np.random.normal(size=[d_in, d_out], scale=s)

    def forward(self, X):
        """
        Pass forward the inputs to produce the outputs.

        Args:
            X: input vector or matrix (each row is an input)

        Returns:
            output vector or matrix.
        """
        if len(np.shape(X)) == 1:
            X = np.array([X])
            batch = False
        else:
            batch = True

        self.values[0] = X

        for layer in range(self.depth):
            W = self.weights[layer]
            b, W = W[0], W[1:]

            O = self.sigma(X @ W + b)
            self.values[layer + 1] = O
            X = O  # this layer's output is next layer's input

        return O if batch else O[0]

    def backward(self, E):
        """
        Backprop the errors between the outputs and the actual labels.

        Args:
            E: array of differences between the outputs and the actual labels
        """
        for layer in reversed(range(self.depth)):
            input = self.values[layer]
            output = self.values[layer + 1]
            D = E * self.sigma_deriv(output)

            # compute gradient
            dw = input.T @ D
            db = np.sum(D, axis=0)  # gradient of bias
            grad = np.insert(dw, 0, db, axis=0)
            self._grads[layer] = grad

            if layer > 0:
                W = self.weights[layer][1:]  # bias removed
                E = D @ W.T

    def update(self):
        for layer in range(self.depth):
            grad = self._grads[layer]
            prev_delta = self._deltas[layer]
            alpha = self.momentum
            delta = alpha * prev_delta - (1 - alpha) * grad

            # add the gradient of bias
            self.weights[layer] += self.lr * delta

            # record the new delta
            self._deltas[layer] = delta

    def fit(self, X, Y, epochs=20, animate=False, anim_with_data=True,
            plot_curves=False, eval_performance=False):
        assert X.shape[1] == self.shape[0], 'input dimension mismatch'
        assert Y.shape[1] == self.shape[-1], 'output dimension mismatch'

        batches = BatchLoader(X, Y)
        history = {'loss': [], 'val_acc': []}

        for epoch in range(epochs):
            print('\nEpoch:', epoch + 1)
            loss = 0

            with Profile('test MLP'):
                output, labels = [], []

                for Xb, Yb in pbar(batches):
                    # compute output
                    Ob = self.forward(Xb)

                    # backprop error
                    Eb = Ob - Yb
                    loss += np.sum(Eb ** 2)
                    self.backward(Eb)

                    # update weights
                    self.update()

                    output.extend(Ob)
                    labels.extend(Yb)

            history['loss'].append(loss / len(X))
            
            val_acc = accuracy(self.forward(x_val), y_val)
            history['val_acc'].append(val_acc)

            print(', '.join('%s = %.2f' % (k, v[-1])
                            for k, v in history.items()))

    def sigma(self, x):
        """Nonlinearity function."""
        return np.tanh(x)

    def sigma_deriv(self, sigma):
        """The derivative of sigma expressed in terms of sigma itself."""
        return 1 - sigma ** 2


def classifier(classes):
    return Compose(
        affine(30), tanh,#reLU, dropout, leakyReLU,
        affine(classes), tanh
    )


# x_train, x_test = x_train.reshape(-1,1,*im_size), x_test.reshape(-1,1,*im_size)
y_train, y_test = [onehot(y, 10) for y in (y_train, y_test)]
(x_train, y_train), (x_val, y_val) = train_val_split(x_train, y_train)


def accuracy(out, labels):
    return (np.argmax(out, axis=1) == np.argmax(labels, axis=1)).astype(float).mean()


setloglevel('debug')

# clf = MLP([input_dim, 30, 10])
# clf.fit(x_train, y_train, epochs=10)

clf = classifier(10)
history = clf.fit(x_train, y_train, epochs=10, loss='l2', optimizer='sgd',
                  val_data=(x_val, y_val), metrics={'val_acc': accuracy})
