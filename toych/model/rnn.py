from .basic import *
del sum


class RNN(Model):
    """ A simple RNN model. """
    
    def __init__(self, hidden_dim, out_dim):
        self.Axh = affine(hidden_dim)
        self.Ahh = affine(hidden_dim)
        self.Ahy = affine(out_dim)
        self.h = zeros(hidden_dim)
        
    def reset_hstate(self):
        self.h = zeros(*self.h.shape)
        
    def apply(self, x):
        self.h = tanh(self.Axh(x) + self.Ahh(self.h))
        return self.Ahy(self.h)
    
    def detach(self):
        self.h.detach()
    
    def __call__(self, input):
        self.detach()
        return [self.apply(x) for x in input]
    
    def generate(self, n, start, reset=True):
        if reset: self.reset_hstate()
        out = [self(start)[-1]]
        with Param.not_training():
            [out.append(self.apply(out[-1])) for _ in range(n - 1)]
        return np.array(out)
    
    @classmethod
    def default_loss(cls, output, labels):
        loss, n = 0, 0
        for y, t in zip(output, labels):
            loss += y.smce(t)
            n += 1
        return loss / n

    def fit(self, input, target=None, *, epochs=10, lr=None, bs=None, optimizer=None, loss=None, 
            val_data=None, val_bs=500, metrics={}, callbacks=(), callback_each_batch=False,
            showgraph=False) -> dict:
        if target is None:
            input, target = input[:-1], input[1:]
            if val_data: val_data = val_data[:-1], val_data[1:]
        BatchLoader.randperm = False
        return super().fit(input, target, epochs=epochs, lr=lr, bs=bs, optimizer=optimizer,
                           loss=loss, val_data=val_data, val_bs=val_bs, metrics=metrics, 
                           callbacks=callbacks, callback_each_batch=callback_each_batch, showgraph=showgraph)

class LSTM(RNN):
    """Long-short term memory."""

    def __init__(self, hid_dim, out_dim):
        self.w = [affine(hid_dim) for _ in range(4)]
        self.u = [affine(hid_dim) for _ in range(4)]
        self.v = affine(out_dim)
        self.h, self.c = zeros(hid_dim), zeros(hid_dim)
        
    def reset_hstate(self):
        self.h = zeros(*self.h.shape)
        self.c = zeros(*self.c.shape)
        
    def detach(self):
        self.h.detach()
        self.c.detach()

    def apply(self, x):
        wf, wi, wo, wc = self.w
        uf, ui, uo, uc = self.u
        h = self.h
        f = sigmoid(wf(x) + uf(h))  # forget
        i = sigmoid(wi(x) + ui(h))  # input
        o = sigmoid(wo(x) + uo(h))  # output
        c = tanh(wc(x) + uc(h))     # candidate
        self.c = self.c*f + i*c     # update
        self.h = o * tanh(self.c)
        # setparnames()
        return self.v(o)
