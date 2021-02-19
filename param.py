import numpy as np


class Parameter(np.ndarray):
    """A trainable parameter in a node."""

    grad_lim = 1e8  # limit of the magnitude of each element of the gradient

    def __new__(cls, value=None, *, size=None, mean=0, scale=None):
        """Create a new Parameter.

        If `value` is given, then it will be converted to a Parameter.
        However, if `size` is additionally specified, then a new Parameter
        of this size will be created filled with the given `value`.
        
        If `value` is not given, then `size` must be provided to generate
        a random Parameter following Gaussian distribution. Additionally,
        `mean` and `scale` of the Gaussian can be specified.
        """
        if value is None:  # random initialization
            if size is None:
                raise AssertionError('the size of the Parameter must be'
                                     'given for random initialization')

            if scale is None:
                length = size[0] if hasattr(size, '__len__') else size
                scale = 1 / np.sqrt(length)  # Xavier initialization

            param = np.random.normal(loc=mean, scale=scale, size=size)
            return Parameter(param)

        else:  # convert the value to an array
            if size is not None:  # fill an array of the given size
                value = np.full(size, value)
            return np.asarray(value).view(cls)

    @property
    def grad(self):
        if not hasattr(self, '_grad'):
            self._grad = 0
            self._dirty = False
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = np.clip(value, -self.grad_lim, self.grad_lim)
        self._dirty = True

    def zero_grad(self):
        self._grad = 0
        self._dirty = False

    @property
    def need_update(self):
        return self._dirty
