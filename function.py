from utils import abstractmethod


class Function:
    """Base class of functions that implement forward and backward."""
    
    @abstractmethod
    def forward(self, *args, **kwds):
        """Take the input and compute the output."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwds):
        """Back-propagate the error."""
        raise NotImplementedError
        
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
