from utils import abstractmethod


class Function:
    """Base class of functions that implement forward and backward."""
    
    def __init__(self):
        self._forward = super().__getattribute__("forward")
        self._backward = super().__getattribute__("backward")
    
    @abstractmethod
    def forward(self, *args, **kwds):
        """Take the input and compute the output."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwds):
        """Back-propagate the error."""
        raise NotImplementedError
    
    @abstractmethod
    def _wrapped_forward(self, *args, **kwds):
        """Wrapper of forward method to add preprocessing and postprocessing.
        
        When the user calls `forward`, `_wrapped_forward` will be called instead.
        The `forward` method implemented by a subclass is accessed via `self._forward`.
        """
        return self._forward(*args, **kwds)
    
    @abstractmethod
    def _wrapped_backward(self, *args, **kwds):
        """Wrapper of backward method to add preprocessing and postprocessing."""
        return self._backward(*args, **kwds)

    def __getattribute__(self, attr):
        if attr == "forward":
            return super().__getattribute__("_wrapped_forward")
        elif attr == "backward":
            return super().__getattribute__("_wrapped_backward")
        else:
            return super().__getattribute__(attr)
        
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
