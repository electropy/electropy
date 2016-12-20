from .particle import Particle

class Charge(Particle):
    """Base class for particles

    Attributes:
        pos: 1D numpy array of length 3
    
    Methods:
        getPosition(): Returns numpy array

    """

    def __init__(self, pos):
        """Constructor takes 1 argument, which can be a numpy array or a list"""
        Particle.__init__(self, pos)
