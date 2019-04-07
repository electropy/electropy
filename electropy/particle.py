import numpy as np


class Particle(object):
    """Base class for particles

    Attributes:
        pos: 1D numpy array of length 3

    Methods:
        getPosition(): Returns numpy array

    """

    def __init__(self, pos):
        """
        Constructor takes 1 argument, which can be a numpy array or a list
        """
        self.pos = np.asarray(pos)
        if not self.__verify3D__(self.pos):
            raise TypeError(
                "Initializer argument must be a \
                            1D numpy array or list of length 3"
            )

    # X-Y-Z position value decorators
    @property
    def x(self):
        """'x' position value"""
        return self.pos[0]

    @property
    def y(self):
        """'y' position value"""
        return self.pos[1]

    @property
    def z(self):
        """'z' position value"""
        return self.pos[2]

    # Private methods
    def __verify3D__(self, nparray):
        return nparray.shape == (3,)
