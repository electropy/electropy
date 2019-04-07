from .particle import Particle
import numpy as np
from scipy import constants

# Globals
epsilon_0 = constants.epsilon_0
pi = constants.pi


class Charge(Particle):
    """Base class for a point electric charge

    Attributes:
        pos: particle position, 1D numpy array of length 3
        charge: electric charge in Coulomb.

    Methods:
        getPosition(): Returns numpy array

    """

    def __init__(self, pos, charge):
        """Charge clas initializer

        Args:
            pos: position. units: meters. numpy array or a list.
            charge: electric charge. units: Coulombs. float.
        """
        Particle.__init__(self, pos)
        self.charge = charge

    @property
    def q(self):
        """Electric charge value in Coulomb"""
        return self.charge

    def field(self, fpos):
        """Electric field at a given position.

        Args:
            fpos: field position. numpy array or a list.
        """

        if isinstance(fpos, np.ndarray):
            if self.__verify3D__(fpos):
                pass
            else:
                raise TypeError(
                    "Position must be a 1D numpy array or list of length 3"
                )

        elif isinstance(fpos, list):
            if self.__verify3D__(np.array(fpos)):
                fpos = np.array(fpos)
            else:
                raise TypeError(
                    "Position must be a 1D numpy array or list of length 3"
                )

        if np.array_equal(fpos, self.pos):
            electric_field = fpos.astype(float)
            electric_field.fill(np.nan)

            return electric_field

        displacement = fpos - self.pos

        electric_field = (
            self.q
            * (4 * pi * epsilon_0) ** -1
            * displacement
            * np.linalg.norm(displacement) ** -3
        )

        return electric_field
