import numpy as np


class Particle(object):
    """Base class for particles

    Attributes:
        pos: 1D numpy array of length 3

    Methods:
        getPosition(): Returns numpy array

    """

    def __init__(
        self, position, velocity=[0, 0, 0], acceleration=[0, 0, 0], mass=np.inf
    ):
        """
        Constructor arguments need to be numpy arrays or lists.
        Only position argument is required; velocity and acceleration are
        optional.
        """
        self.position = np.asarray(position)
        self.velocity = np.asarray(velocity)
        self.acceleration = np.asarray(acceleration)

        self.mass = mass

        if not (
            self.__verify3D__(self.position)
            and self.__verify3D__(self.velocity)
            and self.__verify3D__(self.acceleration)
        ):
            raise TypeError(
                "Initializer arguments must be a \
                            1D numpy array or list of length 3"
            )

    def set_force(self, force):
        """ Set force
        Sets force property of the object, and calculate the current
        acceleration.

        If mass == 0, a ValueError is raised, as force has no meaning on such
        an object.

        If mass == np.inf, acceleration is zero, and the object will not react
        to this the force. i.e. If it was immobile, its position will not
        change, whereas if it has a velocity, it will continue moving at that
        velocity.
        """

        if self.mass == 0:
            raise ValueError("Cannot set force for a particle with zero mass.")

        self.force = np.asarray(force)

        if not (self.__verify3D__(self.force)):
            raise TypeError(
                "Force arguments must be a \
                            1D numpy array or list of length 3"
            )

        self.acceleration = self.force / self.mass

    def update_motion(self, dtime):
        """ Update motion
        Update the position and velocity for a given time and acceleration.

        Args:
            dtime: duration of time (in seconds) where acceleration (or force)
            is assumed to be constant in the calculation.
        """
        initial_position = self.position
        initial_velocity = self.velocity

        self.position = (
            initial_position
            + initial_velocity * dtime
            + (self.acceleration * (dtime ** 2)) / 2
        )
        self.velocity = initial_velocity + self.acceleration * dtime

    # Physical property decorators
    @property
    def momentum(self):
        """Momentum (kg * m/s)"""
        return self.mass * self.velocity

    @property
    def energy_kinetic(self):
        """Kinetic energy (joules)"""
        if self.mass == 0:
            raise ValueError(
                "Cannot calculate kinetic energy of a zero-mass object"
            )
        return (self.momentum ** 2) / (2 * self.mass)

    # X-Y-Z position value decorators
    @property
    def x(self):
        """'x' position value"""
        return self.position[0]

    @property
    def y(self):
        """'y' position value"""
        return self.position[1]

    @property
    def z(self):
        """'z' position value"""
        return self.position[2]

    # X-Y-Z velocity value decorators
    @property
    def vx(self):
        """'x' velocity value"""
        return self.velocity[0]

    @property
    def vy(self):
        """'y' velocity value"""
        return self.velocity[1]

    @property
    def vz(self):
        """'z' velocity value"""
        return self.velocity[2]

    # X-Y-Z acceleration value decorators
    @property
    def ax(self):
        """'x' acceleration value"""
        return self.acceleration[0]

    @property
    def ay(self):
        """'y' acceleration value"""
        return self.acceleration[1]

    @property
    def az(self):
        """'z' acceleration value"""
        return self.acceleration[2]

    # Private methods
    def __verify3D__(self, nparray):
        return nparray.shape == (3,)
