import unittest
import electropy as ep
import numpy as np

class ParticleTest(unittest.TestCase):

    def setUp(self):
        self.positionList = [0,0,0]

    def tearDown(self):
        pass

    def test_that_1D_list_position_throws_error_for_particle_initialization(self):
        position1D = [0]
        self.assertRaises(TypeError, ep.Particle, position1D)

    def test_that_1D_numpy_position_throws_error_for_particle_initialization(self):
        position1D = np.array([0])
        self.assertRaises(TypeError, ep.Particle, position1D)

    def test_that_3D_position_list_returns_an_object_of_type_Particle(self):
        assert(isinstance(ep.Particle(self.positionList), ep.Particle))

    def test_that_3D_numpy_array_returns_an_object_of_type_Particle(self):
        assert(isinstance(ep.Particle(np.array(self.positionList)), ep.Particle))
