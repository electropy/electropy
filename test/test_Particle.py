import unittest
import electropy as ep

class ParticleTest(unittest.TestCase):

    def setUp(self):
        self.positionList = [0,0,0]

    def tearDown(self):
        pass

    def test_that_1D_position_throws_error_for_particle_initialization(self):
        position1D = [0]
        self.assertRaises(TypeError, ep.Particle, position1D)
