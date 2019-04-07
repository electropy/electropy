import unittest
from electropy.charge import Charge
from electropy.particle import Particle
import numpy as np
from scipy import constants

# Globals
epsilon_0 = constants.epsilon_0
pi = constants.pi


class ParticleTest(unittest.TestCase):
    def setUp(self):
        self.positionList = [0, 0, 0]
        self.charge = 7e-9

    def tearDown(self):
        pass

    # Valid object initiation tests
    def test_that_3D_position_list_returns_an_object_of_type_Charge(self):
        assert isinstance(Charge(self.positionList, self.charge), Charge)

    def test_Charge_object_has_parent_type_Particle(self):
        assert isinstance(Charge(self.positionList, self.charge), Particle)

    #  Object property initialization tests
    def test_charge_property_equal_to_initialized_value(self):
        testObj = Charge(self.positionList, self.charge)
        self.assertEqual(testObj.charge, self.charge)

    def test_q_decorator_equal_to_initialized_charge_value(self):
        testObj = Charge(self.positionList, self.charge)
        self.assertEqual(testObj.q, self.charge)

    #  Field method tests
    def test_field_returns_numpy_array(self):
        testObj = Charge(self.positionList, self.charge)

        fpos = [1, 0, 0]
        electric_field = testObj.field(fpos)

        assert isinstance(electric_field, np.ndarray)

    def test_field_returns_nan_at_particle_position(self):
        testObj = Charge(self.positionList, self.charge)

        fpos = self.positionList

        self.assertTrue(np.isnan(testObj.field(fpos)).all())

    def test_field_returns_known_constant_case(self):
        pos = [0, 0, 0]
        charge = 1

        testObj = Charge(pos, charge)

        fpos = [1, 0, 0]
        field = testObj.field(fpos)

        field_constant = (4 * pi * epsilon_0) ** -1
        expected_field = np.array([field_constant, 0, 0])

        np.testing.assert_equal(field, expected_field)

    def test_field_returns_expected_values(self):
        testObj = Charge(self.positionList, -self.charge)

        fpos = [1, 0, 0]
        electric_field = testObj.field(fpos)

        expected_field = np.array([-62.9, 0, 0])
        np.testing.assert_allclose(electric_field, expected_field, atol=1e-1)

    def test_field_from_potential_returns_near_analytical(self):
        testObj = Charge(self.positionList, -self.charge)

        fpos = [1, 0, 0]
        electric_field = testObj.field(fpos, "potential")

        expected_field = testObj.field(fpos, "analytical")
        np.testing.assert_allclose(electric_field, expected_field, atol=1e-1)

    #  Potential method tests
    def test_potential_returns_float(self):
        testObj = Charge(self.positionList, self.charge)

        ppos = [1, 0, 0]
        electric_potential = testObj.potential(ppos)

        assert isinstance(electric_potential, float)

    def test_potential_returns_nan_at_particle_position(self):
        testObj = Charge(self.positionList, self.charge)

        ppos = self.positionList

        self.assertTrue(np.isnan(testObj.potential(ppos)))

    def test_potential_returns_known_constant_case(self):
        pos = [0, 0, 0]
        charge = 1

        testObj = Charge(pos, charge)

        ppos = [1, 0, 0]
        potential = testObj.potential(ppos)

        expected_potential = (4 * pi * epsilon_0) ** -1

        np.testing.assert_equal(potential, expected_potential)

    def test_neg_gradient_of_potiental_returns_field(self):
        pos = [0, 0, 0]
        charge = 7e-9

        testObj = Charge(pos, charge)

        ppos = [1, 2, -1]
        h = 0.001
        potential_grid = np.empty([3, 3, 3], dtype=object)
        x = np.linspace(ppos[0] - h, ppos[0] + h, 3)
        y = np.linspace(ppos[1] - h, ppos[1] + h, 3)
        z = np.linspace(ppos[2] - h, ppos[2] + h, 3)

        for (i, j, k), _ in np.ndenumerate(potential_grid):
            potential_grid[i][j][k] = testObj.potential([x[i], y[j], z[k]])

        xgrad, ygrad, zgrad = np.gradient(potential_grid, h)
        grad_potential = np.array(
            [xgrad[1, 1, 1], ygrad[1, 1, 1], zgrad[1, 1, 1]]
        )

        electric_field = testObj.field(ppos)
        np.testing.assert_allclose(-grad_potential, electric_field, atol=1e-1)
