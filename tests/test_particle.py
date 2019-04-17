import unittest
from electropy.particle import Particle
import numpy as np


class ParticleTest(unittest.TestCase):
    def setUp(self):
        self.positionList = [-1, 2.5, 0]
        self.velocity = [-9, 1, 4]
        self.acceleration = [0.2, -0.3, 8]

    def tearDown(self):
        pass

    # Error/Exception Raise tests
    def test_particle_initialization_fails_for_non_list_or_np_array(self):
        positionVal = 0
        self.assertRaises(TypeError, Particle, positionVal)

    def test_1D_list_position_throws_error_for_particle_initialization(self):
        position1D = [0]
        self.assertRaises(TypeError, Particle, position1D)

    def test_1D_numpy_position_throws_error_for_particle_initialization(self):
        position1D = np.array([0])
        self.assertRaises(TypeError, Particle, position1D)

    # Valid object initiation tests
    def test_that_3D_position_list_returns_an_object_of_type_Particle(self):
        assert isinstance(Particle(self.positionList), Particle)

    def test_that_3D_numpy_array_returns_an_object_of_type_Particle(self):
        assert isinstance(Particle(np.array(self.positionList)), Particle)

    #  Object property initialization tests
    def test_pos_attribute_not_equal_list_after_init_using_list(self):
        testObj = Particle(self.positionList)
        assert ~isinstance(testObj.position, list)

    def test_list_pos_input_matches_nparr_pos_attribute_after_init(self):
        testObj = Particle(self.positionList)
        np.testing.assert_allclose(
            testObj.position, np.array(self.positionList)
        )

    def test_nparray_pos_input_matches_nparr_pos_attribute_after_init(self):
        testObj = Particle(np.array(self.positionList))
        np.testing.assert_allclose(
            testObj.position, np.array(self.positionList)
        )

    def test_velocity_attribute_not_equal_list_after_init_using_list(self):
        testObj = Particle(self.positionList, velocity=self.velocity)
        assert ~isinstance(testObj.velocity, list)

    def test_list_velocity_input_matches_nparr_velocity_attribute_after_init(
        self
    ):
        testObj = Particle(self.positionList, velocity=self.velocity)
        np.testing.assert_allclose(testObj.velocity, np.array(self.velocity))

    def test_nparray_velocity_input_matches_nparr_velocity_attribute_after_init(
        self
    ):
        testObj = Particle(self.positionList, velocity=np.array(self.velocity))
        np.testing.assert_allclose(testObj.velocity, np.array(self.velocity))

    def test_acceleration_attribute_not_equal_list_after_init_using_list(self):
        testObj = Particle(self.positionList, acceleration=self.acceleration)
        assert ~isinstance(testObj.acceleration, list)

    def test_list_acceleration_input_matches_nparr_accelerationattribute_after_init(
        self
    ):
        testObj = Particle(self.positionList, acceleration=self.acceleration)
        np.testing.assert_allclose(
            testObj.acceleration, np.array(self.acceleration)
        )

    def test_nparray_acceleration_input_matches_nparr_acceleration_attribute_after_init(
        self
    ):
        testObj = Particle(
            self.positionList, acceleration=np.array(self.acceleration)
        )
        np.testing.assert_allclose(
            testObj.acceleration, np.array(self.acceleration)
        )

    #  set_force()
    def test_set_force_stores_expected_force_and_acceleration_value(self):
        testObj = Particle(
            self.positionList,
            acceleration=np.array(self.acceleration),
            mass=2.5,
        )

        force = np.array([5.5, -3, 0.1])

        testObj.set_force(force)

        expected_force = force
        expected_acceleration = force / 2.5

        actual_force = testObj.force
        actual_acceleration = testObj.acceleration

        np.testing.assert_equal(expected_force, actual_force)

        np.testing.assert_equal(expected_acceleration, actual_acceleration)

    def test_set_force_throws_exception_for_zero_mass(self):
        testObj = Particle(
            self.positionList, acceleration=np.array(self.acceleration), mass=0
        )

        force = np.array([5.5, -3, 0.1])

        with self.assertRaises(ValueError):
            testObj.set_force(force)

    def test_set_force_gives_zero_acceleration_for_infinite_mass(self):
        testObj = Particle(
            self.positionList,
            acceleration=np.array(self.acceleration),
            mass=np.inf,
        )

        force = np.array([5.5, -3, 0.1])

        testObj.set_force(force)

        expected_force = force
        expected_acceleration = np.array([0, 0, 0])

        actual_force = testObj.force
        actual_acceleration = testObj.acceleration

        np.testing.assert_equal(expected_force, actual_force)

        np.testing.assert_equal(expected_acceleration, actual_acceleration)

    #  update_motion()
    def test_update_motion_returns_same_position_for_no_velocity_or_accel(
        self
    ):
        testObj = Particle(self.positionList)

        dtime = 10

        testObj.update_motion(dtime)

        expected_position = self.positionList

        actual_position = testObj.position

        np.testing.assert_equal(expected_position, actual_position)

    def test_update_motion_returns_expected_position_for_given_velocity(self):
        testObj = Particle(self.positionList, self.velocity)

        expected_position = np.array(self.positionList)
        constant_velocity = np.array(self.velocity)

        dtime = 10

        testObj.update_motion(10)

        expected_position = expected_position + (constant_velocity * dtime)

        actual_position = testObj.position

        np.testing.assert_equal(expected_position, actual_position)

    def test_update_motion_returns_expected_position_and_velocity_for_given_accel(
        self
    ):
        testObj = Particle(self.positionList, self.velocity, self.acceleration)

        expected_position = np.array(self.positionList)
        initial_velocity = np.array(self.velocity)
        acceleration = np.array(self.acceleration)

        dtime = 10

        testObj.update_motion(10)

        expected_position = (
            expected_position
            + (initial_velocity * dtime)
            + (acceleration * (dtime ** 2)) / 2
        )

        expected_velocity = initial_velocity + (acceleration * dtime)

        actual_position = testObj.position
        actual_velocity = testObj.velocity

        np.testing.assert_equal(expected_position, actual_position)
        np.testing.assert_equal(expected_velocity, actual_velocity)

    #  Momentum decorator
    def test_momentum_returns_expected_value(self):
        mass = 2.5
        velocity = np.array(self.velocity)
        testObj = Particle(self.positionList, velocity, mass=mass)
        np.testing.assert_equal(testObj.momentum, mass * velocity)

    #  Kinetic energy decorator
    def test_energy_kinetic_returns_expected_value(self):
        mass = 6.5
        velocity = np.array(self.velocity)
        testObj = Particle(self.positionList, velocity, mass=mass)
        np.testing.assert_equal(
            testObj.energy_kinetic, (mass * (velocity ** 2)) / 2
        )

    #  X-Y-Z decorators
    def test_that_x_decorator_gives_the_correct_pos_value(self):
        testObj = Particle(self.positionList)
        self.assertEqual(testObj.x, self.positionList[0])

    def test_that_y_decorator_gives_the_correct_pos_value(self):
        testObj = Particle(self.positionList)
        self.assertEqual(testObj.y, self.positionList[1])

    def test_that_z_decorator_gives_the_correct_pos_value(self):
        testObj = Particle(self.positionList)
        self.assertEqual(testObj.z, self.positionList[2])

    def test_that_vx_decorator_gives_the_correct_velocity_value(self):
        testObj = Particle(self.positionList, velocity=self.velocity)
        self.assertEqual(testObj.vx, self.velocity[0])

    def test_that_vy_decorator_gives_the_correct_velocity_value(self):
        testObj = Particle(self.positionList, velocity=self.velocity)
        self.assertEqual(testObj.vy, self.velocity[1])

    def test_that_vz_decorator_gives_the_correct_velocity_value(self):
        testObj = Particle(self.positionList, velocity=self.velocity)
        self.assertEqual(testObj.vz, self.velocity[2])

    def test_that_ax_decorator_gives_the_correct_acceleration_value(self):
        testObj = Particle(self.positionList, acceleration=self.acceleration)
        self.assertEqual(testObj.ax, self.acceleration[0])

    def test_that_ay_decorator_gives_the_correct_acceleration_value(self):
        testObj = Particle(self.positionList, acceleration=self.acceleration)
        self.assertEqual(testObj.ay, self.acceleration[1])

    def test_that_az_decorator_gives_the_correct_acceleration_value(self):
        testObj = Particle(self.positionList, acceleration=self.acceleration)
        self.assertEqual(testObj.az, self.acceleration[2])
