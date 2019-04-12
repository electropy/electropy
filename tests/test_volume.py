import unittest
from electropy.charge import Charge
import numpy as np

from electropy import volume


class VolumeTest(unittest.TestCase):
    def setUp(self):
        self.position_1 = [0, 0, 0]
        self.position_2 = [-2, 4, 1]

        self.charge = 7e-9

    def tearDown(self):
        pass

    # Potential function volume tests
    def test_potential_volume_at_point_equal_class_potential(self):

        charge = Charge(self.position_1, self.charge)

        potential_volume = volume.potential(
            [charge],
            x_range=[-10, 10],
            y_range=[-10, 10],
            z_range=[-10, 10],
            h=1,
        )

        # Point = [-6, -6, -6]
        potential_at_point = potential_volume[4][4][4]

        expected_potential = charge.potential([-6, -6, -6])

        np.testing.assert_equal(potential_at_point, expected_potential)

    def test_two_charge_potential_volume_eq_sum_of_class_potential(self):

        charges = [Charge(self.position_1, self.charge)]
        charges.append(Charge(self.position_2, -self.charge))

        potential_volume = volume.potential(
            charges,
            x_range=[-10, 10],
            y_range=[-10, 10],
            z_range=[-10, 10],
            h=1,
        )

        # Point = [-6, -5, -3]
        potential_at_point = potential_volume[4][5][7]

        expected_potential = np.add(
            charges[0].potential([-6, -5, -3]),
            charges[1].potential([-6, -5, -3]),
        )

        np.testing.assert_equal(potential_at_point, expected_potential)

    # Field function volume tests
    def test_field_volume_at_point_equal_class_field(self):

        charge = Charge(self.position_1, self.charge)

        field_volume = volume.field(
            [charge],
            x_range=[-10, 10],
            y_range=[-10, 10],
            z_range=[-10, 10],
            h=1,
        )

        # Point = [-10, -6, -3]
        field_at_point = field_volume[0][4][7]

        expected_field = charge.field([-10, -6, -3])

        np.testing.assert_equal(field_at_point, expected_field)

    def test_two_charge_field_volume_eq_sum_of_class_field(self):

        charges = [Charge(self.position_1, self.charge)]
        charges.append(Charge(self.position_2, -self.charge))

        field_volume = volume.field(
            charges,
            x_range=[-10, 10],
            y_range=[-10, 10],
            z_range=[-10, 10],
            h=1,
        )

        # Point = [-6, -5, -3]
        field_at_point = field_volume[4][5][7]

        expected_field = np.add(
            charges[0].field([-6, -5, -3]), charges[1].field([-6, -5, -3])
        )

        np.testing.assert_equal(field_at_point, expected_field)

    def test_charge_field_volume_x_components_eq_sum_of_class_field_x(self):

        charges = [Charge(self.position_1, self.charge)]
        charges.append(Charge(self.position_2, -self.charge))

        field_volume = volume.field(
            charges,
            x_range=[-10, 10],
            y_range=[-10, 10],
            z_range=[-10, 10],
            h=1,
            component="x",
        )

        # Point = [-6, -5, -3]
        field_at_point = field_volume[4][5][7]

        expected_field = np.add(
            charges[0].field([-6, -5, -3], component="x"),
            charges[1].field([-6, -5, -3], component="x"),
        )

        np.testing.assert_equal(field_at_point, expected_field)

    def test_charge_field_volume_y_components_eq_sum_of_class_field_y(self):

        charges = [Charge(self.position_1, self.charge)]
        charges.append(Charge(self.position_2, -self.charge))

        field_volume = volume.field(
            charges,
            x_range=[-10, 10],
            y_range=[-10, 10],
            z_range=[-10, 10],
            h=1,
            component="y",
        )

        # Point = [-6, -5, -3]
        field_at_point = field_volume[4][5][7]

        expected_field = np.add(
            charges[0].field([-6, -5, -3], component="y"),
            charges[1].field([-6, -5, -3], component="y"),
        )

        np.testing.assert_equal(field_at_point, expected_field)

    def test_charge_field_volume_z_components_eq_sum_of_class_field_z(self):

        charges = [Charge(self.position_1, self.charge)]
        charges.append(Charge(self.position_2, -self.charge))

        field_volume = volume.field(
            charges,
            x_range=[-10, 10],
            y_range=[-10, 10],
            z_range=[-10, 10],
            h=1,
            component="z",
        )

        # Point = [-6, -5, -3]
        field_at_point = field_volume[4][5][7]

        expected_field = np.add(
            charges[0].field([-6, -5, -3], component="z"),
            charges[1].field([-6, -5, -3], component="z"),
        )

        np.testing.assert_equal(field_at_point, expected_field)

    def test_field_returns_singleton_dim_for_single_slice(self):

        charge = Charge(self.position_1, self.charge)

        field_volume = volume.field(
            [charge],
            x_range=[-10, 10],
            y_range=[1, 1],
            z_range=[-10, 10],
            h=0.1,
        )

        expected_shape = (201, 1, 201)
        actual_shape = field_volume.shape

        np.testing.assert_equal(actual_shape, expected_shape)

    def test__arange_almost_equals_numpy_arange(self):

        actual = volume._arange(-10, 10, 0.1)  # Mine is rounder anyways =)
        expected = np.arange(-10, 10 + 0.1, 0.1)
        np.testing.assert_almost_equal(actual, expected)
