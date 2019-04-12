import numpy as np


def potential(
    charge_objs,
    x_range=[-10, 10],
    y_range=[-10, 10],
    z_range=[-10, 10],
    h=0.01,
):
    """Calculate potential in a volume

    Args:
        charge_objs: list of Charge objects
        x_range, y_range, z_range: [min, max] distances of volume.
                                    units: meters
        h: spacing between array elements. units: meters

    Return: 3D numpy array

    """
    x = _arange(x_range[0], x_range[1], h)
    y = _arange(y_range[0], y_range[1], h)
    z = _arange(z_range[0], z_range[1], h)

    potential_grid = np.zeros([x.size, y.size, z.size], dtype=float)

    for charge in charge_objs:
        for (i, j, k), _ in np.ndenumerate(potential_grid):
            potential_grid[i][j][k] += charge.potential([x[i], y[j], z[k]])

    return potential_grid


def field(
    charge_objs,
    x_range=[-10, 10],
    y_range=[-10, 10],
    z_range=[-10, 10],
    h=0.01,
    type="analytical",
    component=None,
):
    """Calculate field in a volume

    Args:
        charge_objs: list of Charge objects
        x_range, y_range, z_range: [min, max] distances of volume.
                                    units: meters
        h: spacing between array elements. units: meters
        type: type of field calculation. 'analytical' (default) or from
                gradient of potential.
        component: 'x', 'y', 'z', or None (default)

    Return: 3D numpy array

    """
    x = _arange(x_range[0], x_range[1], h)
    y = _arange(y_range[0], y_range[1], h)
    z = _arange(z_range[0], z_range[1], h)

    if component is None:
        field_grid = np.empty([x.size, y.size, z.size], dtype=object)
    else:
        field_grid = np.zeros([x.size, y.size, z.size], dtype=float)

    for charge in charge_objs:
        for (i, j, k), _ in np.ndenumerate(field_grid):
            if field_grid[i][j][k] is None:
                if component is None:
                    field_grid[i][j][k] = charge.field(
                        [x[i], y[j], z[k]], type=type
                    )
            else:
                if component is None:
                    field_grid[i][j][k] += charge.field(
                        [x[i], y[j], z[k]], type=type
                    )
                elif component == "x":
                    field_grid[i][j][k] += charge.field(
                        [x[i], y[j], z[k]], type=type
                    )[0]
                elif component == "y":
                    field_grid[i][j][k] += charge.field(
                        [x[i], y[j], z[k]], type=type
                    )[1]
                elif component == "z":
                    field_grid[i][j][k] += charge.field(
                        [x[i], y[j], z[k]], type=type
                    )[2]

    return field_grid


def _arange(_min, _max, _step):
    """ Alternative to numpy's arange that handles well floating point steps
   Also happens to give "rounder" decimals than np.arange =)
   """
    return np.linspace(_min, _max, 1 + int(np.rint((_max - _min) / _step)))
