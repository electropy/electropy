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
    x = np.arange(x_range[0], x_range[1] + h, h)
    y = np.arange(y_range[0], y_range[1] + h, h)
    z = np.arange(z_range[0], z_range[1] + h, h)

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
    component=None,
):
    """Calculate field in a volume

    Args:
        charge_objs: list of Charge objects
        x_range, y_range, z_range: [min, max] distances of volume.
                                    units: meters
        h: spacing between array elements. units: meters
        component: 'x', 'y', 'z', or None (default)

    Return: 3D numpy array

    """
    x = np.arange(x_range[0], x_range[1] + h, h)
    y = np.arange(y_range[0], y_range[1] + h, h)
    z = np.arange(z_range[0], z_range[1] + h, h)

    if component is None:
        field_grid = np.empty([x.size, y.size, z.size], dtype=object)
    else:
        field_grid = np.zeros([x.size, y.size, z.size], dtype=float)

    for charge in charge_objs:
        for (i, j, k), _ in np.ndenumerate(field_grid):
            if field_grid[i][j][k] is None:
                if component is None:
                    field_grid[i][j][k] = charge.field(
                        [x[i], y[j], z[k]], type="analytical"
                    )
            else:
                if component is None:
                    field_grid[i][j][k] += charge.field(
                        [x[i], y[j], z[k]], type="analytical"
                    )
                elif component == "x":
                    field_grid[i][j][k] += charge.field(
                        [x[i], y[j], z[k]], type="analytical"
                    )[0]
                elif component == "y":
                    field_grid[i][j][k] += charge.field(
                        [x[i], y[j], z[k]], type="analytical"
                    )[1]
                elif component == "z":
                    field_grid[i][j][k] += charge.field(
                        [x[i], y[j], z[k]], type="analytical"
                    )[2]

    return field_grid
