import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def excluding_mesh(x, y, nx=30, ny=30):
    """
    Construct a grid of points, that are some distance away from points (x,
    """

    dx = x.ptp() / nx
    dy = y.ptp() / ny

    xp, yp = np.mgrid[x.min() - 2 * dx: x.max() + 2 * dx: (nx + 2) * 1j,
                      y.min() - 2 * dy: y.max() + 2 * dy: (ny + 2) * 1j]
    xp = xp.ravel()
    yp = yp.ravel()

    # Use KDTree to answer the question: "which point of set (x,y) is the
    # nearest neighbors of those in (xp, yp)"
    tree = KDTree(np.c_[x, y])
    dist, j = tree.query(np.c_[xp, yp], k=1)

    # Select points sufficiently far away
    m = (dist > np.hypot(dx, dy))
    return xp[m], yp[m]

# Prepare fake data points



def main():
    # Some input data
    t = 1.2 * np.pi * np.random.rand(3000)
    r = 1 + np.random.rand(t.size)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = x ** 2 - y ** 2
    xi, yi = np.ogrid[-3:3:350j, -3:3:350j]
    zi = griddata((xi, yi), z, (xi, yi),
    # -- Way 1: seed input with nan
    xp, yp = excluding_mesh(x, y, nx=35, ny=35)
    zp = np.nan + np.zeros_like(xp)

    # Grid the data plus fake data points
    xi, yi = np.ogrid[-3:3:350j, -3:3:350j]
    zi = griddata((np.r_[x, xp], np.r_[y, yp]), np.r_[z, zp], (xi, yi),
                  method='linear')
    plt.imshow(zi)
    plt.show()
    pass


if __name__ == '__main__':
    main()
