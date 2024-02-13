import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import scipy as sp
import einops as eo
from itertools import repeat
import magpylib as magpy

folders = [
    "MLX02062024_D32N52_test1"
    "MLX02062024_D32N52_test2"
    "MLX02072024_D101N52_test1"
    "MLX02072024_D101N52_test2"
    "MLX02072024_D32N52_test1"
    "MLX02072024_D32N52_test2"
    "MLX02072024_D32N52_test3"
    "MLX02072024_D32N52_test4"
    "MLX02102024_D101N52_12mm"
    "MLX02102024_D101N52_z12mm"
    "MLX02120204_D201N52_z10mm"
    "MLX02122024_D201N52_z15mm"
    "MLX02122024_D201N52_z20mm"
    "MLX02122024_D21BN52_z10mm"
    "MLX02122024_D21BN52_z10mm1"
    "MLX02122024_D21BN52_z15mm"
    "MLX02122024_D21BN52_z20mm"
    "MLX02122024_D301N52_z10mm"
    "MLX02122024_D301N52_z15mm"
]
diameters = [3, 3, 1, 1, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3]
heights = [2, 2, 0.5, 0.5, 2, 2, 2, 2, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 0.5, 0.5]
mag_zs = [10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 10, 15, 20, 10, 10, 15, 20, 10, 15]

folder = "MLX02122024_D21BN52_z10mm"
# folder = "MLX02122024_D301N52_z10mm"
B0 = -1480  # mT
# B0 = 1480
diameter = 25.4 * 2 / 16
height = 25.4 * 1 / 16
mag_z = 10
field_func = "dipole"
# field_func = "cyl"


class InFile(object):
    def __init__(self, infile):
        self.infile = open(infile)

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def read(self, *args, **kwargs):
        return self.__next__()

    def next(self):
        try:
            line: str = self.infile.readline()
            line = (
                line.replace('"', "").replace("[", "").replace("]", "")
            )  # do some fixing
            return line
        except:
            self.infile.close()
            raise StopIteration


def B_dipole(position, rotation, M0, shape):
    # position = -position
    R = np.sqrt(np.sum(position**2, axis=1))
    B = (M0 * (shape[0]) ** 2 * shape[1] / (16)) * (
        (
            3
            * position
            / R[:, np.newaxis] ** 5
            * (eo.einsum(position, rotation, "sensor dim,  dim -> sensor"))[
                :, np.newaxis
            ]
        )
        - rotation[np.newaxis, :] / (R[:, np.newaxis] ** 3)
    )
    return B


def getField_dipole(x, positions, M0, shape):
    # magnetization=x[5]
    # magnetization=1210
    position = x[:3]
    axis = x[3:6]
    # axis = np.array([0, 0, 1])
    # phi = x[3]
    # theta = x[4]
    # axis = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return B_dipole(positions - position, axis, M0, shape)


def getField_dipole_fixed(x, positions, M0, shape):
    # magnetization=x[5]
    # magnetization=1210
    position = x[:3]
    axis = x[3:6]
    # axis = np.array([0.0, 0.0, 1.0])
    # phi = x[3]
    # theta = x[4]
    # axis = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return B_dipole(positions - position, axis, M0, shape)


def getField_cyl(x, positions, M0, shape):
    position = x[:3]
    rot = x[3:6]
    z = np.array([0, 0, 1])
    angle = np.arccos(np.dot(z, rot))
    axis = np.cross(z, rot)
    orientation = sp.spatial.transform.Rotation.from_rotvec(angle * axis)

    B = magpy.getB(
        sources="Cylinder",
        observers=positions,
        dimension=shape,
        magnetization=np.array([0, 0, M0]),
        position=position,
        orientation=orientation,
    )
    return np.array(B)


def cost_dipole(x, B, positions, M0, shape):
    diff = getField_dipole(x, positions, M0, shape) - B
    return np.sum((diff) ** 2)


def cost(x, B, positions, M0, shape, getField):
    diff = getField(x, positions, M0, shape) - B
    return np.sum((diff) ** 2)


def minimize(x0, args):
    print("Starting mimimization")
    # x0 = [x, y, z, th_x, th_y, th_z]
    cons = [{"type": "eq", "fun": lambda x: x[3] ** 2 + x[4] ** 2 + x[5] ** 2 - 1}]
    # cons = []
    bounds = [(-400, 400), (-400, 400), (-200, 200), (-1, 1), (-1, 1), (-1, 1)]
    res = sp.optimize.minimize(
        fun=cost, x0=x0, args=args, tol=1e-10000, constraints=cons, bounds=bounds
    ).x  # , options={'maxiter': 1000}).x
    print(f"Finished mimimization at {res}")
    return res


def read_data(folder):
    dfs = [
        pd.read_csv(
            InFile(f"{folder}/sensor_{i}_averages.csv"),
            names=["Bx", "By", "Bz"],
            sep=",",
            skiprows=1,
        )
        for i in range(1, 17)
    ]
    data = np.array(dfs)
    data = eo.rearrange(data, "s t d -> t s d")
    return data

sensor_actual = np.array(
    [
        [0, 0, 0],
        [-4.5, 0, 0],
        [-9.0, 0, 0],
        [-13.5, 0, 0],
        [0, 4.5, 0],
        [-4.5, 4.5, 0],
        [-9.0, 4.5, 0],
        [-13.5, 4.5, 0],
        [0, 9.0, 0],
        [-4.5, 9.0, 0],
        [-9.0, 9.0, 0],
        [-13.5, 9.0, 0],
        [0, 13.5, 0],
        [-4.5, 13.5, 0],
        [-9.0, 13.5, 0],
        [-13.5, 13.5, 0],
    ]
)


def getResults(data, sensor_actual, diameter, height, field_func):
    results = []
    field_funcs = {"cyl": getField_cyl, "dipole": getField_dipole}
    field_func = field_funcs[field_func]
    for d in data:
        args = (d, sensor_actual, B0, (diameter, height), field_func)
        x0 = np.array([0, 0, 30, 0, 0, 1])
        result = minimize(x0, args)
        results.append(result)
    results = np.array(results)
    return results


def getDisplacements(results):
    displacement = np.sqrt(np.sum((results[:, :3] - results[0, :3]) ** 2, axis=1))
    return displacement


def save_plots(folder, filename, displacement, mag_z):
    x, y, z = np.linspace(0, 5, 6), np.linspace(0, 5, 6), np.array([mag_z])
    xx, yy, zz = np.meshgrid(x, y, z)
    grid = np.array([xx, yy, zz])
    points = eo.rearrange(grid, "v x y z -> (y x z) v")
    actual_displacement = np.sqrt(np.sum((points[:, :3] - points[0, :3]) ** 2, axis=1))
    vals = np.linspace(0, 9, 1000)
    plt.plot(actual_displacement, displacement, ".")
    m, b, r, p, err = sp.stats.linregress(actual_displacement, displacement)
    xxx = np.linspace(np.min(actual_displacement), np.max(actual_displacement), 1000)
    yyy = m * xxx + b
    max_err = np.sqrt(np.max((actual_displacement - displacement) ** 2))
    avg_err = np.sqrt(np.median((actual_displacement - displacement) ** 2))
    print(1 - r)
    print(max_err)
    print(avg_err)
    print(m, b)

    plt.plot(xxx, yyy, "-")
    plt.annotate(f"1-r^2 = {1-r:.3e}", (0, 7))
    plt.annotate(f"Max err: {max_err:.3f}", (0, 6.5))
    plt.annotate(f"Avg err: {avg_err:.3f}", (0, 6))
    plt.annotate(
        f"A fitted line(not related to above stats): {m:.3f}x + {b:.3f}", (0, 5.5)
    )
    # plt.plot(vals, vals, "-")
    plt.savefig(f"{folder}/{filename}.png")
    plt.close()


folders = [
    "MLX02062024_D32N52_test1",
    "MLX02062024_D32N52_test2",
    "MLX02072024_D101N52_test1",
    "MLX02072024_D101N52_test2",
    "MLX02072024_D32N52_test1",
    "MLX02072024_D32N52_test2",
    "MLX02072024_D32N52_test3",
    "MLX02072024_D32N52_test4",
    "MLX02102024_D101N52_12mm",
    "MLX02102024_D101N52_z12mm",
    "MLX02120204_D201N52_z10mm",
    "MLX02122024_D201N52_z15mm",
    "MLX02122024_D201N52_z20mm",
    "MLX02122024_D21BN52_z10mm",
    "MLX02122024_D21BN52_z10mm1",
    "MLX02122024_D21BN52_z15mm",
    "MLX02122024_D21BN52_z20mm",
    "MLX02122024_D301N52_z10mm",
    "MLX02122024_D301N52_z15mm",
]
diameters = [3, 3, 1, 1, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3]
heights = [2, 2, 0.5, 0.5, 2, 2, 2, 2, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 0.5, 0.5]
mag_zs = [10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 10, 15, 20, 10, 10, 15, 20, 10, 15]

field_func = "cyl"
# field_func = "cyl"
print(list(map(len, [folders, diameters, heights, mag_zs])))

for folder, diameter, height, mag_z in zip(folders, diameters, heights, mag_zs):
    data = read_data(folder)
    results = getResults(data, sensor_actual, diameter, height, field_func)
    displacement = getDisplacements(results)
    filename = "cyl"
    save_plots(folder, filename, displacement, mag_z)


scaled_data = 1 * data
norm = np.sum(data**2, axis=2)

X, Y = (sensor_actual[:, 0], sensor_actual[:, 1])
sample = scaled_data[0]
U, V = (sample[:, 0], sample[:, 1])

fig, ax = plt.subplots(1, 1)
Q = ax.quiver(X, Y, U, V, norm[0], pivot="mid", color="r", units="xy", scale=0.05)
P = ax.scatter(results[0, 0], results[0, 1])

# ax.set_xlim(-1, 7)
# ax.set_ylim(-1, 7)


def update_quiver(num, Q, P, X, Y):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """

    sample = scaled_data[num]
    norm = np.sum(data**2, axis=2)
    U, V = (sample[:, 0], sample[:, 1])
    xy = results[num, :2]

    Q.set_UVC(U, V, norm[num])
    P.set_offsets(xy)

    return (
        Q,
        P,
    )


# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(
    fig,
    update_quiver,
    fargs=(Q, P, X, Y),
    interval=300,
    blit=True,
    frames=data.shape[0],
)
# plt.show()
