"""
2D magnetic field lines in the **Z–X plane** where:
- **X** = axial direction (along the magnet axis)
- **Z** = radial direction (represents symmetry in the transverse plane)

So we plot only Z >= 0 (half-plane) because of cylindrical symmetry.

Your inputs:
    M0 = 1480 mT  (treated as polarization/Br = 1.480 T)
    shape = [[radius_mm, height_mm], [radius_mm, height_mm]]

Internals:
- Axially magnetized cylinder approximated as a stack of circular current loops.
- Fast elliptic-integral loop field if SciPy is available; otherwise numeric Biot–Savart fallback.
"""

import numpy as np
import matplotlib.pyplot as plt

MU0 = 4e-7 * np.pi

# ---------------------- YOUR PARAMETERS ----------------------
M0_mT = 1480  # mT
shape = [
    [5 * 25.4 / 16, 25.4 / 8],      # radius_mm, height_mm
    [5 * 25.4 / 16, 2 * 25.4 / 8],  # radius_mm, height_mm
]
shape_index = 0  # choose 0 or 1
# ------------------------------------------------------------

# ---------------------- PLOT / GRID SETTINGS ----------------------
# Axial coordinate (X) range in mm
X_min_mm, X_max_mm = -60.0, 60.0

# Radial coordinate (Z) range in mm (Z >= 0 for symmetry)
Z_max_mm = 40.0

stream_density = 2.0
clip_B_T = 2.0      # clip |B| for prettier streamlines (Tesla)
n_loops = 80        # loop-stack resolution along magnet length
# ------------------------------------------------------------

# ---- Try fast elliptic-integral loop field ----
try:
    from scipy.special import ellipk, ellipe
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# If SciPy isn't available, reduce grid size (numeric fallback is slower)
if HAVE_SCIPY:
    nZ, nX = 220, 280
    NPHI_NUMERIC = 0
else:
    nZ, nX = 120, 170
    NPHI_NUMERIC = 220


def loop_field_rz_elliptic(r, z, a_loop, I):
    """
    Field from a circular current loop (radius a_loop, current I),
    in cylindrical coords (r,z) relative to loop center.
    Returns (Br, Bz).
    """
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)

    Br = np.zeros_like(r)
    Bz = np.zeros_like(r)

    ap = a_loop + r
    am = a_loop - r
    Q = ap * ap + z * z
    P = am * am + z * z

    with np.errstate(divide="ignore", invalid="ignore"):
        m = 4.0 * a_loop * r / Q  # m = k^2

    K = ellipk(m)
    E = ellipe(m)
    denom = np.sqrt(Q)

    off_axis = r > 1e-12
    if np.any(off_axis):
        r0 = r[off_axis]
        z0 = z[off_axis]
        P0 = P[off_axis]
        denom0 = denom[off_axis]
        K0 = K[off_axis]
        E0 = E[off_axis]

        Br[off_axis] = (MU0 * I * z0) / (2.0 * np.pi * r0 * denom0) * (
            -K0 + (a_loop * a_loop + r0 * r0 + z0 * z0) / P0 * E0
        )

        Bz[off_axis] = (MU0 * I) / (2.0 * np.pi * denom0) * (
            K0 + (a_loop * a_loop - r0 * r0 - z0 * z0) / P0 * E0
        )

    # On-axis: Br=0, Bz simple
    on_axis = ~off_axis
    if np.any(on_axis):
        z1 = z[on_axis]
        Bz[on_axis] = MU0 * I * a_loop * a_loop / (2.0 * (a_loop * a_loop + z1 * z1) ** 1.5)

    return Br, Bz


def loop_field_xyz_numeric(x, y, z, a_loop, I, nphi=240):
    """
    Numeric Biot–Savart for a circular loop in x-y plane centered at origin.
    Returns (Bx, By, Bz).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    dphi = phi[1] - phi[0]

    xw = a_loop * np.cos(phi)
    yw = a_loop * np.sin(phi)

    dlx = -a_loop * np.sin(phi) * dphi
    dly =  a_loop * np.cos(phi) * dphi

    X = x[..., None]
    Y = y[..., None]
    Z = z[..., None]

    rx = X - xw[None, ...]
    ry = Y - yw[None, ...]
    rz = Z

    R3 = (rx * rx + ry * ry + rz * rz) ** 1.5 + 1e-30

    # dl × r
    cx = dly * rz
    cy = -dlx * rz
    cz = dlx * ry - dly * rx

    pref = MU0 * I / (4.0 * np.pi)
    Bx = pref * np.sum(cx / R3, axis=-1)
    By = pref * np.sum(cy / R3, axis=-1)
    Bz = pref * np.sum(cz / R3, axis=-1)
    return Bx, By, Bz


def loop_field_rz(r, z, a_loop, I):
    """Fast elliptic if SciPy is available, else numeric loop integration."""
    if HAVE_SCIPY:
        return loop_field_rz_elliptic(r, z, a_loop, I)
    # numeric: evaluate in x-z plane (y=0), Br == Bx for r>=0
    Bx, _, Bz = loop_field_xyz_numeric(r, 0.0, z, a_loop, I, nphi=NPHI_NUMERIC)
    return Bx, Bz


def cylinder_field_ZX(Z_rad_m, X_ax_m, radius_m, height_m, Br_T, nloops):
    """
    Field in Z–X plane:
      Z = radial coordinate r (>=0)
      X = axial coordinate

    Returns:
      BZ (radial component), BX (axial component)
    """
    # Treat Br_T as polarization J (Tesla). Then M = J/mu0 (A/m).
    M_Apm = Br_T / MU0

    r = Z_rad_m
    x = X_ax_m

    # Stack loops along axial direction X
    x_centers = np.linspace(-height_m / 2.0, height_m / 2.0, nloops)
    dx = x_centers[1] - x_centers[0] if nloops > 1 else height_m
    I_loop = M_Apm * dx  # A

    Br_total = np.zeros_like(r)
    Bx_total = np.zeros_like(r)  # axial

    for x0 in x_centers:
        Br_i, Bax_i = loop_field_rz(r, x - x0, radius_m, I_loop)  # (Br, B_axial)
        Br_total += Br_i
        Bx_total += Bax_i

    # In this plane, radial component aligns with +Z
    BZ = Br_total
    BX = Bx_total
    return BZ, BX


def main():
    radius_mm, height_mm = shape[shape_index]
    radius_m = radius_mm * 1e-3
    height_m = height_mm * 1e-3
    Br_T = M0_mT * 1e-3  # 1480 mT -> 1.480 T

    # 1D axes (mm) for streamplot: x-axis=Z, y-axis=X
    Z_mm = np.linspace(0.0, Z_max_mm, nZ)             # radial (>=0)
    X_mm = np.linspace(X_min_mm, X_max_mm, nX)        # axial
    Zg_mm, Xg_mm = np.meshgrid(Z_mm, X_mm)            # shape (nX, nZ)

    # Convert to meters
    Zg_m = Zg_mm * 1e-3
    Xg_m = Xg_mm * 1e-3

    # Field (components along plot axes)
    BZ, BX = cylinder_field_ZX(Zg_m, Xg_m, radius_m, height_m, Br_T, n_loops)

    # Clip for nicer streamlines
    Bmag = np.sqrt(BZ * BZ + BX * BX)
    if clip_B_T is not None and clip_B_T > 0:
        scale = np.minimum(1.0, clip_B_T / (Bmag + 1e-30))
        BZ *= scale
        BX *= scale

    fig, ax = plt.subplots(figsize=(8, 7))

    # streamplot expects U,V as components along x,y axes => (Z,X)
    ax.streamplot(
        Z_mm, X_mm, BZ, BX,
        density=stream_density,
        arrowsize=1.0,
        minlength=0.05,
    )

    # Draw magnet cross-section in this half-plane:
    # Z from 0..radius, X from -height/2..+height/2
    rect_Z = [0, radius_mm, radius_mm, 0, 0]
    rect_X = [-height_mm / 2, -height_mm / 2, height_mm / 2, height_mm / 2, -height_mm / 2]
    ax.plot(rect_Z, rect_X, linewidth=2)

    ax.set_title(
        f"Field Lines in Z–X Plane (Z radial, X axial) | shape_index={shape_index}\n"
        f"radius={radius_mm:.4g} mm, height={height_mm:.4g} mm, M0={M0_mT} mT (Br={Br_T:.3f} T)\n"
        f"Loop approx: {n_loops} loops | {'elliptic (SciPy)' if HAVE_SCIPY else 'numeric (no SciPy)'}"
    )
    ax.set_xlabel("Z (mm)  [radial]")
    ax.set_ylabel("X (mm)  [axial]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()