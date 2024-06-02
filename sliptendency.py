"""
Please read the short project description before using the script
https://github.com/giboul/sliptendency/blob/main/README.md
"""
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
import numpy as np


# ====================== Numerical ======================
alpha, beta = np.meshgrid(np.linspace(0, 2*np.pi, num=50, endpoint=True),
                          np.linspace(0, np.pi, num=50, endpoint=True))
positions = np.array((np.sin(beta)*np.cos(alpha),  # X
                      np.sin(beta)*np.sin(alpha),  # Y
                      np.cos(beta)))  # Z


def stress_tendancy(stress_matrix):
    """Compute the stress tendency distrbution over a sphere from a 3x3 stress matrix"""

    t = np.einsum('ij,ikl->jkl', stress_matrix, positions)  # t = stress_matrix @ n
    tnorm = np.linalg.norm(t, axis=0)  # |t|

    cos = (t*positions).sum(axis=0)/tnorm  # cos(phi) = dot(t, n)/|t|
    cos = np.minimum(np.maximum(cos, 0), 1)  # Bounding rounding errors
    sin = np.sqrt(1-cos**2)  # sin(phi) = sqrt(1-cos(phi)^2)

    sn = tnorm * cos  # sigma_n = |t|*cos(phi)
    tn = tnorm * sin  # tau_n = |t|*sin(phi)

    Ts = np.full_like(sn, float('nan'))
    mask = ~ np.isclose(tnorm, 0)
    Ts[mask] = tn[mask]/sn[mask]  # Ts = tau_n / sigma_n

    return sn, tn, Ts

# ====================== Visualization ======================
cmap = plt.cm.inferno

fig = plt.figure()
gs = GridSpec(15, 2, width_ratios=(1,3), figure=fig)

axsn = fig.add_subplot(gs[5:9, 0], projection='3d', box_aspect=(1, 1, 1))
axsn.set_xlabel(r"$x$")
axsn.set_ylabel(r"$y$")
axsn.set_zlabel(r"$z$")
axsn.set_title(r"Normal stress $\sigma_n$")

axtn = fig.add_subplot(gs[11:, 0], projection='3d', box_aspect=(1, 1, 1))
axtn.set_xlabel(r"$x$")
axtn.set_ylabel(r"$y$")
axtn.set_zlabel(r"$z$")
axtn.set_title(r"Shear stress $\tau_n$")

axts = fig.add_subplot(gs[:, 1], projection='3d', box_aspect=(1, 1, 1))
axts.set_xlabel(r"$x$")
axts.set_ylabel(r"$y$")
axts.set_zlabel(r"$z$")
axts.set_title(r"Slip-tendency $T_s$")


def normalize(a: np.ndarray, min=None, max=None):  # => a in [0;1]
    if min is None:
        min = a.min()
    if max is None:
        max = a.max()
    if np.isclose(min, max):
        return np.full_like(a, min)
    return (a - min) / (max - min)


def replot(val):
    """Compute new Ts field and update 3D sphere color plot"""
    stresses = np.array([s.val for s in sliders])
    stress_mat = np.diag(stresses)  # stress tensor
    s3, s2, s1 = np.sort(stresses)
    sn, tn, Ts = stress_tendancy(stress_mat)   # Slip tendency

    axts.set_title(
        "Slip-tendency $T_s$ "
        rf"($\phi = {(s2-s3)/(s1-s3):.2f}$)"
    )

    for surf, cbar, v in zip(surfaces, cbars, (sn, tn, Ts)):
        surf.set_color(cmap(normalize(v)[:-1, :-1].flatten()))
        cbar.set_ticklabels((f"{v.min():.5n}", f"{v.max():.5n}"))


# Defining sliders
sliders = []
for i, direction in enumerate("xyz"):
    axs = fig.add_subplot(gs[i, 0])
    axs.axis('off')
    s = Slider(axs, label=rf"$\sigma_{direction}$",
               valmin=1e-10, valmax=100, valinit=0.)
    s.on_changed(replot)
    sliders.append(s)
# Initialize surface plots
surfaces = [
    ax.plot_surface(*positions,
                    rstride=1,
                    cstride=1,
                    facecolors=cmap(np.full_like(alpha, 0.)),
                    cmap=cmap)
    for ax in (axsn, axtn, axts)
]
# Colorbars
cbars = []
for ax, surf in zip((axsn, axtn, axts), surfaces):
    cbar = plt.colorbar(surf, ax=ax)
    cbar.set_ticks((0, 1))
    cbars.append(cbar)

plt.show()
