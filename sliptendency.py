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
    Ts = tn / sn

    return sn, tn, Ts

# ====================== Visualization ======================
scatter_plot = False
cmap = plt.cm.inferno

# On slider change command
def replot(args):
    """Compute new Ts field and update 3D sphere color plot"""
    stresses = np.array([s.val for s in sliders])
    stress_mat = np.diag(stresses)  # stress tensor
    s3, s2, s1 = np.sort(stresses)
    sn, tn, Ts = stress_tendancy(stress_mat)   # Slip tendency

    bounds = np.hstack((sn, tn))
    bounds = dict(min=bounds.min(), max=bounds.max())

    sn = normalize(sn, min=bounds["min"], max=bounds["max"])
    tn = normalize(tn, min=bounds["min"], max=bounds["max"])
    Ts = normalize(Ts)

    for variable, axis, ball, title in zip(
        (sn, tn, Ts), (axsn, axtn, axts), ballpoints, (
                r"Normal stress $\sigma_n$",
                r"Normal shear stress $\tau_n$",
                "Slip-tendency $T_s/T_{s,max}$"rf" ($\phi = {(s2-s3)/(s1-s3):.2f}$)"
            )
    ):

        if scatter_plot:
            ball.set_color(cmap(variable.flatten()))
        else:
            plt.sca(axis)
            plt.cla()
            axis.set_title(title)
            surf = axis.plot_surface(*positions, facecolors=cmap(variable),
                                     rstride=1, cstride=1)
            axis.set_xlabel(r"$x$")
            axis.set_ylabel(r"$y$")
            axis.set_zlabel(r"$z$")

    plt.tight_layout()


def normalize(a: np.ndarray, min=None, max=None):  # => a in [0;1]
    if min is None:
        min = a.min()
    if max is None:
        max = a.max()
    return (a - min) / (max - min)


# Figure definition
fig = plt.figure(layout="tight")
gs = GridSpec(9, 9, figure=fig)

axsn = fig.add_subplot(gs[3:6, :2], projection='3d', box_aspect=(1, 1, 1))
axsn.set_xlabel(r"$x$")
axsn.set_ylabel(r"$y$")
axsn.set_zlabel(r"$z$")
axsn.set_title(r"Normal stress $\sigma_n$")

axtn = fig.add_subplot(gs[6:, :2], projection='3d', box_aspect=(1, 1, 1))
axtn.set_xlabel(r"$x$")
axtn.set_ylabel(r"$y$")
axtn.set_zlabel(r"$z$")
axtn.set_title(r"Shear stress $\tau_n$")

axts = fig.add_subplot(gs[:, 2:], projection='3d', box_aspect=(1, 1, 1))
axts.set_xlabel(r"$x$")
axts.set_ylabel(r"$y$")
axts.set_zlabel(r"$z$")
axts.set_title(r"Slip-tendency $T_s$")

# Defining sliders
sliders = []
for i, direction in enumerate("xyz"):
    axs = fig.add_subplot(gs[i, :2])
    axs.axis('off')
    s = Slider(axs, label=rf"$\sigma_{direction}$",
               valmin=1e-10, valmax=100, valinit=0.)
    s.on_changed(replot)
    sliders.append(s)

# Initial plot
if scatter_plot:
    ballsn = axsn.scatter(*positions, alpha=1, s=100)
    balltn = axtn.scatter(*positions, alpha=1, s=100)
    ballts = axts.scatter(*positions, alpha=1, s=100)
    ballpoints = [ballsn, balltn, ballts]
else:
    ballpoints = [None for s in sliders]
plt.show()
