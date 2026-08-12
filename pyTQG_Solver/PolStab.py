"""
Ensemble de fonctions permettant le calcul des zones de stabilité de quelque schémas temporels explicites

"""

import numpy as np
import matplotlib.pyplot as plt


def StabEuler(zz):
    return np.abs(1.0+zz) <= 1.0

def StabRK2(zz):
    return np.abs(1.0+zz+(zz**2)/2.0) <= 1.0

def StabRK4(zz):
    pol_RK4 = 1.0 + zz + 0.5*zz**2 + (zz**3)/6.0 + (zz**4)/24.0
    return np.abs(pol_RK4) <= 1.0

def StabHeun(zz):
    "Pas sur, mais c'esr quasi pareil que RK2 (de toute façon, je l'utilise pas)"
    return np.abs(1.0+zz+(zz**2)/2.0) <= 1.0

def stab_LF(zz):
    "cf. Shampine2009. On suppose ici que le premier pas est effectué avec un schéma d'Euler"
    zeta1 = zz+np.sqrt(1.0+zz**2)
    zeta2 = zz-np.sqrt(1.0+zz**2)
    pol_stab = zeta1+(1.0+zz)*zeta2
    return np.abs(pol_stab) <= 1.0

def stability_diagram(stab_func, ax=None, Re_inf=-4.0, Re_sup=4.0, Im_inf = -4.0, Im_sup=4.0, N=200, title = None):
    "Petit utilitaire permettant de tracer le diagramme de stabilité dans le plan complexe"
    Re = np.linspace(Re_inf, Re_sup, N)
    Im = np.linspace(Im_inf, Im_sup, N)
    RR, II = np.meshgrid(Re, Im)
    z=RR+1.0j*II
    z_stab = stab_func(z)
    #print(z[z_stab])
    if ax is None:
        ax = plt.axes()
    ax.set_facecolor('mistyrose')
    ax.contourf(RR, II, z_stab, levels = [0.5, 1.5], colors = 'palegreen')
    ax.contour(RR, II, z_stab,
               levels=[0.5],
               colors="black",
               linewidths=2)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\operatorname{Re}(z)$")
    ax.set_ylabel(r"$\operatorname{Im}(z)$")
    ax.grid(True)
    if title is not None:
        ax.set_title(title)
    return ax