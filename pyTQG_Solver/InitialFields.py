"""
On rassemble ici quelques champs de vitesse/flottabilité pouvant servir de condition initiales 
Pour le moment, j'ai juste mis des fonctions pour les jets ...
"""

import numpy as np

def BickleyJet(y, gamma):
    return (np.cosh(y/gamma))**(-2.0)

def UniformJet(y, U0):
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    return U0*np.ones_like(y)

def ConstantGrad(y, alpha, y_inf = -1.0, y_sup = 1.0, b_inf = 0.0, b_sup = 1.0):
    "Créé un gradient constant selon y"
    if y_sup >= y_inf:
        raise ValueError(f"y_inf must be < y_sup: y_inf = {y_inf}, y_sup = {y_sup}")
    if np.sign(b_sup-b_inf) != np.sign(alpha):
        raise ValueError(f"(b_sup-b_inf) must have the same sign than alpha")#ça va c'est clairement exprimé en anglais ?
    intercept_1 = (b_sup - alpha *y_sup)
    intercept_2 = (b_inf - alpha *y_inf)
    return alpha * y + (intercept_1+intercept_2)/2.0
    