"""
Fonctions de bas niveau : opérateurs de dérivation et intégration pour des fonctions périodiques
Oui, il n'y a que ça !
"""

import numpy as np
import scipy

def Fourier_deriv(y, a = 0., b = 2.0*np.pi, real = True, k=None, order = 1, axis = None):
    """"
    calcul de la dérivée d'ordre n (paramètre order) d'une fonction périodique (1D ou 2D, définie sur l'intervalle [a, b],
    , k (vecteur d'onde) peut etre fourni, axis = (None, 'x', 'y'). 'x' et 'y' correspondent aux axes 0 et 1, dans le cas où la grille 2D est définie      avec np.meshgrid(..., indexing = 'ij')
    """
    
    if axis is None:
        axis = -1
        N = y.size
    elif axis == "x":
        axis = 0
        N = y.shape[0]
    elif axis == "y":
        axis = 1
        N = y.shape[1]
    else:
        raise ValueError(f"axis must be in ('x', 'y', None), current value = {axis}")

    if k is None : 
        k = 2.0*np.pi*scipy.fft.fftfreq(N, d=(b-a)/N)
        if axis == 0:#On transforme k en vecteur colonne
            k = k[..., None]
            
    y_hat = scipy.fft.fft(y, axis = axis)

    k_exp = (1.j*k)**order
    y_hat_p =  k_exp * y_hat

    y_p = scipy.fft.ifft(y_hat_p, axis = axis)
    if real : 
        y_p = y_p.real

    coeff_interval = (2.0*np.pi/(b-a))**order
    return coeff_interval*y_p


def Fourier_quad(y, dx, axis = None):
    """
    Intégration d'une fonction périodique 1D par méthode des trapèzes ;  dx : pas spatial, axis = ('x', 'y',None).
    Axis = 'x' -> axis = 1 (lignes), Axis = 'y' -> axis = 0 (colonnes). Adapté si fonction 2D avec grille générée avec np.meshgrid(..., indexing='ij')
    """
    if axis == 'x':
        axis = 1
    elif axis == 'y':
        axis = 0
    elif axis is None:
        pass
    else : 
        raise ValueError(f"axis must be in ('x', 'y', None), current value {axis}")
    return np.sum(y, axis=axis)*dx

def Fourier_cumsum(y, dx):
    """"Intégration d'une fonction périodique 1D, le long d'un domaine ; dx: pas spatial
        Attention : précision faible (10^-7 pour N=128). Utilisé uniquement pour reconstituer psi à partir de u et v
    """
    I = scipy.integrate.cumulative_simpson(y, dx=dx, initial=0.0)
    return I

    