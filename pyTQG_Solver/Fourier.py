"""
Fonctions de bas niveau : opérateurs de dérivation et intégration pour des fonctions périodiques
Oui, il n'y a que ça !
"""

import numpy as np
import scipy

def Fourier_deriv(y, a = 0., b = 2.0*np.pi, real = True, order = 1, k=None):
    """"
    calcul de la dérivée d'ordre n (paramètre order) d'une fonction périodique 1D définie sur l'intervalle [a, b]
    k : nombre d'onde (peut etre passé en paramètre)
    """
    if k is None:
        k = 2.0*np.pi * scipy.fft.fftfreq(y.size, d=((b-a)/y.size))
    y_hat = scipy.fft.fft(y)
    
    y_hat_p = (1.j*k)**order * y_hat
    y_p = scipy.fft.ifft(y_hat_p)
    if real : 
        y_p = y_p.real

    coeff_interval = (2.0*np.pi/(b-a))**order
    return coeff_interval*y_p

def Fourier_quad(y, dx, axis =0):
    "Intégration d'une fonction périodique 1D par méthode des trapèzes ;  dx : pas spatial"
    return np.sum(y)*dx

def Fourier_cumsum(y, dx):
    """"Intégration d'une fonction périodique 1D, le long d'un domaine ; dx: pas spatial
        Attention : précision faible (10^-7 pour N=128). Utilisé uniquement pour reconstituer psi à partir de u et v
    """
    I = scipy.integrate.cumulative_simpson(y, dx=dx, initial=0.0)
    return I