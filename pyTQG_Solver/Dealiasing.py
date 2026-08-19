"""
Quelque fonctions liées au déaliasing : 
-Règle des 2/3 (Orszag et al) pour Fourier
-Filtre exponentiel pour la base de Chebyshev. Le filtre est appliqué sur les coefficients, calculés par DCT

Ces fonctions ne s'appliquent que sur des vecteurs 1D (comme toutes les fonctions bas-niveau)
"""
import numpy as np


def Fourier_dealias(v, dx, coeff_dealias=2.0/3.0, real = True):
    kx = 2.0*np.pi*scipy.fft.fftfreq(v.size, d=dx)
    v_hat = scipy.fft.fft(v)
    v_hat_dealias = np.where(np.abs(kx) < coeff_dealias*np.abs(kx).max(), v_hat.copy(), 0.0)
    v_dealias = scipy.fft.ifft(v_hat_dealias)

    if real : 
        return v_dealias.real
    return v_dealias

def exp_filter_DCT(v, alpha=100.0, p=8.0, interior = True):
    if type(v) !=np.ndarray:
        v = np.array(v)
    N = v.size
    if not interior:
        v_fil = v.copy()
        k = np.arange(0, N)
        V_DCT = scipy.fft.dct(v, type=1)
    else:
        v_fil = np.zeros_like(v)
        v_fil[0] = v[0]
        v_fil[-1] = v[-1]
        k = np.arange(0, N-2)
        V_DCT = scipy.fft.dct(v[1:-1], type=1)

    filtre_exp = np.exp(-alpha*(k/k.max())**p)
    V_DCT_filter = filtre_exp*V_DCT

    V_FILTER = scipy.fft.idct(V_DCT_filter, type=1)
    if not interior:
        v_fil = V_FILTER
    else:
        v_fil[1:-1] = V_FILTER
    return v_fil