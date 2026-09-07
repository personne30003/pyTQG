"""
Quelque fonctions liées au déaliasing : 
-Règle des 2/3 (Orszag et al) pour Fourier
-Filtre exponentiel pour la base de Chebyshev. Le filtre est appliqué sur les coefficients, calculés par DCT

Ces fonctions ne s'appliquent que sur des vecteurs 1D (comme toutes les fonctions bas-niveau)
"""
import numpy as np
import scipy
import dataclasses

liste_modes_dealias = ('after_product', 'after_derivative')
#petite classe rassemblant les paramètres de déaliasing
@dataclasses.dataclass
class DealiasParams:
    apply_dealias : bool = False
    dealias_order : str = 'after_product'#valeurs possibles "after_product", "after_derivative"
    dealias_Fourier_coeff : float = 2.0/3.0
    dealias_exp_alpha : float = 100.0
    dealias_exp_p : float = 8.0

    def __post_init__(self):
        if self.dealias_order not in liste_modes_dealias:
            raise ValueError(f"dealias_order must be str, with value in {liste_modes_dealias}, current value : {liste_mod_dealias}")



def Fourier_dealias(v, dx, coeff_dealias=2.0/3.0, real = True, axis = None):
    """"
    déaliase un tableau 1D/2D sur un axe donnée, par la règle des 2/3 (on retire toutes les composantes avec |k| > 2/3*kmax)
    axis = (None, 'x', 'y'). 'x' et 'y' correspondent aux axes 0 et 1, dans le cas où la grille 2D est définie 
    avec np.meshgrid(..., indexing = 'ij')
    """
    if axis is None:
        axis = -1
        N = v.size
    elif axis == "x":
        axis = 0
        N = v.shape[0]
    elif axis == "y":
        axis = 1
        N = v.shape[1]
    else:
        raise ValueError(f"axis must be in ('x', 'y', None), current value = {axis}")
    
    kx = 2.0*np.pi*scipy.fft.fftfreq(N, d=dx)
    if axis == 0:#On transforme k en vecteur colonne
        kx = kx[..., None]

    kx_max = np.abs(kx).max()
    v_hat = scipy.fft.fft(v, axis = axis)
    v_hat_dealias = np.where(np.abs(kx) < coeff_dealias*kx_max, v_hat, 0.0)#TODO : adapter ça pour une FFT 2D
    v_dealias = scipy.fft.ifft(v_hat_dealias, axis = axis)

    if real : 
        return v_dealias.real
    return v_dealias

def exp_filter_DCT(v, alpha=100.0, p=8.0, interior = True, axis = None):
    if type(v) !=np.ndarray:
        v = np.array(v)

        
    if axis is None:
        axis = -1
        N = v.size
    elif axis == "x":
        axis = 0
        N = v.shape[0]
    elif axis == "y":
        axis = 1
        N = v.shape[1]
    else:
        raise ValueError(f"axis must be in ('x', 'y', None), current value = {axis}")
    
    v_res = v.copy()
    if interior:
        N-=2
        v_fil = sel_interior_points(v, axis)
    else : 
        v_fil = v.copy()
    k=np.arange(0, N)
    if axis == 0:#On transforme k en vecteur colonne
            k = k[..., None]
    #print(f"interior = {interior}")
    #print(f"v_fil.shape = {v_fil.shape}")
    #print(f"v.shape = {v.shape}")
    V_DCT = scipy.fft.dct(v_fil, type = 1, axis = axis)
    filtre_exp = np.exp(-alpha*(k/k.max())**p)
    V_DCT_filter = filtre_exp*V_DCT

    V_FILTER = scipy.fft.idct(V_DCT_filter, type=1, axis = axis)
    if not interior:
        return v_fil
    else:
        insert_interior_points(v_res, V_FILTER, axis)
    return v_res

#############Fonctions à usage interne seulement#############################
def sel_interior_points(val, axis):
    sel = [slice(None)]*val.ndim
    sel[axis] = slice(1, -1)
    return val[tuple(sel)]

def insert_interior_points(array, vals_to_insert, axis):
    sel = [slice(None)]*array.ndim
    sel[axis] = slice(1, -1)
    array[tuple(sel)] = vals_to_insert
    return array