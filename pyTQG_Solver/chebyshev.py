"""
Fonctions liées au calcul de dérivées/intégrales par la méthode de Collocation (Chebyshev)
Ces fonctions sont (pour la plupart) adaptées des codes Matlab du bouquin de L.Trefethen "spectral methods in Matlab"
Pour plus de détails théoriques, je renvoie à ce bouquin, à ceux de Boyd et Peyret (cf biblio de mon rapport)

"""
import numpy as np
import numpy.polynomial.chebyshev as npcheb
import numba

def Cheb_mat(N, a=-1., b=1., Dirichlet_BC = False, M=1):#L : etendue du domaine
    """"
    Retourne une matrice de différenciation d'ordre M sur une grille de taille NxN, sur un intervalle (a, b).
    Tiré de Trefethen (fonction cheb.m, chapitre 6)
    Dirichlet_BC : si vrai, retourne une matrice adaptée aux CLs de Dirichlet.
    """
    if N == 0 : 
        raise ValueError("N must be not null")
    N-=1
    k = np.arange(0, N+1, 1)*np.pi/N
    x = np.cos(k)
    range_N = np.arange(0,N+1, 1)
    #construction de la partie non diagonale
    c = np.array([np.ones(N+1)])#on déclare un vecteur colonne
    c[0, 0] = c[0, -1] = 2.
    c = c*(-1.0)**range_N
    
    X = np.tile(np.flip(x), N+1).reshape(N+1, N+1)
    dX = X - X.T
    #elements non diagonaux
    ci_s_cj = c.T * (1.0/c)
    D_1 = ci_s_cj /(dX+np.eye(N+1))
    #elements diagonaux
    D = D_1 - np.diag(np.sum(D_1, axis=1))
    D = 2.0*D/(b-a)
    if M > 1:
        D = np.linalg.matrix_power(D, M)
    if M <= 0:
        raise ValueError(f"order M must be >= 1 (current value M = {M})")
    #imposition de CLs de Dirichlet
    if Dirichlet_BC:
        D[0, 0] = 1.0
        D[0, 1:] = 0.0
        D[-1, -1] = 1.0
        D[-1, 0:-1] = 0.0
    return D

def collocation_points(N, a = -1., b = 1.) :
    "génère une grille 1D avec points de Gauss-Lobatto, sur un intervalle (a, b). Attention, les valeurs sont rangées en ordre décroissantes." 
    k = np.arange(0, N, 1)*np.pi/(N-1)
    x = np.cos(k)
    if a != -1.0 or b !=1.0 : 
        return ( (b-a)*x+ a+b)/2.0
    else :
        return x




#################################################################################################################################
#############################################Opérateurs d'intégration.###########################################################

@numba.jit#Petite optimisation par précompilation : on gagne un petit facteur 2 en temps dans Cheb_quad
def Clenshaw_Curtis_weight(N, a=-1.0, b=-1.0):
    """Calcule les poids de CLenshaw-Curtis, pour une fonction définie sur N points, sur un intervalle (a, b),
       et interpolée par un polynome de degré N-1
       Adapté du programme MATLAB clencurt.m (Trefethen, chapitre 12).
    """
    N -=1#définit le degré des polynomes
    
    theta = np.pi*np.arange(0, N+1)/N
    w = np.zeros(N+1)
    ii = np.arange(1, N)

    v = np.ones(N-1)

    if ( N % 2 ) == 0 : 

        w[0] = 1.0/ ( N**2 - 1.0 )
        w[-1] = w[0]

        for k in np.arange(1, N/2):
            v = v-2.0*np.cos(2.0*k*theta[ii])/( 4.0* ( k**2 ) -1 )

        v = v - np.cos(N*theta[ii])/(N**2-1.0)

    else:

        w[0] = 1.0/N**2
        w[-1] = w[0]
        for k in np.arange(1, (N-1)/2+1):
            v = v - 2.0*np.cos(2.0*k*theta[ii])/( 4.0 * ( k**2 ) - 1.0 )

    w[ii] = 2.0*v/N
    
    if (a !=-1.0) or (b !=1.0) : 
        w = ((b -a )/2.0)*w
    return w

def Cheb_quad(y, a=-1.0, b = 1.0, weights = None, axis = None) :
    """Intégre une fonction 1D calculée sur points de Gauss-Lobatto, sur intervalle [a, b], sur l'axe ('x', 'y', None).
    Pour un simple tableau 1D, laisser axis = None"""
    if weights is None : 
        weights = Clenshaw_Curtis_weight(y.size, a = a, b = b)
    if axis == 'x':
        axis = 1
    elif axis == 'y':
        axis = 0
        if weights.shape != (y.shape[1], 1):
            weights = weights[..., None]#on transforme k en vecteur colonne
    elif axis is None:
        pass
    else:
        raise ValueError(f"axis must be in ('x', 'y', None), current value : axis = {axis}")
        
    return np.sum(weights*y, axis = axis)


def Cheb_cumsum(y, x) :
    """
    Idem que Chebquad, mais pour une intégrale calculée sur l'intervalle [a, x] (où x>= a est croissant).
    On travaille ici directement sur les coefficients. C'est lent, mais c'est précis,
     et de toute façon, c'est pas utilisé pour les diagnostics
    """
    coeffs = npcheb.chebfit(x, y, len(y)-1)
    coeffs_int = npcheb.chebint(coeffs)
    y_int = npcheb.chebval(x, coeffs_int)
    return y_int-y_int[-1]
