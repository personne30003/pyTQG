"""
Fonctions liées au calcul de dérivées/intégrales par la méthode de Collocation (Chebyshev)
Ces fonctions sont (pour la plupart) adaptées des codes Matlab du bouquin de L.Trefethen "spectral methods in Matlab"
Pour plus de détails théoriques, je renvoie à ce bouquin, à ceux de Boyd et Peyret (cf biblio de mon rapport)

ChebDiff_FFT est adaptée de la bibliothèque MATLAB développée par Weidemann (https://appliedmaths.sun.ac.za/~weideman/research/differ.html). 
Elle permet de calculer la dérivée n-ème de n'importe quelle fonction 1D par FFT
Voir commentaires au dessus du code correspondant.
"""
import numpy as np
import scipy
import numpy.polynomial.chebyshev as npcheb
import numba

def Cheb_mat(N, a=-1., b=1., Dirichlet_BC = False, M=1):#L : etendue du domaine
    """"
    Retourne une matrice de différenciation d'ordre M sur une grille de taille NxN, sur un intervalle (a, b). Tiré de Trefethen (fonction cheb.m, chapitre 6)
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


def Cheb_FFT(v, a=-1., b=1.):#v  : vecteur 1D réel
    "Différentiation 1D par FFT, sur intervalle (a, b). Tiré de Trefethen (Programme chebfft.m, chapitre 8)"
    N = v.size -1
    if N==0 : 
        raise ValueError("N must be >1")
    
    k = np.arange(0, N+1)
    #x = ( (b-a)*np.cos(np.pi*k/N)+ a+b)/2.0
    x = np.cos(np.pi*k/N)
    #x = collocation_points(N, a = a, b=b)
    
    ii = np.arange(0, N)

    v_col = np.array(v)
    v_flip = np.flip(v_col[1:-1])

    V = np.concatenate((v_col, v_flip))
    U = scipy.fft.fft(V).real
    W = U.copy()
    
    ik = 1.j * np.concatenate((ii, [0], -np.flip(k)[1:-1]))#Ouais, c'est pas explicite !
    W = scipy.fft.ifft(ik * U).real
    w = np.zeros(N+1)
    w[1:-1] = -W[1:N]/np.sqrt(1.0-x[1:-1]**2)
    w[0] = np.sum(ii**2 * U[ii])/N + 0.5*N*U[N]
    wN_1 = np.sum((-1.)**(ii+1) * (ii**2.) * U[ii])/N
    wN_2 = .5 * ( (-1)**(N+1) ) * N * U[N]
    w[-1] = wN_1+wN_2
    return 2.0*w/(b-a)


def Cheb_second_FFT(v, a = -1., b=1.):
    "calcul de la dérivée seconde. Tiré de Trefethen. Ne pas utiliser, c'est pas précis et ça prend pas en compte les points extérieurs (nécessitent des formules spéciales"
    N = v.size-1

    k = np.arange(0, N+1)

    x =  np.cos( np.pi * k / N )
    ii = np.arange(1, N)
    
    v_pp = v.copy()
    v_col = np.array(v)
    v_flip = np.flip(v_col[1:-1])

    V = np.concatenate((v_col, v_flip))
    U = scipy.fft.fft(V).real
    
    ik_1 = 1.j*np.concatenate((k[:-1], [0], -np.flip(k[1:-1])))
    W1 = scipy.fft.ifft(ik_1 * U).real
    ik_2 =  ik_1**2
    W2  = scipy.fft.ifft(ik_2 * U).real
    v_pp[ii] = W2[ii]/(1. - x[ii]**2) - x[ii] * W1[ii]/( (1.- x[ii]**2)**(3./2.) )
    
    return 4.*v_pp/(b-a)**2

###################Fonction de dérivation par FFT (tout ordre confondu)###############################################################################
def ChebDiff_FFT(f, M=1, inf_bound=-1., sup_bound=1.):
    """
    Calcule la dérivée d'ordre M d'une fonction 1D sur un domaine (inf_bound, sup_bound)
    Adapté de la bibliothèque MATLAB développée par Weidemann https://appliedmaths.sun.ac.za/~weideman/research/differ.html
    A voir à l'usage, c'est un peu plus précis que la méthode matricielle, mais moins rapide au final
    Code généré par IA (au bout d'1.5 jours, j'en avais marre), je lui ai demandé d'émuler les indices MATLAB. Donc pas optimisé !
    La partie diffile étant ce qui se passe dans les boucles for...
    """
    f = np.asarray(f, dtype=complex).flatten()
    N = len(f)
    
    # En MATLAB: f=f(:); a0=fft([f; flipud(f(2:N-1))]);
    # f(2:N-1) en MATLAB correspond aux indices 2 à N-1 inclus.
    # En Python (index-0), cela correspond à f[1:N-1]
    f_flip = f[1:N-1][::-1]
    f_ext = np.concatenate([f, f_flip])
    a0_fft = np.fft.fft(f_ext)
    
    # En MATLAB: a0=a0(1:N).*[0.5; ones(N-2,1); 0.5]/(N-1);
    # On crée un tableau indexé de 1 à N pour coller au MATLAB
    a0_matlab = np.zeros(N + 1, dtype='complex')
    weights = np.ones(N)
    weights[0] = 0.5
    weights[-1] = 0.5
    a0_matlab[1:N+1] = a0_fft[:N] * weights / (N - 1)
    
    # En MATLAB: a=[a0 zeros(N,M)];
    # Taille MATLAB : N lignes, M+1 colonnes.
    # On crée une matrice de taille (N+1) x (M+2) pour utiliser les indices 1..N et 1..M+1
    a = np.zeros((N + 1, M + 2), dtype='complex')
    a[1:N+1, 1] = a0_matlab[1:N+1]

    # Boucle MATLAB exacte
    for ell in range(1, M + 1):
        # a(N-ell,ell+1)=2*(N-ell)*a(N-ell+1,ell);
        a[N - ell, ell + 1] = 2.0 * (N - ell) * a[N - ell + 1, ell]
        
        # for k=N-ell-2:-1:1
        for k in range(N - ell - 2, 0, -1):
            # a(k+1,ell+1)=a(k+3,ell+1)+2*(k+1)*a(k+2,ell);
            a[k + 1, ell + 1] = a[k + 3, ell + 1] + 2.0 * (k + 1.0) * a[k + 2, ell]
            
        # a(1,ell+1)=a(2,ell)+a(3,ell+1)/2;
        a[1, ell + 1] = a[2, ell] + a[3, ell + 1] / 2.0

    # back=[2*a(1,M+1); a(2:N-1,M+1); 2*a(N,M+1); flipud(a(2:N-1,M+1))];
    # Construction du vecteur 'back' en suivant scrupuleusement les indices MATLAB
    # a(2:N-1, M+1) signifie de la ligne 2 à N-1 incluse
    part2 = a[2:N, M + 1]
    part4 = part2[::-1] # flipud
    
    back = np.concatenate([
        [2.0 * a[1, M + 1]],
        part2,
        [2.0 * a[N, M + 1]],
        part4
    ])
    
    # Dmf=0.5*fft(back); Dmf=Dmf(1:N);
    Dmf = 0.5 * np.fft.fft(back)
    Dmf = Dmf[:N]

    coeff_domaine = (2.0/(sup_bound-inf_bound))**M
    return coeff_domaine * (Dmf.real)

#################Calcul de la matrice de dérivée d'ordre 2 en tenant compte des CLs##############################################
def Cheb2_BC(N, a=-1.0, b=1.0, BC_xm1 = 'Dirichlet', BC_xp1 = 'Dirichlet', get_D = False):
    """
    Matrice de dérivée du 2e ordre avec prise en compte des BCs de Dirichlet/Neumann
    De la forme u(i)+d(u(i)), i = 0, N.
    Valeurs possibles de BC_xm1 (CL en -1 ou en a) et BC_xp1 (CL en 1 ou en b) : 'Dirichlet', 'Neumann', 'Robin'. 
    SANS GARANTIE POUR ROBIN !!!
    """
    D = Cheb_diff(N, a=a, b=b)
    D2 = D@D
    
    if BC_xm1 == 'Dirichlet':
        D2[0, 0] = 1.0
        D2[0, 1:] = 0.0
        D[0, 0] = 1.0
        D[0, 1:] = 0.0
    
    elif BC_xm1 == 'Neumann':
        D2[0, :] = D[0, :].copy()
    
    elif BC_xm1 == 'Robin':
        D2[0, :] = D[0, :]
        D2[0, 0] +=1.0
        D[0, 0] = 1.0
        D[0, 1:-1] = 0.0
    else:
        raise ValueError(f"BC_xm1 must be in ('Dirichlet', 'Neumann', 'Robin'), current value '{BC_xm1}'")

    if BC_xp1 == 'Dirichlet':
        D2[-1, -1] = 1.0
        D2[-1, 0:-1] = 0.0
        D[-1, -1] = 1.0
        D[-1, 0:-1] = 0.0
    
    elif BC_xp1 == 'Neumann':
        D2[-1, :] = D[-1, :].copy()
    
    elif BC_xp1 == 'Robin':
        D2[-1, :] = D[-1, :].copy()
        D2[-1, -1] +=1.0
        D[-1, 0:-1] = 0.0
        D[-1, -1] = 1.0
    else:
        raise ValueError(f"BC_xp1 must be in ('Dirichlet', 'Neumann', 'Robin'), current value '{BC_xp1}'")
    if get_D:
        return D, D2
    return D2
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
    On travaille ici directement sur les coefficients. C'est lent, mais c'est précis, et de toute façon, c'est pas utilisé pour les diagnostics
    """
    coeffs = npcheb.chebfit(x, y, len(y)-1)
    coeffs_int = npcheb.chebint(coeffs)
    y_int = npcheb.chebval(x, coeffs_int)
    return y_int-y_int[-1]
