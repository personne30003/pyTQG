"""
Fonctions liées au calcul de dérivées/intégrales par la méthode de Collocation (Chebyshev)
Ces fonctions sont adaptées des codes Matlab du bouquin de L.Trefethen "spectral methods in Matlab"
Pour plus de détails théoriques, je renvoie à ce bouquin.
"""
import numpy as np
import scipy


def Cheb_mat(N, a=-1., b=1.):#L : etendue du domaine
    "Retourne une matrice de différenciation de taille NxN, sur un intervalle (a, b)"
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
    return 2.0*D/(b-a)

def collocation_points(N, a = -1., b = 1.) :
    "génère une grille 1D avec points de Gauss-Lobatto, sur un intervalle (a, b). Attention, les valeurs sont rangées en ordre décroissant :" 
    k = np.arange(0, N, 1)*np.pi/(N-1)
    x = np.cos(k)
    if a != -1.0 and b !=1.0 : 
        return ( (b-a)*x+ a+b)/2.0
    else :
        return x


def Cheb_FFT(v, a=-1., b=1.):#v  : vecteur 1D réel
    "Différentiation 1D par FFT. Tiré de Trefethen"
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

#################Opérateurs d'intégration. On utilise la méthode de Clenshaw-Curtis##############################################


def Clenshaw_Curtis_weight(N, a= -1.0, b = 1.0):
    N -=1
    theta = np.pi*np.arange(0, N+1)/N
    #print(f"theta = {theta}")
    w = np.zeros(N+1)
    
    ii = np.arange(1, N)
    v = np.ones(N-1)
    
    if ( N % 2 ) == 0 : 

        w[0] = 1.0/ ( N**2 - 1.0 )
        w[-1] = w[0]
        for k in range(1, N//2):
            v = v-2.0*np.cos(2.0*k*theta[ii])/( 4.0* ( k**2 ) -1 )
        v = v - np.cos(N*theta[ii])/(N**2-1.0)
    else:
        w[0] = 1.0/N**2
        w[-1] = w[0]
        for k in np.arange(1, (N-1)/2):
            v = v - 2.0*np.cos(2.0*k*theta[ii])/( 4.0 * ( k**2 ) - 1.0 )
    w[ii] = ((b -a )/2.0) * 2.0*v/N
    return w

def Cheb_quad(y, a=-1.0, b = 1.0) :
    "Intégre une fonction 1D calculée sur points de Gauss-Lobatto, sur intervalle [a, b]"
    weights = Clenshaw_Curtis_weight(y.size, a = a, b = b)
    #return np.sum(weights * y, axis = axis)
    return np.dot(weights, y)


################Fonctions appliquées sur des tableaux 2D, ne pas utiliser, préférer la classe Grid (à venir)#####################

def Cheb_FFT_2D(V, order = 1, axis ='x', a=-1., b= 1.):
    "Calcule la dérivée par FFT suivant une dimension (axis =0/'x ou axis = 1/'y')"
    if order ==1 : 
        func_deriv = lambda xx : Cheb_FFT(xx, a=a, b=b)
    elif order == 2 : 
        #func_deriv = lambda xx : Cheb_second_FFT(xx, a=a, b=a)
        func_deriv = lambda xx : Cheb_FFT(Cheb_FFT(xx, a=a, b=b), a=a, b=b)
    else :
        raise ValueError
    
    if axis == 'x' or axis == 0 : 
        dVdx = np.zeros_like(V)
        for i in range(0, V.shape[0]):
            dVdx[i, :] = func_deriv(V[i, :])
        return dVdx
    elif axis == 'y' or axis == 1:
        dVdy = np.zeros_like(V)
        for i in range(0, V.shape[1]):
            dVdy[:, i] = func_deriv(V[:,i])
        return dVdy
    else:
        raise ValueError
def Cheb_derivative_2D(V, order = 1, axis ='x', a = -1., b=1.):
    "idem que la prédédente mais avec la matrice. Préférable pour la dérivée seconde"
    if axis == 'x' or axis == 0 : 
        dVdx = np.zeros_like(V)
        Nx = dVdx.shape[0]
        D = Cheb_mat(Nx, a=a, b=b)
        #D = D[1:-1, 1:-1]
        if order == 2 : 
            D = D@D
        for i in range(0, V.shape[1]):
            dVdx[i, :] = D@V[i, :]
        return dVdx
    elif axis == 'y' or axis == 1:
        dVdy = np.zeros_like(V)
        for i in range(0, V.shape[0]):
            dVdy[i, :] = D@V[:, i]
        return dVdy
    else:
        raise ValueError