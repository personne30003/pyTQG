"""
Solveur de Helmholtz pour grille périodique selon x et avec CLs de Dirichlet selon y (les CLs de Neumann/Robin ne sont pas implémentées)
Résout une équation de la forme (dx^2 + dy^2 - alpha^2)psi=RHS
Validé du premier coup !!

"""
import numpy as np
import scipy
import Chebyshev
import sys
import time
import warnings

class HelmholtzChannel:
    def __init__(self,
                 Nx,
                 Ny,
                 x_bounds = (-np.pi, np.pi),
                 y_bounds = (-1.0, 1.0),
                 BC_y =('Dirichlet', 'Dirichlet'),
                 kx = None):
        self.Nx = Nx
        self.Ny = Ny
        #Vérifications
        if len(x_bounds) !=2:
            raise ValueError("x_bounds must contain 2 values")
        if len(y_bounds) !=2:
            raise ValueError("y_bounds must contain 2 values")

        if x_bounds[0] >= x_bounds[1]:
            raise ValueError("x_bounds's values must be sorted in ascending order")
        if y_bounds[0] >= y_bounds[1]:
            raise ValueError("y_bounds's values must be sorted in ascending order")

        if len(BC_y) != 2:
            raise ValueError("BC_y must contain 2 values")
        possible_BC = ('Dirichlet', 'Neumann')
        if (BC_y[0] not in possible_BC) or (BC_y[1] not in possible_BC):
            raise ValueError(f"BC_y's values must be in {possible_BC}, here BC_y = {BC_y}")
        if (BC_y[0] != 'Dirichlet') or (BC_y[1] != 'Dirichlet'):
            raise NotImplementedError("Only Dirichlet BC are implemented ...")


        print("Elliptic solver choosen : HelmholtzChannel")

        if kx is None:
            self.kx = 2.0*np.pi*scipy.fft.fftfreq(self.Nx, d=(x_bounds[1] - x_bounds[0])/self.Nx)
        else : 
            self.kx = kx
        self.D2 = Chebyshev.Cheb_mat(self.Ny, a = y_bounds[0], b=y_bounds[-1], Dirichlet_BC = False, M=2)
        self.__dic_alpha2 = {}

    def Solve(self, rhs, alpha2, bc_y_inf=0.0, bc_y_sup=0.0, real = True):
        "Doc ..."
        #Vérifications usuelles
        if np.isscalar(rhs) : 
            RHS = rhs * np.ones((self.Nx, self.Ny))
        elif isinstance(rhs, np.ndarray): 
            if (rhs.shape[0] != self.Nx) or (rhs.shape[1] != self.Ny):
                raise ValueError(f"rhs must have shape Nx*Ny (here rhs.shape = {rhs.shape})")
            RHS = rhs.copy()
        else:
            raise ValueError(f"rhs must be scalar or np.ndarray of size Nx x Ny")

        if np.isscalar(bc_y_inf) :
            BC_y_inf = np.ones(self.Nx) * bc_y_inf
        elif (( isinstance(bc_y_inf, np.ndarray) == True ) and
              ( bc_y_inf.shape[0] == self.Nx and len(bc_y_inf.shape) == 1 )):
            BC_y_inf = bc_y_inf.copy()
        else : 
            raise ValueError(f"bc_y_inf must be scalar or 1D array of size Nx (bc_y_inf = {bc_y_inf}")
        
        if np.isscalar(bc_y_sup) :
            BC_y_sup = np.ones(self.Nx) * bc_y_sup
        elif (( isinstance(bc_y_sup, np.ndarray) == True ) and
              ( bc_y_sup.shape[0] == self.Nx and len(bc_y_sup.shape) == 1 )):
            BC_y_sup = bc_y_sup.copy()
        else : 
            raise ValueError(f"bc_y_sup must be scalar or 1D array of size Nx (bc_y_sup = {bc_y_sup}")

        if alpha2 not in self.__dic_alpha2.keys() and (type(alpha2) == float):
            liste_piv, liste_F_fac = self.__factorize_matrix(alpha2)
        elif alpha2 in self.__dic_alpha2.keys():
            liste_piv = self.__dic_alpha2[alpha2][0]
            liste_F_fac = self.__dic_alpha2[alpha2][1]
        else:
            raise KeyError(f"alpha2 = {alpha2} not float, str or key of __dic_alpha2 (keys {self.__dic_alpha2.keys()}")

        psi_num = np.zeros((self.Nx, self.Ny))
        psi_num_hat = np.zeros((self.Nx, self.Ny), dtype = 'complex')

        rhs_hat = scipy.fft.fft(RHS, axis = 0)

        BC_y_inf_hat = scipy.fft.fft(BC_y_inf)
        BC_y_sup_hat = scipy.fft.fft(BC_y_sup)


        for i in range(0, self.Nx):
            rhs_hat_i = rhs_hat[i, 1:-1].copy()
            coeff_BC_sup = self.D2[1:-1, -0].copy()
            coeff_BC_inf = self.D2[1:-1, -1].copy()
            rhs_hat_BC = rhs_hat_i - coeff_BC_sup*BC_y_sup_hat[i] - coeff_BC_inf*BC_y_inf_hat[i]
            psi_num_hat[i, 1:-1] = scipy.linalg.lu_solve((liste_F_fac[i], liste_piv[i]), rhs_hat_BC)
            #on ajoute les BC
            psi_num_hat[i, 0] = BC_y_sup_hat[i]
            psi_num_hat[i, -1] = BC_y_inf_hat[i]

        psi_num = scipy.fft.ifft(psi_num_hat, axis = 0)
        
        if real:
            return psi_num.real
        return psi_num

    @property
    def alpha2(self):
        return self.__dic_alpha2

    @alpha2.setter
    def alpha2(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"value must be list or tuple, not {type(value)}")
        if len(value) != 2:
            raise ValueError(f"value must have lenght of 2, not {len(value)}")
        if type(value[0]) not in (int, str):
            warnings.warn(f"type {type(value)} not recommended for dictionnary key, please use int or str instead",
                          RuntimeWarning)

        self.__dic_alpha2[value[0]] = self.__factorize_matrix(value[1])

    def __factorize_matrix(self, alpha2):
        liste_piv = [0 for i in range(0, self.Nx)]
        liste_F_fac = [0 for i in range(0, self.Nx)]

        alpha_Id = alpha2*np.eye(self.Ny-2)
        print(f"Matrix factorisation")
        for i in range(0, self.Nx):
            F_i = np.zeros((self.Ny - 2, self.Ny - 2))
            np.fill_diagonal(F_i, -(alpha2+self.kx[i]**2))

            F_i+=self.D2[1:-1, 1:-1].copy()
            lu_i, piv_i  = scipy.linalg.lu_factor(F_i)
            liste_piv[i] = piv_i
            liste_F_fac[i] = lu_i

        size_matrix_prefac = self.__size_prefac_matrix(liste_piv, liste_F_fac)
        print(f"Matrix's size in memory : {size_matrix_prefac:.2f} MiB")
        return (liste_piv, liste_F_fac)

    def __size_prefac_matrix(self, *list_matrix):
        size_liste_mat = lambda liste_mat: (
                (sys.getsizeof(liste_mat) + sum(mat.nbytes for mat in liste_mat)) / (1024 ** 2))
        size_prefac = np.sum([size_liste_mat(liste_mat) for liste_mat in list_matrix])
        return size_prefac

