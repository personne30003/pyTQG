"""
Classe regroupant les opérateurs de dérivation, intégration et déaliasing

"""


import parameters
import fourier
import chebyshev
import dealiasing
import numpy as np

import collections

class Grid:
    def __init__(self, params : parameters.Params):
        self.__Nx = params.Nx
        self.__Ny = params.Ny
        self.__x_bounds = params.x_bounds
        self.__y_bounds = params.y_bounds
        self.__dealias_params = params.dealias_params
        self.__geometry = params.geometry
        
        ####Petites vérifications##################
        if type(self.__x_bounds) != tuple:
            raise ValueError(f"x_bounds must be tuple, not {type(self.__x_bounds)}")
        if len(self.__x_bounds) != 2:
            raise ValueError(f"x_bounds must have lenght 2, not {len(self.__x_bounds)}")
        
        if type(self.__y_bounds) != tuple:
            raise ValueError(f"y_bounds must be tuple, not {type(self.__y_bounds)}")
        if len(self.__y_bounds) != 2:
            raise ValueError(f"x_bounds must have lenght 2, not {len(self.__y_bounds)}")

        #####A faire : clarifier le message !###########
        if self.__x_bounds[0] >= self.__x_bounds[1]:
            raise ValueError(f"x_bounds[0] must not be >= x_bounds[1]")
        if self.__y_bounds[0] >= self.__y_bounds[1]:
            raise ValueError(f"y_bounds[0] must not be >= y_bounds[1]")
        ############Initialisation des valeurs des opérateurs####################################
        self.__dealias_x_1D = None
        self.__dealias_y_1D = None

        self.__dx_1D = None
        self.__dy_1D = None
        self.__d2x_1D = None
        self.__d2y_1D = None

        self.__dx_1D_BC = None
        self.__dy_1D_BC = None
        self.__d2x_1D_BC = None
        self.__d2y_1D_BC = None

        self.__int_x_1D = None
        self.__int_y_1D = None

        self.__int_cum_x_1D = None
        self.__int_cum_y_1D = None
        ##########Eventuelles matrices de différenciation (seulement pour configurations "canal" et "basin")######################################
        self.__Dy = None
        self.__Dy_BC = None
        self.__D2y = None
        self.__D2y_BC = None
        ############################coefficients de poids pour les fonctions d'intégrations#######################################################
        self.__int_weight_x = None
        self.__int_weight_y = None
        ####On génère la grille, et on attribue les fonctions#######################
        if self.__geometry == 'biperiodic':
            self.x = np.linspace(self.__x_bounds[0], self.__x_bounds[1], self.__Nx, endpoint = False)
            self.y = np.linspace(self.__y_bounds[0], self.__y_bounds[1], self.__Ny, endpoint = False)
            self.__dx = self.x[1] - self.x[0]
            self.__dy = self.y[1] - self.y[0]

            #fonctions 1D
            self.__dealias_x = lambda xx : dealiasing.Fourier_dealias(xx,
                                                                         self.__dx,
                                                                         coeff_dealias = self.__dealias_params.dealias_Fourier_coeff,
                                                                         axis = 'x')
            self.__dealias_y = lambda yy : dealiasing.Fourier_dealias(yy,
                                                                         self.__dy,
                                                                         coeff_dealias = self.__dealias_params.dealias_Fourier_coeff,
                                                                         axis = 'y')

            self.__int_weight_x = self.__dx
            self.__int_weight_y = self.__dy
            self.__int_x_1D = lambda xx : fourier.Fourier_quad(xx, self.__int_weight_x, axis = 'x')
            self.__int_y_1D = lambda yy : fourier.Fourier_quad(yy, self.__int_weight_y, axis = 'y')

            self.__int_cum_x_1D = lambda xx : fourier.Fourier_cumsum(xx, self.__dx)
            self.__int_cum_y_1D = lambda yy : fourier.Fourier_cumsum(yy, self.__dy)
            
            self.__dx_1D = lambda xx : fourier.Fourier_deriv(xx, order = 1, a = self.__x_bounds[0], b = self.__x_bounds[1], axis = 'x')
            self.__dy_1D = lambda yy : fourier.Fourier_deriv(yy, order = 1, a = self.__y_bounds[0], b = self.__y_bounds[1], axis = 'y')

            self.__dy_1D_BC = self.__dy_1D
            self.__dx_1D_BC = self.__dx_1D

            self.__d2x_1D = lambda xx : fourier.Fourier_deriv(xx, order = 2, a = self.__x_bounds[0], b = self.__x_bounds[1], axis = 'x')
            self.__d2y_1D = lambda yy : fourier.Fourier_deriv(yy, order = 2, a = self.__y_bounds[0], b = self.__y_bounds[1], axis = 'y')

            self.__d2y_1D_BC = self.__d2y_1D
            self.__d2x_1D_BC = self.__d2x_1D
            
        elif self.__geometry == 'zonal_channel' : 
            self.x = np.linspace(self.__x_bounds[0], self.__x_bounds[1], self.__Nx, endpoint = False)
            self.y = chebyshev.collocation_points(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1])
            self.__dx = self.x[1] - self.x[0]
            self.__dy = np.abs(self.y[1]-self.y[0])#les points sont les plus rapprochés aux bords


            self.__int_weight_x = self.__dx
            self.__int_weight_y = chebyshev.Clenshaw_Curtis_weight(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1])

            self.__Dy = chebyshev.Cheb_mat(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1]).T
            self.__D2y = chebyshev.Cheb_mat(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1], M=2).T

            self.__Dy_BC = chebyshev.Cheb_mat(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1], Dirichlet_BC = True).T
            self.__D2y_BC = chebyshev.Cheb_mat(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1], Dirichlet_BC = True, M = 2).T

            self.__dx_1D = lambda xx : fourier.Fourier_deriv(xx, order = 1, a = self.__x_bounds[0], b = self.__x_bounds[1], axis = 'x')
            self.__dy_1D = lambda yy : yy @ self.__Dy

            self.__d2x_1D = lambda xx : fourier.Fourier_deriv(xx, order = 2, a = self.__x_bounds[0], b = self.__x_bounds[1], axis = 'x')
            self.__d2y_1D = lambda yy : yy @ self.__D2y

            self.__dx_1D_BC = self.__dx_1D
            self.__dy_1D_BC = lambda yy : yy @ self.__Dy_BC

            self.__d2x_1D_BC = self.__d2x_1D
            self.__d2y_1D_BC = lambda yy : yy @ self.__D2y_BC

            self.__dealias_x = lambda xx : dealiasing.Fourier_dealias(xx,
                                                                         self.__dx,
                                                                         coeff_dealias = self.__dealias_params.dealias_Fourier_coeff,
                                                                         axis = 'x')
            
            self.__dealias_y = lambda yy : dealiasing.exp_filter_DCT(yy,
                                                                        alpha = self.__dealias_params.dealias_exp_alpha,
                                                                        p = self.__dealias_params.dealias_exp_p,
                                                                        axis = 'y')
            
            self.__int_x = lambda xx : fourier.Fourier_quad(xx, self.__int_weight_x, axis = 'x')
            self.__int_y = lambda yy : chebyshev.Cheb_quad(yy, weights = self.__int_weight_y, axis = 'y')

            self.__int_cum_x_1D = lambda xx : fourier.Fourier_cumsum(xx, self.__dx)
            self.__int_cum_y_1D = lambda yy : chebyshev.Cheb_cumsum(yy, self.y)
            
        elif self.__geometry == 'basin':
            raise NotImplementedError("configuration basin not (yet) implemented")
        else:
            raise ValueError(f"params.geometry must be in {params.list_geometry}, current value {self.__geometry}")
        

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing = 'ij')
        #On rend immutables les variables liées à la géométrie, pour ne pas que l'utilisateur les modifie.
        self.X.setflags(write = False)
        self.Y.setflags(write = False)
        self.x.setflags(write = False)
        self.y.setflags(write = False)
        #Fonctions d'intégration
        self.__int_cum_x = self.__apply_along_axis(self.__int_cum_x_1D, "x")
        self.__int_cum_y = self.__apply_along_axis(self.__int_cum_y_1D, "y")

        #####On met ça dans l'ordre#################################
        Deriv_axis = collections.namedtuple('Deriv_axis', ['di', 'di_BC', 'd2i', 'd2i_BC'])
        self.__liste_deriv_x = Deriv_axis(di = self.__dx_1D,
                                          di_BC = self.__dx_1D_BC,
                                          d2i = self.__d2x_1D,
                                          d2i_BC = self.__d2x_1D_BC)
        
        self.__liste_deriv_y = Deriv_axis(di = self.__dy_1D,
                                          di_BC = self.__dy_1D_BC,
                                          d2i = self.__d2y_1D,
                                          d2i_BC = self.__d2y_1D_BC)
    
    ##############Fonctions pour l'utilisateur##########################
    def derivative(self, val, axis, order =1, BC_Dirichlet = False):
        "Calcule la dérivée d'ordre 1 ou 2 d'un array de taille (Nx, Ny), sur l'axe 'x' ou 'y', avec/sans CLs de Dirichlet (pas valable pour Fourier)"
        if axis == "x" : 
            liste_func_axis = self.__liste_deriv_x
        elif axis == "y" : 
            liste_func_axis = self.__liste_deriv_y
        else :
            raise ValueError(f"axis must be in ('x', 'y'), current value : {axis}")
        pass

        if order == 1 : 
            liste_keys = ('di', 'di_BC')
        elif order == 2 :
            liste_keys = ('d2i', 'd2i_BC')
        elif order < 1:
            raise ValueError(f"order must be > 0")
        else :
            raise NotImplementedError("order > 2 are not implemented")
        
        liste_keys_index = 0
        if BC_Dirichlet:
            liste_keys_index = 1
        func_deriv = getattr(liste_func_axis, liste_keys[liste_keys_index])
        return func_deriv(val)

    def integrate(self, val, axis):
        """"
        Intègre une fonction sur seul axe ('x', 'y', 'all'). 
        Renvoie :
        -un array de taille Ny si axis = 'x',
        -un array de taille Nx si axis = 'y'
        -un scalaire si axis = 'all' """
        self.__check_shape(val)
        if axis == 'x':
            return np.sum(self.__int_weight_x*val, axis = 0)
        elif axis == 'y':
            return np.sum(self.__int_weight_y*val, axis = 1)
        elif axis == 'all':
            return self.integrate_all_domain(val)
        else:
            raise ValueError("axis must be in ('x', 'y')")

    def integrate_all_domain(self, val):
        "Intègre sur tout le domaine. Renvoie un scalaire. Peut etre appelé directement depuis integrate"
        self.__check_shape(val)
        I_x = np.sum(self.__int_weight_x*val, axis = 0)
        I = np.sum(self.__int_weight_y*I_x)
        return I
        
    def dealias(self, value, axis):
        "Dé-aliase_selon une ou plusieurs directions (utile si le modèle inclut des produits de termes)"
        self.__check_shape(value)
        if axis == "x" : 
            return self.__dealias_x(value)
        elif axis == "y" : 
            return self.__dealias_y(value)
        elif axis == 'all' : ###ATTENTION, NON FIXE, JE SAIS PAS SI C'EST COMMUTATIF !!!!
            return self.__dealias_y(self.__dealias_x(value))
        else :
            raise ValueError(f"axis must be in ('x', 'y', 'all'), current value : {axis}")

    def dealias_product(self, A, B, axis_A, axis_B):
        """
        Applique un déaliasing sur le produit A*B
        Deux possibilités : soit déaliasing 'before_product', soit 'after_product'
        A voir si je le mets. Pour le moment, pas d'utilité immédiate
        """
        raise NotImplementedError("dealias_product not (yet ?) implemented :-(")

    def jacobien(self,val_A, val_B, BC_A = False, BC_B = True):
        "Calcule J(A, B) = dx(A)*dy(B) - dy(A)*dx(B), en appliquant un déaliasing si nécessaire"
        key_BC_A = 'di'
        key_BC_B = 'di'
        if BC_A:
            key_BC_A = 'di_BC'

        if BC_B:
            key_BC_B = 'di_BC'

        dx_A = getattr(self.__liste_deriv_x, key_BC_A)(val_A)
        dy_A = getattr(self.__liste_deriv_y, key_BC_A)(val_A)
        dx_B = getattr(self.__liste_deriv_x, key_BC_B)(val_B)
        dy_B = getattr(self.__liste_deriv_y, key_BC_B)(val_B)
        
        if self.__dealias_params.dealias_order == 'before_product' and self.__dealias_params.apply_dealias:
            dx_A = self.dealias(dx_A, "x")
            dx_B = self.dealias(dx_B, "x")
            dy_A = self.dealias(dy_A, "y")
            dy_B = self.dealias(dy_B, "y")

        dxA_dyB = dx_A*dy_B
        dyA_dxB = dy_A*dx_B
        if self.__dealias_params.dealias_order == 'after_product' and self.__dealias_params.apply_dealias:
            dxA_dy_B = self.dealias(dxA_dyB, "all")
            dyA_dxB = self.dealias(dyA_dxB, "all")
        return dxA_dyB - dyA_dxB

    def laplacien(self, psi, BC = False):
        "Calcule le laplacien 2D d'une fonction scalaire"
        return (self.derivative(psi, "x", order = 2, BC_Dirichlet = BC)
                + self.derivative(psi, "y", order = 2, BC_Dirichlet = BC))

    def int_cum(self, val, axis):
        "Calcule l'intégrale cumulée selon un axe ('x', 'y'). Renvoie un array de taille (Nx, Ny). Lent et peu utile ...."
        self.__check_shape(val)
        if axis == 'x':
            return self.__int_cum_x(val)
        elif axis == 'y':
            return self.__int_cum_y(val)
        else:
            raise ValueError(f"axis must be in ('x', 'y'), current value = {axis}")

    
    def __repr__(self):
        str_out = f"""
Grid : \n
Geometry = {self.__geometry}\n
-Size {self.X.shape} \n
-Step : dx = {self.__dx}, dy = {self.__dy}\n
-Intervals : x = {self.__x_bounds}, y = {self.__y_bounds} \n
-DealiasParams : {self.__dealias_params}"""
        return str_out
###########Fonctions privées########################################################
    def __apply_along_axis(self, func, axis):
        """
        applique une fonction de la forme f(x) sur un axe donné ('x', 'y') ;
        x est un tableau de taille (Nx, Ny) ; renvoie aussi un tableau de taille Ny. Utilisé seulement pour les
        intégrales cumulées (car trop lent)
        """
        axis_sel = None
        if axis == "x" : 
            axis_sel = 1
            N_ax = self.__Ny
        elif axis == "y" : 
            axis_sel = 0
            N_ax = self.__Nx
        else :
            raise ValueError(f"axis must be in ('x', 'y'), current value : {axis}")

        def func_along_axis(x):
            self.__check_shape(x)
            x_res = np.zeros_like(x)
            
            for i in range(0, N_ax):
                x_i = self.__take_along_axis(x, i, axis_sel)
                self.__put_along_axis(x_res, i, func(x_i), axis_sel)
            return x_res
            
        return func_along_axis

    
    def __check_shape(self, val):
        if type(val) != np.ndarray:
            raise ValueError(f"val must be np.ndarray, current type = {type(val)}")
        if val.shape != (self.__Nx, self.__Ny):
            raise ValueError(f"val.shape must be equal to (Nx, Ny)= ({self.__Nx}, {self.__Ny}), current shape = ({val.shape[0]}, {val.shape[1]})")

    def __take_along_axis(self, val, index, axis):
        if axis == 0:
            return val[index, :]
        elif axis == 1:
            return val[:, index]
        else : 
            raise ValueError(f"axis must be 0 or 1, not {axis}")

    def __put_along_axis(self, array, index, val, axis):
        if axis == 0:
            array[index, :] = val
        elif axis == 1:
            array[:, index] = val
        else:
            raise ValueError(f"axis must be 0 or 1, not {axis}")
    #############Accès à quelque variables privées#####################################
    @property
    def dx(self):
        return self.__dx

    @property
    def dy(self):
        return self.__dy