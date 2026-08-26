"""
Classe regroupant les opérateurs de dérivation, intégration et déaliasing

"""


import Params
import Fourier
import Chebyshev
import Dealiasing
import numpy as np
import scipy
import collections

class Grid:
    def __init__(self, params):
        self.__Nx = params.Nx
        self.__Ny = params.Ny
        self.__x_bounds = params.x_bounds
        self.__y_bounds = params.y_bounds
        self.__dealias_params = params.dealias_params
        self.__geometry = params.geometry
        
        ####Petites vérifications##################
        if type(self.__x_bounds) != tuple:
            raise ValueError(f"x_bounds must be tuple, not {type(x_bounds)}")
        if len(self.__x_bounds) != 2:
            raise ValueError(f"x_bounds must have lenght 2, not {len(self.__x_bounds)}")
        
        if type(self.__y_bounds) != tuple:
            raise ValueError(f"y_bounds must be tuple, not {type(y_bounds)}")
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
        ####On génère la grille, et on attribue les fonctions#######################
        if self.__geometry == 'biperiodic':
            self.x = np.linspace(self.__x_bounds[0], self.__x_bounds[1], self.__Nx, endpoint = False)
            self.y = np.linspace(self.__y_bounds[0], self.__y_bounds[1], self.__Ny, endpoint = False)
            self.__dx = self.x[1] - self.x[0]
            self.__dy = self.y[1] - self.y[0]

            #fonctions 1D
            self.__dealias_x_1D = lambda xx : Dealiasing.Fourier_dealias(xx, self.__dx, self.__dealias_params.dealias_Fourier_coeff)
            self.__dealias_y_1D = lambda yy : Dealiasing.Fourier_dealias(yy, self.__dy, self.__dealias_params.dealias_Fourier_coeff)

            self.__int_x_1D = lambda xx : Fourier.Fourier_quad(xx, self.__dx)
            self.__int_y_1D = lambda yy : Fourier.Fourier_quad(yy, self.__dy)

            self.__int_cum_x_1D = lambda xx : Fourier.Fourier_cumsum(xx, self.__dx)
            self.__int_cum_y_1D = lambda yy : Fourier.Fourier_cumsum(yy, self.__dy)
            
            self.__dx_1D = lambda xx : Fourier.Fourier_deriv(xx, order = 1, a = self.__x_bounds[0], b = self.__x_bounds[1])
            self.__dy_1D = lambda yy : Fourier.Fourier_deriv(yy, order = 1, a = self.__y_bounds[0], b = self.__y_bounds[1])

            self.__dy_1D_BC = self.__dy_1D
            self.__dx_1D_BC = self.__dx_1D

            self.__d2x_1D = lambda xx : Fourier.Fourier_deriv(xx, order = 2, a = self.__x_bounds[0], b = self.__x_bounds[1])
            self.__d2y_1D = lambda yy : Fourier.Fourier_deriv(yy, order = 2, a = self.__y_bounds[0], b = self.__y_bounds[1])

            self.__d2y_1D_BC = self.__d2y_1D
            self.__d2x_1D_BC = self.__d2x_1D
            
        elif self.__geometry == 'zonal_channel' : 
            self.x = np.linspace(self.__x_bounds[0], self.__x_bounds[1], self.__Nx, endpoint = False)
            self.y = Chebyshev.collocation_points(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1])
            self.__dx = self.x[1] - self.x[0]
            self.__dy = np.abs(self.y[1]-self.y[0])#les points sont les plus rapprochés aux bords

            self.__Dy = Chebyshev.Cheb_mat(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1])
            self.__D2y = Chebyshev.Cheb_mat(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1], M=2)

            self.__Dy_BC = Chebyshev.Cheb_mat(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1], Dirichlet_BC = True)
            self.__D2y_BC = Chebyshev.Cheb_mat(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1], Dirichlet_BC = True, M = 2)

            self.__dx_1D = lambda xx : Fourier.Fourier_deriv(xx, order = 1, a = self.__x_bounds[0], b = self.__x_bounds[1])
            self.__dy_1D = lambda yy : self.__Dy@yy

            self.__d2x_1D = lambda xx : Fourier.Fourier_deriv(xx, order = 2, a = self.__x_bounds[0], b = self.__x_bounds[1])
            self.__d2y_1D = lambda yy : self.__D2y@yy

            self.__dx_1D_BC = self.__dx_1D
            self.__dy_1D_BC = lambda yy : self.__Dy_BC@yy

            self.__d2x_1D_BC = self.__d2x_1D
            self.__d2y_1D_BC = lambda yy : self.__D2y_BC@yy

            self.__dealias_x_1D = lambda xx : Dealiasing.Fourier_dealias(xx, self.__dx, self.__dealias_params.dealias_Fourier_coeff)
            self.__dealias_y_1D = lambda yy : Dealiasing.exp_filter_DCT(yy,
                                                                        alpha = self.__dealias_params.dealias_exp_alpha,
                                                                        p = self.__dealias_params.dealias_exp_p)
            
            self.__int_x_1D = lambda xx : Fourier.Fourier_quad(xx, self.__dx)
            self.__int_y_1D = lambda yy : Chebyshev.Cheb_quad(yy, a = self.__y_bounds[0], b = self.__y_bounds[1])

            self.__int_cum_x_1D = lambda xx : Fourier.Fourier_cumsum(xx, self.__dx)
            self.__int_cum_y_1D = lambda yy : Chebyshev.Cheb_cumsum(yy, self.x)
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
        #fonctions de déaliasing
        self.__dealias_x = self.__apply_along_axis(self.__dealias_x_1D, "x")
        self.__dealias_y = self.__apply_along_axis(self.__dealias_y_1D, "y")
        #fonctions de dérivation
        dx = self.__apply_along_axis(self.__dx_1D, "x")
        dy = self.__apply_along_axis(self.__dy_1D, "y")
        dx_BC = self.__apply_along_axis(self.__dx_1D_BC, "x")
        dy_BC = self.__apply_along_axis(self.__dy_1D_BC, "y")
        #idem pour les dérivées secondes
        d2x = self.__apply_along_axis(self.__d2x_1D, "x")
        d2y = self.__apply_along_axis(self.__d2y_1D, "y")
        d2x_BC = self.__apply_along_axis(self.__d2x_1D_BC, "x")
        d2y_BC = self.__apply_along_axis(self.__d2y_1D_BC, "y")
        #####TODO : optimiser les fonctions appliquant un produit matriciel, en utilisant numba##############
        #####On met ça dans l'ordre#################################
        Deriv_axis = collections.namedtuple('Deriv_axis', ['di', 'di_BC', 'd2i', 'd2i_BC'])
        self.__liste_deriv_x = Deriv_axis(di = dx, di_BC = dx_BC, d2i = d2x, d2i_BC = d2x_BC)
        self.__liste_deriv_y = Deriv_axis(di = dy, di_BC = dy_BC, d2i = d2y, d2i_BC = d2y_BC)
    
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
        "Intègre une fonction sur seul axe ('x', 'y'). Renvoie un array de meme taille que val. Je suis franchement pas sur"
        self.__check_shape(val)
        if axis == 'x':
            return self.__int_cum_x(val)
        elif axis == 'y':
            return self.__int_cum_y(val)
        else:
            raise ValueError("axis must be in ('x', 'y')")
        
    def integrate_all_domain(self, val):
        "intègre une fonction sur toute la grille. Renvoie un scalaire"
        self.__check_shape(val)
        I_x = np.zeros(self.__Ny)
        
        for i in range(0, self.__Ny):
            I_x[i] = self.__int_x_1D(val[:, i])
            
        I = self.__int_y_1D(I_x)
        return I
        
    def dealias(self, value, axis):
        "Dé-aliase_selon une ou plusieurs directions (utile si le modèle inclut des produits de termes)"
        if axis == "x" : 
            return self.__dealias_x(value)
        elif axis == "y" : 
            return self.__dealias_y(value)
        elif axis == 'all' : ###ATTENTION, NON FIXE, JE SAIS PAS SI C'EST COMMUTATIF !!!!
            return self.__dealias_y(self.__dealias_x(value))
        else :
            raise ValueError(f"axis must be in ('x', 'y', 'all'), current value : {axis}")

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
        
        if self.__dealias_params.dealias_order == 'before_product':
            dx_A = self.dealias(dx_A, "x")
            dx_B = self.dealias(dx_B, "x")
            dy_A = self.dealias(dy_A, "y")
            dy_B = self.dealias(dy_B, "y")

        dxA_dyB = dx_A*dy_B
        dyA_dxB = dy_A*dx_B
        if self.__dealias_params.dealias_order == 'after_product':
            dx_A_dy_B = self.dealias(dx_A_dy_B, "all")
            dyA_dxB = self.dealias(dyA_dxB, "all")
        return dxA_dyB - dyA_dxB

    def __repr__(self):
        str_out = f"""
        Grid : \n
        Geometry = {self.__geometry}\n
        -Size ({self.__Nx}, {self.__Ny})\n
        -Step : dx = {self.__dx}, dy = {self.__dy}\n
        -Intervals : x = {self.__x_bounds}, y = {self.__y_bounds} \n
        -DealiasParams : {self.__dealias_params}
        """
        return str_out
###########Fonctions privées########################################################
    def __apply_along_axis(self, func, axis):
        """
        applique une fonction de la forme f(x) sur un axe donné ('x', 'y') ;
        x est un tableau de taille (Nx, Ny) ; renvoie aussi un tableau de taille Ny. Utilisé pour la dérivation et le déaliasing 
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