"""
Classe regroupant les opérateurs de dérivation, intégration et déaliasing

"""


import Params
import Fourier
import Chebyshev
import Dealiasing
import numpy as np
import scipy

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

        #####A faire : clarifier le message###########
        if self.__x_bounds[0] >= self.__x_bounds[1]:
            raise ValueError(f"x_bounds[0] must not be >= x_bounds[1]")
        if self.__y_bounds[0] >= self.__y_bounds[1]:
            raise ValueError(f"y_bounds[0] must not be >= y_bounds[1]")
        ####On génère la grille#######################
        if self.geometry == 'biperiodic':
            self.x = np.linspace(self.__x_bounds[0], self.__x_bounds[1], self.__Nx, endpoint = False)
            self.y = np.linspace(self.__y_bounds[0], self.__y_bounds[1], self.__Ny, endpoint = False)
            self.dx = self.x[1] - self.x[0]
            self.dy = self.x[1] - self.x[0]
        elif self.geometry == 'zonal_channel' : 
            self.x = np.linspace(self.__x_bounds[0], self.__x_bounds[1], self.__Nx, endpoint = False)
            self.y = Chebyshev.collocation_points(self.__Ny, a = self.__y_bounds[0], b = self.__y_bounds[1])
            self.dx = self.x[1] - self.x[0]
            self.dy = np.abs(self.y[1]-self.y[0])#les points sont les plus rapprochés aux bords
        elif self.geometry == 'basin':
            raise NotImplementedError("configuration bassin not yet implemented")
        else:
            raise ValueError(f"params.geometry must be in {params.list_geometry}, current value {self.geometry}")
        
        self.variables = []

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing = 'ij')
        
        #fonction d'intégration sur tout le plan
        self.integrate = self.__integrate_func()
        
    def add_variable(self, var):
        pass
    
    def __getitem__(self, item):
        pass

    def dealias(self, value, axis):
        pass

    def jacobien(self, var_A, val_A, var_B, val_B, BC_A = False, BC_B = True):
        pass

    def __apply_along_axis(self, func, axis):
        "applique une fonction de la forme f(x) sur un axe donné ('x', 'y') "
        axis_sel = None
        if axis == "x" : 
            axis_sel = 0
        elif axis == "y" : 
            axis_sel = 1
        else :
            raise ValueError(f"axis must be in ('x', 'y'), current value : {axis}")

    def __integrate_func(self):
        int_x = None
        int_y = None
        if self.geometry == 'biperiodic':
            int_x = lambda xx : Fourier.Fourier_quad(xx, self.dx)
            int_y = lambda yy : Fourier.Fourier_quad(yy, self.dy)
        if self.geometry == 'zonal_channel':
            int_x = lambda xx : Fourier.Fourier_quad(xx, self.dx)
            int_y = lambda yy : Chebyshev.Cheb_quad(yy, a = self.__y_bounds[0], b = self.__y_bounds[1])

        def integrate(val):
            "intègre une fonction sur toute la grille"
            if type(val) != np.ndarray:
                raise ValueError(f"val must be np.ndarray, current type = {type(val)}")
            if val.shape != (self.__Nx, self.__Ny):
                raise ValueError(f"val.shape must be equal to (Nx, Ny)= ({self.__Nx}, {self.__Ny}), current shape = ({val.shape[0]}, {val.shape[1]})")
            
            I_x = 0.0

            for i in range(0, self.__Ny):
                I_x += int_x(val[:, i])
            
            I = int_y(I_x)
            return I

        return integrate