"""
Classe parente avec les fonctions et attributs communs des autres classes implémentant les modèles.
C'est spécifique aux modèles type QG
"""

import Grid
import State
import HelmholtzChannel
import HelmholtzBiperiodic

class Model:
    def __init__(self, Grid, params):
        self.beta = 0.0
        self.State = State.State()
        self.geometry = params.geometry
        #A raffiner
        if self.geometry == 'biperiodic':
            self.EllipticSolver = HelmholtzBiperiodic(params.Nx,
                                                      params.Ny,
                                                      x_bounds = params.x_bounds,
                                                      y_bounds = params.y_bounds)
        elif self.geometry == 'zonal_channel' :
            pass
        elif self.geometry == 'basin' : 
            raise NotImplementedError("basin geometry not (yet ?) implemented")
        else : 
            raise ValueError(f"geometry parameter must be in {params.list_geometry}")

    def RHS(state, t):
        raise NotImplementedError

    def U_max(self):
        raise NotImplementedError

    def compute_diagnostic(self):
        raise NotImplementedError

    def to_NETCDF_attrs(self):
        "A voir si j'implémente ça ici ou non."
        raise NotImplementedError
    