"""
Modèle TQG 2 couches couplées (cf mail X.Carton)


"""

import parameters
import grid
import variable
import qg_model


class ThermalQG_coupled(qg_model.QG_model) :
    def __init__(self, params : parameters.Params, Grid : grid.Grid):
        super().__init__(params, Grid)


    def RHS(self, state, t):
        raise NotImplementedError

    def psi_from_PV(self, q, assign = False):
        raise NotImplementedError

    def PV_from_psi(self, psi, assign = False):
        raise NotImplementedError
    def U_max(self):
        raise NotImplementedError

    def compute_diagnostics(self):
        raise NotImplementedError