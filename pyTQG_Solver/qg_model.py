"""
Classe parente avec les fonctions et attributs communs des autres classes implémentant les modèles.
C'est spécifique aux modèles type QG
"""
import numpy as np

import grid
import state
import parameters

import helmholtz_channel
import helmholtz_biperiodic

class QG_model:
    def __init__(self, params : parameters.Params, Grid : grid.Grid):

        self.State = state.State()
        self._Grid = Grid
        self._geometry = params.geometry

        self._model_name = 'QG'
        self.beta = 0.0
        self._liste_NC_attrs = ["beta"]
        #A raffiner
        if self._geometry == 'biperiodic':
            self._EllipticSolver = helmholtz_biperiodic.HelmholtzBiperiodic(params.Nx,
                                                                            params.Ny,
                                                                            x_bounds = params.x_bounds,
                                                                            y_bounds = params.y_bounds)
        elif self._geometry == 'zonal_channel' :
            self._EllipticSolver = helmholtz_channel.HelmholtzChannel(params.Nx,
                                                                      params.Ny,
                                                                      x_bounds = params.x_bounds,
                                                                      y_bounds = params.y_bounds,
                                                                      BC_y=('Dirichlet', 'Dirichlet'))
        elif self._geometry == 'basin' :
            raise NotImplementedError("basin geometry not (yet ?) implemented")
        else : 
            raise ValueError(f"geometry parameter must be in {params.list_geometry}")

    def RHS(self, state, t):
        raise NotImplementedError

    def psi_from_PV(self, q, assign = False):
        raise NotImplementedError

    def PV_from_psi(self, psi, assign = False):
        raise NotImplementedError

    def vort_from_PV(self, PV = None):
        raise NotImplementedError

    def PV_from_vort(self, vort = None, assign = False):
        raise NotImplementedError

    def max_speed(self):
        raise NotImplementedError

    def compute_diagnostics(self):
        raise NotImplementedError

    def psi_from_U(self, U : np.ndarray, constant = 0.0):
        "Renvoie la fonction de courant psi(x, y) à partir d'un profil de vitesse zonale u(x, y) = U(y)"
        psi = -self._Grid.int_cum(U, 'y') + constant
        return psi

    def vort_from_U(self, U : np.ndarray):
        "Calcule la vorticité à partir d'un champ de vitesse zonale U"
        return -self._Grid.derivative(U, 'y', order = 1, BC_Dirichlet = False)

    def __repr__(self):
        str_out = f"Model : {self._model_name} \n"
        str_out += "prognostics variables : \n"
        for prognostic_var in self.State.prognostics_vars :
            str_out += f" - {prognostic_var}, field = {self.State[prognostic_var].field} \n"
        str_out += 'diagnostics variables : \n'
        for diagnostic_var in self.State.diagnostics_vars :
            str_out += f" - {diagnostic_var}, field = {self.State[diagnostic_var].field} \n"
        str_out += "parameters : \n"
        for attr in self._liste_NC_attrs:
            str_out += f" - {attr} = {getattr(self, attr)}\n"
        str_out += self._EllipticSolver.__repr__()
        return str_out

    def disp_diags(self):
        str_out = f"Model {self._model_name} diagnostics : \n"
        for diag, val in self.State.scalar_values.items() :
            str_out += f" - {diag} = {val} \n"
        return str_out

    def to_NETCDF_attrs(self):
        "Met certains parametres sous forme d'un dictionnaire {nom:valeur}"
        dic_attrs = {'Model' : self._model_name}
        for attr in self._liste_NC_attrs :
            attr_value = getattr(self, attr)
            if type(attr_value) == bool:
                dic_attrs[attr] = str(attr_value)
                continue
            dic_attrs[attr] = attr_value

        return dic_attrs

    def assign_init_field(self, var, value : np.ndarray):
        if var not in self.State.fields_vars :
            raise ValueError(f"field var must be in {self.State.fields_vars}")
        if var.shape != self._Grid.X.shape:
            raise ValueError(f"var must be np.ndarray of shape {self._Grid.X.shape}")
        self.State[var].value = value

    def check_init_fields(self):
        "verifie si les champs ont bien été initialisés"
        for field_name, field_value in self.State.fields_values.items() :
            if type(field_value) is not np.ndarray :
                raise TypeError(f"field {field_name} must be np.ndarray")
            if field_value.shape != self._Grid.shape :
                raise ValueError(f"field {field_name} must have same Grid shape {self._Grid.shape}, not {field_value.shape}")