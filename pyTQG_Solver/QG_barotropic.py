"""
Modele QG barotrope (1 couche active sur 1 couche passive)  :
q = (d2/dx^2+d2/dy^2)psi - psi/(R_d^2)+beta*y
d q /d t = -Jac(psi, q)

diagnostics :
-Energie cinetique totale
-Enstrophie totale
-PV totale
"""

import numpy as np

import Params
import Grid
import Variable
import Model



class BarotropicQG(Model.Model):
    def __init__(self, params : Params.Params, grid : Grid.Grid):
        super().__init__(params, grid)
        psi = Variable.Variable(name='psi',
                                type_var='diagnostic',
                                field=True)
        PV = Variable.Variable(name='PV',
                               type_var='prognostic',
                               field=True)
        #diagnostics
        kinetic_energy = Variable.Variable(name='kinetic_energy',
                                           type_var='diagnostic',
                                           field=False)
        PV_total = Variable.Variable(name='PV_total',
                                     type_var='diagnostic',
                                     field=False)

        enstrophy = Variable.Variable(name='E_c',
                                      type_var='diagnostic',
                                      field=False)
        self.State.add_variables(psi, PV, kinetic_energy, PV_total, enstrophy)
        self.inv_Rd2 = 0.0
        self.T_0 = 0.0  #Transport moyen (en configuration canal)
        self._model_name = 'BarotropicQG'

        self._liste_NC_attrs += ['inv_Rd2', 'T_0']

        self._EllipticSolver.alpha2 = ('inv_Rd2', self.inv_Rd2)


    def RHS(self, state, t):
        new_state = state.copy()
        new_state['psi'].value = self.psi_from_vort(state['PV'].value)
        new_state['PV'].value = -self._Grid.jacobien(new_state['psi'].value, state['q'].value,
                                                    BC_A=False, BC_B=True)
        return new_state

    def U_max(self):
        u = self._Grid.derivative(self.State['psi'].value, "x")
        v = self._Grid.derivative(self.State['psi'].value, "y")
        U = np.sqrt(u ** 2 + v ** 2)
        return np.max(U)

    def psi_from_vort(self, vort, assign = False):
        psi = self._EllipticSolver.Solve(vort - self.beta * self._Grid.Y,
                                         'inv_Rd2',
                                         bc_y_inf=0.0,
                                         bc_y_sup=self.T_0)
        if assign :
            self.State['psi'].value = psi
        return psi

    def vort_from_psi(self, psi, assign = False):
        vort = self._Grid.laplacien(psi) - self.inv_Rd2 * psi + self.beta * self._Grid.Y

        if assign :
            self.State['q'].value = vort
        return vort
    def compute_diagnostics(self):
        u = self._Grid.derivative(self.State['psi'].value, "x")
        v = self._Grid.derivative(self.State['psi'].value, "y")
        self.State['kinetic_energy'].value = 0.5 * self._Grid.integrate(u ** 2 + v ** 2, "all")
        self.State['PV_total'].value = self._Grid.integrate(self.State['PV'].value, "all")
        self.State['PV_total'].value = self._Grid.integrate(self.State['PV'].value ** 2, "all")
