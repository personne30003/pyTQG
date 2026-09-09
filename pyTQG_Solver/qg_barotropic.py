"""
Modele QG barotrope (1 couche active sur 1 couche passive)  :
(d2/dx^2+d2/dy^2)psi - psi/(R_d^2) = q - beta*y
d q /d t = -Jac(psi, q)

vorticité relative :

zeta = (d2/dx^2+d2/dy^2)psi  = q - beta * y + psi/(R_d^2)

diagnostics (grandeurs intégrées) :
-Energie cinetique totale :
E_c =  1 / 2 ( (d psi / d x)^2 + (d psi / d y) ^2)

-Enstrophie totale :
Z = q^2

-PV totale :
Q = q

"""

import numpy as np

import parameters
import grid
import variable
import qg_model



class BarotropicQG(qg_model.QG_model):
    def __init__(self, params : parameters.Params, Grid : grid.Grid):
        super().__init__(params, Grid)
        psi = variable.Variable(name='psi',
                                type_var='diagnostic',
                                field=True)
        PV = variable.Variable(name='PV',
                               type_var='prognostic',
                               field=True)
        vort = variable.Variable(name='vorticity',
                                 type_var='diagnostic',
                                 field=True)
        #diagnostics
        kinetic_energy = variable.Variable(name='kinetic_energy',
                                           type_var='diagnostic',
                                           field=False)
        potential_energy = variable.Variable(name='potential_energy',
                                             type_var='diagnostic',
                                             field=False)

        total_energy = variable.Variable(name='total_energy',
                                         type_var='diagnostic',
                                         field=False)

        PV_total = variable.Variable(name='PV_total',
                                     type_var='diagnostic',
                                     field=False)

        enstrophy = variable.Variable(name='enstrophy',
                                      type_var='diagnostic',
                                      field=False)
        self.State.add_variables(psi,
                                 vort,
                                 PV,
                                 kinetic_energy,
                                 potential_energy,
                                 total_energy,
                                 PV_total,
                                 enstrophy)
        self.inv_Rd2 = 0.0
        self.T_0 = 0.0  #Transport moyen (en configuration canal)
        self._model_name = 'BarotropicQG'

        self._liste_NC_attrs += ['inv_Rd2', 'T_0']

        self._EllipticSolver.alpha2 = ('inv_Rd2', self.inv_Rd2)


    def RHS(self, state, t):
        new_state = state.copy()
        new_state['psi'].value = self.psi_from_PV(state['PV'].value)
        new_state['PV'].value = -self._Grid.jacobien(new_state['psi'].value,
                                                     state['PV'].value,
                                                     BC_A=False,
                                                     BC_B=False)
        new_state['vorticity'].value = self.vort_from_PV(new_state['PV'].value)
        return new_state

    def U_max(self):
        u = self._Grid.derivative(self.State['psi'].value, "x")
        v = self._Grid.derivative(self.State['psi'].value, "y")
        U = np.sqrt(u ** 2 + v ** 2)
        return np.max(U)

    def psi_from_PV(self, vort, assign = False):
        psi = self._EllipticSolver.Solve(vort - self.beta * self._Grid.Y,
                                         'inv_Rd2',
                                         bc_y_inf=0.0,
                                         bc_y_sup=self.T_0)
        if assign :
            self.State['psi'].value = psi
        return psi


    def PV_from_psi(self, psi, assign = False):
        PV = self._Grid.laplacien(psi) - self.inv_Rd2 * psi + self.beta * self._Grid.Y

        if assign :
            self.State['PV'].value = PV
            self.State['vorticity'] = self.vort_from_PV(PV)
        return PV

    def vort_from_PV(self, PV = None):
        pv = self.State['PV'].value
        if PV is not None:
            pv = PV
        vort = pv - self.beta*self._Grid.Y + self.State['psi'].value*self.inv_Rd2
        return vort

    def compute_diagnostics(self):
        u = self._Grid.derivative(self.State['psi'].value, "x")
        v = self._Grid.derivative(self.State['psi'].value, "y")
        self.State['kinetic_energy'].value = 0.5 * self._Grid.integrate(u ** 2 + v ** 2, "all")
        self.State['potential_energy'].value = 0.5 * self.inv_Rd2 * self._Grid.integrate(self.State['psi'].value ** 2,
                                                                                         'all')
        self.State['total_energy'].value = self.State['kinetic_energy'].value + self.State['potential_energy'].value
        self.State['PV_total'].value = self._Grid.integrate(self.State['PV'].value, "all")
        self.State['enstrophy'].value = 0.5 * self._Grid.integrate(self.State['PV'].value ** 2, "all")
