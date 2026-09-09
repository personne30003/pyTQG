"""
Modele TQG :
(d2/dx^2+d2/dy^2)psi - psi/(R_d^2) = q -theta/R_d^2- beta*y
d q / d t = -Jac(psi, q) + Jac(psi, theta)/R_d^2
d theta / d t = -Jac(psi, theta)

vorticité relative :

zeta = (d2/dx^2+d2/dy^2)psi  = q - beta * y + (psi - theta)/(R_d^2)

Diagnostics (grandeurs intégrées):

-Energie cinetique totale :
E_c =  1 / 2 ( (d psi / d x)^2 + (d psi / d y) ^2)

-Enstrophie totale :
Z = q^2

-PV totale :
Q = q

Flottabilité totale :
Theta = theta

Energie potentielle totale :
E_p = psi^2/(2 R_d^2)

Energie totale :
E = E_c + E_p

"""

import numpy as np

import parameters
import grid
import variable
import qg_model



class ThermalQG(qg_model.QG_model):
    def __init__(self, params : parameters.Params, Grid : grid.Grid):
        super().__init__(params, Grid)
        psi = variable.Variable(name='psi',
                                type_var='diagnostic',
                                field=True)
        PV = variable.Variable(name = 'PV',
                               type_var = 'prognostic',
                               field = True)
        vort = variable.Variable(name='vorticity',
                                 type_var='diagnostic',
                                 field=True)
        theta = variable.Variable(name = 'theta',
                                  type_var = 'prognostic',
                                  field = True)
        PV_total = variable.Variable(name = 'PV_total',
                                     type_var = 'diagnostic',
                                     field = False)
        enstrophy = variable.Variable(name = 'enstrophy',
                                      type_var = 'diagnostic',
                                      field = False)
        theta_total = variable.Variable(name = 'theta_total',
                                        type_var = 'diagnostic',
                                        field = False)
        kinetic_energy = variable.Variable(name = 'kinetic_energy',
                                           type_var = 'diagnostic',
                                           field = False)
        potential_energy = variable.Variable(name = 'potential_energy',
                                             type_var ='diagnostic',
                                             field = False)
        total_energy = variable.Variable(name = 'total_energy',
                                         type_var = 'diagnostic',
                                         field = False)
        self.State.add_variables(psi,
                                 vort,
                                 PV,
                                 theta,
                                 PV_total,
                                 enstrophy,
                                 theta_total,
                                 kinetic_energy,
                                 potential_energy,
                                 total_energy)

        self.inv_Rd2 = 0.0
        self.T_0 = 0.0  # Transport moyen (en configuration canal)
        self._model_name = 'Thermal_QG'

        self._liste_NC_attrs += ['inv_Rd2', 'T_0']

        self._EllipticSolver.alpha2 = ('inv_Rd2', self.inv_Rd2)

    def RHS(self, state, t):
        new_State = state.copy()
        new_State['psi'].value = self.psi_from_PV(state['PV'].value,
                                                    state['theta'].value)
        jac_psi_PV = self._Grid.jacobien(new_State['psi'].value,
                                         new_State['PV'].value,
                                         BC_A = False,
                                         BC_B = False)
        jac_psi_theta_PV = self._Grid.jacobien(new_State['psi'].value,
                                               new_State['theta'].value,
                                               BC_A = False,
                                               BC_B = False)# Terme source
        jac_psi_theta_buo = self._Grid.jacobien(new_State['psi'].value,
                                                new_State['theta'].value,
                                                BC_A = False,
                                                BC_B = True)#apparait dans conservation de la flottabilité

        new_State['PV'].value = -jac_psi_PV + self.inv_Rd2 * jac_psi_theta_PV
        new_State['theta'].value = -jac_psi_theta_buo
        new_State['vort'].value = self.vort_from_PV(new_State['PV'].value)

        return new_State

    def psi_from_PV(self, q, theta = None, assign = False):
        if theta is None :
            theta = self.State['theta'].value.copy()
        psi = self._EllipticSolver.Solve(q - theta * self.inv_Rd2 - self.beta * self._Grid.Y,
                                         'inv_Rd2',
                                         bc_y_inf=0.0,
                                         bc_y_sup=self.T_0)
        if assign:
            self.State['psi'].value = psi
        return psi

    def PV_from_psi(self, psi, theta = None, assign = False):
        if theta is None:
            theta = self.State['theta'].value.copy()
        vort = self._Grid.laplacien(psi) - (psi - theta)*self.inv_Rd2 + self.beta * self._Grid.Y
        if assign:
            self.State['PV'] = vort
        return vort

    def vort_from_PV(self, PV = None):
        pv = self.State['PV'].value
        if PV is not None:
            pv = PV
        vort = pv - self.beta * self._Grid.Y + (self.State['psi'].value - self.State['theta'].value) * self.inv_Rd2
        return vort

    def max_speed(self):
        u = self._Grid.derivative(self.State['psi'].value, "x")
        v = self._Grid.derivative(self.State['psi'].value, "y")
        U = np.sqrt(u ** 2 + v ** 2)
        return np.max(U)

    def compute_diagnostics(self):
        self.State['PV_total'].value = self._Grid.integrate(self.State['PV'].value, 'all')
        self.State['enstrophy'].value = 0.5 * self._Grid.integrate(self.State['PV'].value**2, 'all')
        self.State['theta_total'].value = self._Grid.integrate(self.State['theta'].value, 'all')
        u = self._Grid.derivative(self.State['psi'].value, "x")
        v = self._Grid.derivative(self.State['psi'].value, "y")
        self.State['kinetic_energy'].value = 0.5 * self._Grid.integrate(u**2 + v**2, 'all')
        self.State['potential_energy'].value = 0.5 * self.inv_Rd2 * self._Grid.integrate(self.State['psi'].value**2,
                                                                                         'all')
        self.State['total_energy'].value = self.State['kinetic_energy'].value + self.State['potential_energy'].value