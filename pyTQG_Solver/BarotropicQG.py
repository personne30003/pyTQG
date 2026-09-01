"""
Modele QG barotrope (1 couche active sur 1 couche passive)  :
q = (d2/dx^2+d2/dy^2)psi - psi/(R_d^2)+beta*y
d q /d t = -Jac(psi, q)

diagnostics :
-Energie cinetique totale
-Enstrophie totale
-PV totale
"""

import Variable
import Grid
import State
import Model
import HelmholtzChannel
import HelmholtzBiperiodic

class BarotropicQG(Model.Model):
    def __init__(self):
        super().__init__()
        psi = Variable.Variable(name = 'psi',
                                type_var = 'diagnostic',
                                field = True)
        q = Variable.Variable(name = 'q',
                              type_var = 'prognostic',
                              field = True)
        #diagnostics
        kinetic_energy = Variable.Variable(name = 'kinetic_energy', 
                                           type_var = 'diagnostic',
                                           field = False)
        PV_total = Variable.Variable(name = 'PV_total', 
                              type_var = 'diagnostic',
                              field = False)
        
        enstrophy = Variable.Variable(name = 'E_c', 
                                      type_var = 'diagnostic',
                                      field = False)
        self.State.add_variables(psi, q, kinetic_energy, PV_total, enstrophy)
        self.inv_Rd2 = 0.O
        
    def U_max(self):
        u = self.Grid.derivative(self.State['psi'].value, "x")
        v = self.Grid.derivative(self.State['psi'].value, "y")
        U = np.sqrt(u**2 + v**2)
        return np.max(U)

    def RHS(self, state, t):
        pass

    def psi_from_vort(self):
        pass

    def compute_diags(self):
        u = self.Grid.derivative(self.State['psi'].value, "x")
        v = self.Grid.derivative(self.State['psi'].value, "y")
        self.State['kinetic_energy'].value = 0.5*self.Grid.integrate(u**2+v**2, "all")
        self.State['PV_total'].value = self.Grid.integrate(self.State['q'].value, "all")
        self.State['PV_total'].value = self.Grid.integrate(self.State['q'].value**2, "all")
        
        
        