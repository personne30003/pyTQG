"""
Classe générique permettant de documenter les variables : nom, type (prognostique, diagnostique), CLs, attributs (sert pour la doc)

"""


class Variable:
    def __init__(self, name, params,value = 0.0, type_var = 'diagnostic', BC_x = 'Dirichlet', BC_y = 'Dirichlet', attrs=''):
        self.name = name
        self.type = type_var
        self.BC_x = BC_x
        self.BC_y = BC_y
        self.attrs = attrs
        self.value = 0.0
    def __repr__(self):
        pass