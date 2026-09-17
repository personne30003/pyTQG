"""

Classe dédiée au stockage des variables diagnostiques et pronostiques
Permet aussi d'appliquer les méthodes d'intégration temporelle : 
les opérateurs __add__ et __mul__ (par un scalaire) ne concernent que les variables prognostiques

"""

import variable
import copy
import numpy as np
import collections


class State:
    def __init__(self):
        self.__dic_var = {}# Dictionnaire de la forme {var.name : var}
        # liste de clés
        self.__liste_vars = []

    def add_variables(self, *args):
        #détection des doublons dans *args
        liste_duplications_args = self.__check_duplications(args)
        if len(liste_duplications_args) > 0 :
            raise ValueError(f"duplications in input variable : {liste_duplications_args}")
            
        for var in args : 
            if type(var) != variable.Variable :
                raise ValueError(f"*args must be Variable.Variable, not {type(var)}")
            self.__dic_var[var.name] = var.copy()


            if var.name in self.__liste_vars:
                raise ValueError(f"var {var.name} already in State {self.__liste_vars}")

        self.__liste_vars = list(self.__dic_var.keys())

    def del_variables(self, *args):
        if len(args) < 1:
            raise ValueError("there must be 1 or more variables to delete")

        if not set(args).issubset(self.__liste_vars) :
            raise ValueError(f"{args} must be in {self.__liste_vars}")

        for i in range(0, len(self.__liste_vars)):
            var_i = self.__liste_vars[i]
            if var_i in args :
                del self.__dic_var[var_i]
        self.__liste_vars = list(self.__dic_var.keys())

    def assign_fields_prognostic(self, other):
        self.__check_other(other)
        for var in self.prognostics_vars :
            self.__dic_var[var].value[:] = other[var].value


########Operateurs mathématiques#########################
    def __add__(self, other):
        if not isinstance(other, State):
            raise TypeError(f"sum only compatible with State, not {type(other)}")

        self.__check_other(other)
        new_S = self.copy()
        
        for var in self.prognostics_vars :
            new_S[var].value = new_S[var].value+other[var].value
        return new_S

    def __iadd__(self, other):
        if not isinstance(other, State):
            raise TypeError(f"sum only compatible with State, not {type(other)}")

        self.__check_other(other)

        for var in self.prognostics_vars :
            self.__dic_var[var].value += other[var].value
        return self
        
        
    def __mul__(self, scalar):
        if not np.isscalar(scalar) or isinstance(scalar, bool) or isinstance(scalar, str): 
            raise ValueError(f"scalar must be int, float or complex, not {type(scalar)}")
            
        new_S = self.copy()
        
        for var in self.prognostics_vars :
            new_S[var].value  = scalar*new_S[var].value
            
        return new_S

    def __rmul__(self, scalar) : 
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return self.__mul__(1./scalar)

    def __sub__(self, other):
        return self.__add__(-1.*other)#Attention, ne fonctionne que pour des flottants
        
    def __getitem__(self, var):
        
        if var not in self.__dic_var.keys() : 
            raise KeyError(f"key {var} must be in {self.__dic_var.keys()}")
        return self.__dic_var[var]
        #return copy.deepcopy(self.__dic_var[var])

    def __setitem__(self, var, value):
        if var not in self.__dic_var.keys() : 
            raise KeyError(f"key {var} must be in {self.__dic_var.keys()}")
            
        if var in self.scalar_vars and not np.isscalar(value):
            raise ValueError(f"cannot assign {type(value)} to scalar")
            
        if var in self.fields_vars and not isinstance(value, np.ndarray):
            raise ValueError(f"cannot assign {type(value)} to numpy.ndarray")
            
        self.__dic_var[var].value = value
            
    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return f""" 
        State : 
        - data_vars = {self.__liste_vars}
        - prognostics_vars = {self.prognostics_vars}
        - diagnostics_vars = {self.diagnostics_vars}
        - arrays vars = {self.fields_vars}
        - fields vars = {self.scalar_vars}
        - values = {self.values}
        """
    
    def __check_other(self, other):
        "Vérifie si un autre State a le meme nombre de variables que State. Les variables doivent également etre identiques"
        if collections.Counter(self.data_vars) != collections.Counter(other.data_vars):
            raise ValueError("State and other State have differents variables")
        if not all(self[var] == other[var] for var in self.__liste_vars) : 
            raise ValueError("State's variable and other's variables must be equals")
    
    def __check_duplications(self, liste_values):
        liste_view = []
        liste_duplications = []
        for value in liste_values:
            if value in liste_view:
                if value not in liste_duplications:
                    liste_duplications.append(value)
            else:
                liste_view.append(value)
        return liste_duplications
        
    @property
    def data_vars(self):
        return self.__liste_vars
        
    @property
    def prognostics_vars(self):
        return [var for var in self.__liste_vars if self.__dic_var[var].type == 'prognostic']

    @property
    def diagnostics_vars(self):
        return [var for var in self.__liste_vars if self.__dic_var[var].type == 'diagnostic']

    @property
    def scalar_vars(self):
        return [var for var in self.__liste_vars if not self.__dic_var[var].field]

    @property
    def fields_vars(self):
        return [var for var in self.__liste_vars if self.__dic_var[var].field]

    @property
    def values(self):
        return {var : self.__dic_var[var].value for var in self.__liste_vars}

    ######Serviront pour sauvegarder les variables dans le fichier de sortie##########
    @property
    def fields_values(self):
        return {var : self.__dic_var[var].value for var in self.fields_vars}

    @property
    def scalar_values(self):
        return {var : self.__dic_var[var].value for var in self.scalar_vars}

    @property
    def fields_attrs(self):
        return {var : self.__dic_var[var].attrs for var in self.fields_vars}

    @property
    def scalar_attrs(self):
        return {var : self.__dic_var[var].attrs for var in self.scalar_vars}