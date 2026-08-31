"""
Classe permettant de documenter les variables : nom, type (prognostique, diagnostique), attributs (sert pour la doc), valeur. Sert surtout pour State et la classe dédiée aux sorties (je sais pas comment l'appeler)

Globalement valide. 
"""

#import numpy as np

class Variable:

    def __init__(self,
                 value = 0.0,
                 type_var = 'prognostic',
                 name = '',
                 units = '',
                 field = True,
                 attrs = '') : 
        self.value = value
        if type_var not in ('prognostic', 'diagnostic'):
            raise ValueError(f"type must be in ('prognostic, diagnostic'), not {type_var}")
            
        self.__type = type_var
        self.__name = name
        self.__units = units
        self.__attrs = attrs
        self.__field = field
    
    def __eq__(self, other):
        if not isinstance(other, Variable) : 
            raise ValueError(f"cannor compare Variable with {type(other)}")
            
        msk = ((self.__name == other.name) and
               (self.__type == other.type) and
               (self.__units == other.units) and
               (self.__field == other.field) and
               (self.__attrs == other.attrs))
        
        return msk

    def __repr__(self):
        return f"""Variable : 
        Immutables attributes : 
        - name = {self.__name}
        - type = {self.__type}
        - units = {self.__units}
        - field = {self.__field}
        - attrs = {self.__attrs}
        mutable attributes
        - value = {self.value}"""
    ##Accès aux attributs immutables (sauf valeur)
    @property
    def name(self):
        return self.__name

    @property
    def type(self):
        return self.__type

    @property
    def units(self):
        return self.__units

    @property
    def field(self):
        return self.__field

    @property
    def attrs(self):
        return self.__attrs