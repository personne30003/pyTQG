"""
Classe permettant de documenter les variables : nom, type (prognostique, diagnostique), attributs (sert pour la doc). Sert surtout pour State et la classe dédiée aux sorties (je sais pas comment l'appeler)
"""

import dataclasses


@dataclasses.dataclass(frozen = True, slots = True)
class Variable:
    name : str
    type : str#'diagnostic', 'prognostic'
    units : str=''
    integrated : bool = False
    attrs : str = ''

    def __post_init__(self):
        if self.type not in ('prognostic', 'diagnostic'):
            raise ValueError(f"attribute 'type' must be in ('prognostic', 'diagnostic'), current_value = {self.type} ")
    
