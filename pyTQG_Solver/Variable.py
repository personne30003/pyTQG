"""
Classe permettant de documenter les variables : nom, type (prognostique, diagnostique), CLs, attributs (sert pour la doc). Sert surtout pour Grid afin de déterminer quels opérateurs sont nécessaires

"""

import dataclasses


@dataclasses.dataclass(frozen = True, slots = True)
class Variable:
    name : str
    type : str#'diagnostic', 'prognostic'
    units : str=''
    BCs : bool = False
    integrated : bool = False
    attrs : str = ''

    def __post_init__(self):
        if self.type not in ('prognostic', 'diagnostic'):
            raise ValueError(f"attribute 'type' must be in ('prognostic', 'diagnostic'), current_value = {self.type} ")
    
