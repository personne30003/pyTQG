"""
Classe rassemblant les paramètres généraux de la simu. Gère aussi l'affichage de la sortie.
"""
import os
import sys
import datetime

class Params():
    def __init__(self, use_logger = False):
        self.Nx=100
        self.Ny=100
        self.time_scheme='Euler'
        self.max_time=1000.0
        self.max_it=1000000
        self.max_speed = 50.0#arbitraire, à modifier. Un dépassement de cette valeur engendre l'arret du programme
        self.list_configs = ('basin', 'biperiodic', 'channel')
        self.config = 'biperiodic'
        self.output_path=os.getcwd()
        self.date = datetime.datetime.now()
        date_frm = self.date.strftime("%Y:%m:%d-%H:%M:%S")
        self.exp_name = f'Exp_{date_frm}'
        self.use_logger = use_logger
        
        if use_logger :
            self.logger = self.__create_logger()
        else :
            self.logger = None

    def print(self, msg, type_msg ='INFO'):
        "Fonction d'affichage générique"
        if self.logger is None : 
            print(msg)
        else : 
            if type_msg == 'INFO':
                self.logger.info(msg)
            elif type_msg == 'WARNING':
                self.logger.warning(msg)
            elif type_msg == 'ERROR':
                self.logger.error(msg)
            else : 
                raise ValueError(f"type_msg = {type_msg} must be in ['INFO', 'WARNING', 'ERROR']")
        
    def __create_logger(self):
        "Créé un fichier log"
        logger = logging.getLogger(self.exp_name)
        logger.setLevel(logging.INFO)
        logger.setLevel(logging.WARNING)
        logger.setLevel(logging.ERROR)
        if logger.handlers:
            return logger

        formatter = logging.Formatter("%(asctime) %(levelname)s : %(message)s")
        
        #Sorties à la fois sur la console ET dans un fichier
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        
        logfile = logging.FileHandler(
            f"{self.case_name}.log", mode="w"
        )
        logfile.setFormatter(formatter)

        logger.addHandler(console)
        logger.addHandler(logfile)

        return logger

    def __str__(self):
        pass