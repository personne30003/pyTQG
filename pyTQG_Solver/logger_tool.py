"""
module utilitaire (n'apparaissant pas dans le diagramme de classe)
Dirige la sortie à la fois vers un fichier et dans un terminal

Suppose que le fichier est déjà ouvert
"""
import sys
import logging

class Logger:
    def __init__(self, file_name):
        self.file_name = file_name
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers :
            formatter = LoggerFormatter()

            handler_console = logging.StreamHandler(sys.stdout)
            handler_console.setFormatter(formatter)
            self.logger.addHandler(handler_console)

            handler_file = logging.FileHandler(self.file_name,
                                               mode = 'w',
                                               encoding = 'utf-8')
            handler_file.setFormatter(formatter)
            self.logger.addHandler(handler_file)

            #capture des warnings
            logging.captureWarnings(True)

    def print(self, message, mode = 'info', exception = None):
        "Remplace la fonction print"
        if mode == 'info':
            self.logger.info(message)
        elif mode == 'warning' :
            self.logger.warning(message)
        elif mode == 'error' :
            self.logger.error(message, exc_info = exception)
        else :
            raise ValueError(f"mode = {mode} must be in ('info', 'warning', 'error')")

class LoggerFormatter(logging.Formatter):
    "Formatteur dynamique qui adapte la struture de l'affichage selon le type de message"
    def format(self, record):
        format_origin = self._style._fmt
        if record.levelno == logging.INFO :
            self._style._fmt = "%(message)s"
        elif record.levelno == logging.WARNING :
            self._style._fmt = "[WARNING] - %(message)s"
        elif record.levelno == logging.ERROR :
            self._style._fmt = "[ERROR] - %(message)s"
        else :
            self._style._fmt = "[UNKNOWN LEVEL] - %(message)s"
        result = super().format(record)
        self._style._fmt = format_origin
        return result
