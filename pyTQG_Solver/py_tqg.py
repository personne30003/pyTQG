"""
classe Principale, qui sera appelée dans le script de sortie

"""

import os
import sys
import pathlib
import shutil
import time
import warnings
import logging
import numpy as np

import output
import time_scheme
import parameters
import grid
import logger_tool
import qg_model


class pyTQG:

    def __init__(self,
                 params : parameters.Params,
                 Grid: grid.Grid,
                 Model : qg_model.QG_model):
        self.__Grid = Grid
        self.__Model = Model
        self.__Params = params

        self.__exp_dir = pathlib.Path(params.exp_dir)
        self.__exp_name = params.exp_name
        self.__output_dir = self.__exp_dir/self.__exp_name

        self.__create_output_folder()
        #self.__copy_script()

        self.__Logger = logger_tool.Logger(self.__output_dir/"log.txt")

        self.__Logger.print(20 * '#' + 'pyTQG' + 20 * '#')
        self.__Logger.print(45 * '#')
        self.__Logger.print(20 * '#' + 'Params' + 20* '#')
        self.__Logger.print(params)
        self.__Logger.print(20 * '#' + 'Grid' + 20* '#')
        self.__Logger.print(self.__Grid)
        self.__Logger.print(20 * '#' + 'Model' + 20 * '#')
        self.__Logger.print(self.__Model)
        self.__Logger.print(45 * '#')

        self.__Logger.print(f"Output files creation")
        self.__Output = output.Output(params,
                                      self.__Model,
                                      self.__Grid,
                                      self.__output_dir)
        self.__Logger.print("done")

        self.__Logger.print(f"Checking initial fields")
        self.__Model.check_init_fields()
        self.__Logger.print("done")
        self.__TimeScheme = time_scheme.TimeScheme(params,
                                                   self.__Model.RHS,
                                                   self.__Model.State,
                                                   copy_initial = False
                                                   )
        self.__dt_fix = self.__Params.dt
        self.__adaptable_dt = self.__Params.adaptable_dt
        self.__max_speed = self.__Params.max_speed
        self.__cfl = self.__Params.cfl

    def loop(self):
        t = 0.0
        t_max = self.__Params.max_time
        self.__TimeScheme.reset_t()

        N_it = 0

        freq_his = self.__Params.freq_his
        freq_diags = self.__Params.freq_diags

        t_his = freq_his
        t_diags = freq_diags

        stop = False

        self.__Logger.print(15 * '#' + 'iteration : 0, t = 0'+15 * '#')
        self.__Model.compute_diagnostics()
        max_speed = self.__Model.max_speed()
        dt = self.__estimate_dt(max_speed)
        self.__Logger.print(f"max speed = {max_speed}, dt = {dt}")
        self.__Logger.print(self.__Model.disp_diags())
        self.__Logger.print("saving fields")
        self.__Output.save_his(0.0)
        self.__Logger.print("saving diagnostics")
        self.__Output.save_diags(0.0)
        self.__Logger.print(45 * '#')

        list_time_it = []
        t_ini = time.time()
        while (t <= t_max) and not stop:
            t0 = time.time()
            t += dt
            N_it += 1

            self.__TimeScheme.Step(dt)
            self.__Model.compute_diagnostics()

            max_speed = self.__Model.max_speed()
            dt = self.__estimate_dt(max_speed)

            self.__Logger.print(15 * '#' + f'iteration : {N_it}, t = {t}' + 15 * '#')
            self.__Logger.print(f"max speed = {max_speed}, dt = {dt}")
            self.__Logger.print(self.__Model.disp_diags())

            if t >= t_his:
                t_his += freq_his
                self.__Logger.print("saving fields")
                self.__Output.save_his(t)

            if t >= t_diags:
                t_diags += freq_diags
                self.__Logger.print("saving diags")
                self.__Output.save_diags(t)

            t_tot = time.time()
            t_it = 1000*(t_tot - t0)#conversion des secondes en ms
            list_time_it.append(t_it)

            if max_speed >= self.__max_speed :
                self.__Logger.print(f"max speed >= {self.__max_speed}, blow-up detected, stopping",
                                    mode = 'error')
                stop = True
            self.__Logger.print(f"iteration done in {t_it} ms")
            self.__Logger.print(f"time elapsed since the start of the simulation {(t_tot - t_ini)/60} minutes")
            self.__Logger.print(45 * '#')

        t_sim_tot = (time.time() - t_ini)/1000
        self.__Logger.print(f"time elapsed since the start of the simulation {(t_sim_tot - t_ini)/60} minutes")
        mean_time_iteration = np.mean(list_time_it)
        self.__Logger.print(f"mean iteration time : {mean_time_iteration} ms")
        logging.shutdown()
        sys.exit(0)




    def __create_output_folder(self):
        if not self.__exp_dir.is_dir():
            raise FileNotFoundError(f"no directory {self.__exp_dir}")
        if self.__output_dir.exists() :
            warnings.warn(f"directory {self.__output_dir} already exist. It content will be erased",
                          RuntimeWarning)
            shutil.rmtree(self.__output_dir)
        self.__output_dir.mkdir(parents = False, exist_ok = False)

    def __copy_script(self):
        name_script = pathlib.Path(__file__).name
        current_directory = pathlib.Path.cwd()
        shutil.copy(current_directory/name_script,
                    self.__output_dir/name_script)

    def __estimate_dt(self, U_max):
        if self.__adaptable_dt :
            if U_max > 0.0:
                dt = self.__Params.cfl * self.__Grid.dx_min/U_max
            else :
                warnings.Warn(f"zero max speed : dt = params.dt_fix = {self.__dt_fix}",
                              RuntimeWarning)
                dt = self.__dt_fix
        else:
            dt = self.__dt_fix
        return dt




