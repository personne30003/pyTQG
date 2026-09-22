"""
Quelque fonctions dédiées au post-traitement sous Jupyter Notebook. La plupart du temps, xarray fait très bien le
travail tout seul !

"""



import xarray as xr
import hvplot.xarray
import holoviews as hv
import panel as pn


import numpy as np
import scipy
import collections

import dealiasing
import grid

#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors

####### Réglages pour la fonction animate #####################
pn.extension(backend='bokeh')
hv.extension('bokeh')
hv.opts.defaults(
    hv.opts.QuadMesh(active_tools=['pan']),
    hv.opts.Image(active_tools=['pan']),
)


class PostProcess :
    def __init__(self, attrs):
        "Reconstruit une grille à partir des attributs du fichier NC"
        list_params = ['x_bounds','y_bounds', 'Nx', 'Ny', 'geometry']
        dealias_params = dealiasing.DealiasParams(apply_dealias = False)
        #list_dealias_params = [f.name for f in dataclasses.fields(dealias_params)]

        if not set(list_params).issubset(list(attrs.keys())) :
            raise KeyError(f"all values of {list_params} must be presents in attrs")

        list_params.append('dealias_params')
        params_trunc = collections.namedtuple('params', list_params)

        #Inutile d'encapsuler, un namedtuple est déjà immutable
        self.params = params_trunc(x_bounds = tuple(attrs['x_bounds']),
                                   y_bounds = tuple(attrs['y_bounds']),
                                   Nx = attrs['Nx'],
                                   Ny = attrs['Ny'],
                                   geometry = attrs['geometry'],
                                   dealias_params = dealias_params)

        self.Grid = grid.Grid(self.params)
        self.__geometry = self.params.geometry
        #fonctions pour accéder aux coefficients
        if self.__geometry == 'biperiodic' :
            func_transform_x = lambda xx : scipy.fft.fftshift(scipy.fft.fft(xx, axis = 0), axes = 0)
            func_transform_y = lambda yy : scipy.fft.fftshift(scipy.fft.fft(yy, axis = 1), axes = 1)

            coord_transform_x = scipy.fft.fftshift(2.0*np.pi*scipy.fft.fftfreq(self.params.Nx, d = self.Grid.dx), axes = 0)
            coord_transform_y = scipy.fft.fftshift(2.0*np.pi*scipy.fft.fftfreq(self.params.Ny, d = self.Grid.dy), axes = 1)
            name_coord_x = 'kx'
            name_coord_y = 'ky'

        elif self.__geometry == 'zonal_channel' :
            func_transform_x = lambda xx : scipy.fft.fftshift(scipy.fft.fft(xx, axis = 0), axes = 0)
            func_transform_y = lambda yy : scipy.fft.dct(yy, type = 1, axis = 1)

            coord_transform_x = scipy.fft.fftshift(2.0*np.pi*scipy.fft.fftfreq(self.params.Nx, d = self.Grid.dx), axes = 0)
            coord_transform_y = np.arange(0, self.params.Ny)
            name_coord_x = 'kx'
            name_coord_y = 'k'

        elif self.__geometry == 'basin' :
            raise NotImplementedError
        else :
            raise ValueError
        funcs_transform = collections.namedtuple('func_transform',
                                                 ['x',
                                                  'y',
                                                 'coord_x',
                                                 'coord_y',
                                                 'name_coord_x',
                                                 'name_coord_y'])

        self.funcs_transform = funcs_transform(x = func_transform_x,
                                               y = func_transform_y,
                                               coord_x = coord_transform_x,
                                               coord_y = coord_transform_y,
                                               name_coord_x = name_coord_x,
                                               name_coord_y = name_coord_y)

    ########## Fonctions de la classe Grid (mais sans BCs) ############################################
    def derivative(self, var, axis, order = 1):
        func_derivative = lambda val : self.Grid.derivative(val, axis, order = order, BC_Dirichlet = False)
        if type(var) == np.ndarray:
            return func_derivative(var)
        elif type(var) == xr.DataArray:
            return xr.apply_ufunc(func_derivative,
                                  var,
                                  input_core_dims=[['x', 'y']],
                                  output_core_dims = [['x', 'y']],
                                  vectorize = True
                                  )
        else :
            raise TypeError

    def laplacien(self, var):
        func_laplacien = lambda val: self.Grid.laplacien(var, BC = False)
        if type(var) == np.ndarray:
            return func_laplacien(var)
        elif type(var) == xr.DataArray:
            return xr.apply_ufunc(func_laplacien,
                                  var,
                                  input_core_dims=[['x', 'y']],
                                  output_core_dims=[['x', 'y']],
                                  vectorize=True
                                  )
        else :
            raise TypeError
    def jacobien(self, A, B):
        func_jacobien = lambda AA, BB : self.Grid.jacobien(AA, BB, BC_A = False, BC_B = False)
        if (type(A) == np.ndarray) and (type(B) == np.ndarray):
            return func_jacobien(A, B)
        elif (type(A) == xr.DataArray) and (type(B) == xr.DataArray):
            return xr.apply_ufunc(func_jacobien,
                                  A, B,
                                  input_core_dims=[['x', 'y']],
                                  output_core_dims=[['x', 'y']],
                                  vectorize=True
                                  )
        else :
            raise TypeError
    def integrate(self, var, axis = 'all'):
        "Attention : ne fonctionne pas"
        if type(var) == np.ndarray :
            return self.Grid.integrate(var, axis)
        elif type(var) == xr.DataArray :
            func_integrate = lambda XX : self.Grid.integrate(var, axis)
            if axis == 'x':
                output_core_dims = [['y']]
            elif axis == 'y':
                output_core_dims = [['x']]
            elif axis == 'all':
                output_core_dims = [[]]
            else :
                raise ValueError
            return xr.apply_ufunc(func_integrate,
                                  var,
                                  input_core_dims=[['x', 'y']],
                                  output_core_dims=output_core_dims,
                                  vectorize=True
                                  )
        else:
            raise TypeError

    def int_cum(self, var, axis):
        "Attention : non testé"
        if type(var) == np.ndarray:
            return self.Grid.int_cum(var, axis)
        elif type(var) == xr.DataArray :
            func_integrate = lambda XX : self.Grid.integrate(var, axis)
            return xr.apply_ufunc(func_integrate,
                                  var,
                                  input_core_dims=[['x', 'y']],
                                  output_core_dims=[['x', 'y']],
                                  vectorize=True
                                  )

    ########### Calculs et tracés de spectres ##########################################################
    def var_spectrum_2D(self, var : np.ndarray | xr.DataArray, axis):
        """Calcule le spectre de variance sur une dimension donnée :
        Pour configuration bipériodique :
        -|FFT_x(var)|^2(kx, y) si axis = 'x'
        -|FFT_y(var)|^2(x, ky) si axis = 'y'
        Pour configuration canal :
        -|c_i|^2(x, n) si axis = 'x

        """

        if axis == "x":
            transform_to_apply = self.funcs_transform.x
        elif axis == 'y':
            transform_to_apply = self.funcs_transform.y
        else:
            raise ValueError(f"axis must be in ('x', 'y'), value : {axis}")

        if type(var) == xr.DataArray and 't' in var.dims:
            return self.__compute_spectra_Da(var, axis)
        elif type(var) == np.ndarray and var.shape == self.Grid.shape :
            return np.abs(transform_to_apply(var))**2
        else :
            raise TypeError

    def plot_spectra(self, Da, ax = None):
        "Trace un ou plusieurs spectre à des instants différents"
        pass


    def to_regular_grid(self, Ds : xr.Dataset|xr.DataArray, interp = 'linear'):
        if self.params.geometry == 'biperiodic' :
            raise NotImplementedError("in biperiodic configuration, grid is already periodic !!")
        elif self.params.geometry == 'zonal_channel' :
            raise NotImplementedError("TODO ...")
        else :
            raise NotImplementedError("basin configuration not (yet) implemented")


    def __compute_spectra_Da(self, Da_var, axis):
        if axis == "x":
            transform_to_apply = self.funcs_transform.x
        elif axis == 'y':
            transform_to_apply = self.funcs_transform.y
        else :
            raise ValueError
        name_coord_x = self.funcs_transform.name_coord_x if axis == 'x' else 'x'
        name_coord_y = self.funcs_transform.name_coord_y if axis == 'y' else 'y'

        coord_x = self.funcs_transform.coord_x if axis == 'x' else self.Grid.x
        coord_y = self.funcs_transform.coord_y if axis == 'y' else self.Grid.y

        Da = xr.DataArray(data=np.zeros_like(Da_var.values),
                          dims=['t', name_coord_x, name_coord_y],
                          coords={'t': Da_var.coords['t'].values,
                                  name_coord_x: coord_x,
                                  name_coord_y: coord_y}
                          )
        for i in range(0, Da_var.coords['t'].values.size):
            var_i = Da_var.isel(t=i).values
            Da[{'t': i}] = np.abs(transform_to_apply(var_i)) ** 2

        # On ne sélectionne que les kx/y positifs
        name_coord = name_coord_x if axis == 'x' else name_coord_y
        if self.params.geometry == 'biperiodic':
            Da = Da.sel({name_coord : slice(0,None)})
        if self.params.geometry == 'zonal_channel' and axis == 'x':
            Da = Da.sel({name_coord : slice(0, None)})
        Da.attrs = {'geometry' : self.params.geometry,
                    'axis' : axis}
        return Da




def animate(Da:xr.DataArray,
            x = 'x',
            y = 'y',
            t = 't',
            width=500,
            height=500,
            clim=None,
            title=None,
            cmap='seismic',
            logx=False,
            logy=False,
            regular_grid = True,
            log_cnorm = False,
            fps = 5):
    "Genere une animation. Uniquement pour Jupyter Notebook. Tiré et amélioré de mon module (perso) f2dxarray"
    if clim == None:
        clim = (float(Da.min()), float(Da.max()))

    if log_cnorm :
        symlog_norm = 'log'
    else :
        symlog_norm = 'linear'
    if regular_grid :
        anim = Da.hvplot(title = title,
                         x = x,
                         y = y,
                         framewise = False,
                         width = width,
                         height = height,
                         widget_type = "scrubber",
                         widget_location = "bottom",
                         clim = clim,
                         cmap = cmap,
                         cnorm = symlog_norm,
                         logx = logx,
                         logy = logy,
                         dynamic = True)
    else :
        anim = Da.hvplot.quadmesh(title = title,
                                  x = x,
                                  y = y,
                                  framewise = False,
                                  width = width,
                                  height = height,
                                  widget_type = "scrubber",
                                  widget_location = "bottom",
                                  clim = clim,
                                  cmap = cmap,
                                  logx = logx,
                                  logy = logy,
                                  cnorm=symlog_norm,
                                  dynamic = True)

     #anim = anim.options(hooks=[disable_wheel_zoom])

    panel_obj = pn.panel(anim, widgets = {t : pn.widgets.Player}, format = 'png')

    interval = 1000.0/fps#conversion en ms

    player_widget = panel_obj[1][0]
    player_widget.interval = int(interval)

    #On vire quelque boutons inutiles
    player_widget.visible_buttons = ['previous', 'play', 'pause', 'next']
    player_widget.show_loop_controls = False

    return panel_obj
