"""
Quelque fonctions dédiées au post-traitement sous Jupyter Notebook. La plupart du temps, xarray fait très bien le
travail tout seul !

"""



import xarray as xr
import holoviews


def animate(Da:xr.DataArray, width=500, height=500, clim=None, title=None, cmap='seismic', logx=False, logy=False):
    "Genere une animation. Uniquement pour Jupyter Notebook. Tiré de mon module (perso) f2dxarray"
    if clim == None:
        clim = (float(Da.min()), float(Da.max()))

    anim = Da.hvplot(groupby='t',
                     title=title,
                     framewise=False,
                     width=width,
                     height=height,
                     widget_type="scrubber",
                     widget_location="bottom",
                     clim=clim,
                     cmap=cmap,
                     logx=logx,
                     logy=logy)
    return anim