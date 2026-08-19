import numpy as np
import collections

from pathlib import Path
import sys

ROOT = Path.cwd().parent      # pyTQG
SRC = ROOT / "pyTQG_solver"

print(ROOT)
print(SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import TimeScheme



def contour_stab(stab_func, N_x = 2000, N_y=2000, real_interval = (-4.0, 4.0), im_interval = (-4.0, 4.0)):
    "retourne les coordonnées (dans plan complexe) du contour de la zone de stabilité"
    if len(real_interval) != 2:
        raise ValueError("real_interval must be a 2 items list")
    if len(im_interval) != 2:
        raise ValueError("im_interval must be a 2 items list")

    if real_interval[0] >= real_interval[1]:
        raise ValueError(f"real_interval values must be passed in ascending order")
    if im_interval[0] >= im_interval[1]:
        raise ValueError(f"im_interval values must be passed in ascending order")


    Re = np.linspace(real_interval[0], im_interval[1], N_x)
    Im = np.linspace(im_interval[0], im_interval[1], N_y)
    RR, II = np.meshgrid(Re, Im, indexing = 'ij')
    Z=RR+1.0j*II
    Z_mask = np.where(stab_func(Z), True, False)
    cpx_contour = []
    for i in range(0, Z_mask.shape[0]):
        slice_y_i = Z_mask[i, :]
        for j in range(0, Z_mask.shape[1]-1):
            if np.logical_xor(slice_y_i[j], slice_y_i[j+1]):
                cpx_contour.append(Z[i, j])
    
    cpx_contour = np.array(cpx_contour)
    contour_angle = np.angle(cpx_contour)
    contour_abs = np.abs(cpx_contour)
    #renvoie un namedtuple (dictionnaire immutable)
    StabContour = collections.namedtuple('StabContour', ['Complex', 'angle', 'abs'])
    
    idx_angles_sorted = np.argsort(contour_angle)
    StabContour.Complex = cpx_contour[idx_angles_sorted]
    StabContour.angle = contour_angle[idx_angles_sorted]
    StabContour.abs = contour_abs[idx_angles_sorted]
    return StabContour

def corr_VP(liste_eigs, contour_stab_res, coeff_sec = 0.9, ecarts_angles_max = 0.1, get_coeff = False):
    "contour_stab_res : calculé avec fonction contour_stab, trié par arguments croissants"
    angle_eigs = np.angle(liste_eigs)
    dist_eigs = np.abs(liste_eigs)

    liste_k = np.zeros(len(liste_eigs))
    N_val_contour = contour_stab_res.angle.size
    mask_VP = np.full_like(liste_eigs, False)#False = valide, True = invalide
    for i in range(0, len(liste_eigs)):
        angle_i = angle_eigs[i]
        dist_i = dist_eigs[i]
        #print("#######################################")
        #print(f"angle_i = {angle_i}")
        
        idx_angle_sup = np.searchsorted(contour_stab_res.angle, angle_i)
        #print(f"idx_angle_sup = {idx_angle_sup}")
        real_axis = False
        if idx_angle_sup == 0:
            #angle proche de -pi
            #print(f"angle proche de -pi")
            angle_inf = contour_stab_res.angle[0]
            angle_sup = contour_stab_res.angle[-1]
            idx_inf = 0
            idx_sup = -1
            ecarts_angles = np.abs( (angle_sup - angle_inf)%(np.pi) )
            real_axis = True
        elif idx_angle_sup >= len(liste_eigs):
            #print("angle proche de pi")
            angle_inf = contour_stab_res.angle[0]
            angle_sup = contour_stab_res.angle[-1]
            idx_inf = 0
            idx_sup = -1
            ecarts_angles = np.abs( (angle_sup - angle_inf) % (np.pi))
            real_axis = True
        else :
            idx_inf = idx_angle_sup-1
            idx_sup = idx_angle_sup
            angle_inf = contour_stab_res.angle[idx_inf]
            angle_sup = contour_stab_res.angle[idx_sup]
            ecarts_angles = np.abs(angle_sup - angle_inf)
        #Cas où la VP n'est pas dans la zone de stabilité
        if (ecarts_angles > ecarts_angles_max) and (not real_axis):
            liste_k[i] = np.NaN
            mask_VP[i] = True
            continue#On passe à l'itération suivante
        #print(f"angle_sup = {angle_sup}")
        #print(f"angle_inf = {angle_inf}")
        if real_axis :
            #print(f"real_axis")
            #on se ramène dans un intervalle autour de 0
            if liste_eigs[i].real < 0.0:
                #print(f"réel négatif")
                angle_sup = angle_sup - angle_i
                angle_inf = angle_inf + angle_i
                angle_i = 0.0
            else:
                #print("réel positif")
                angle_i = 0.0
                angle_sup = angle_sup-np.pi
                angle_inf = angle_inf+np.pi
            #print(f"angle_i corrigé = {angle_i}")
            #print(f"angle_sup corrigé = {angle_sup}")
            #print(f"angle_ing corrigé = {angle_inf}")
        abs_sup = contour_stab_res.abs[idx_sup]
        abs_inf = contour_stab_res.abs[idx_inf]
        #print(f"dist_i = {dist_i}")
        #print(f"abs_sup = {abs_sup}")
        #print(f"abs_inf = {abs_inf}")
        abs_interp = np.interp(angle_i, [angle_inf, angle_sup], [abs_inf, abs_sup])
        coeff_homotethie = abs_interp/dist_i

        if coeff_homotethie >= 1.0:
            coeff_homotethie = np.NaN
        liste_k[i] = coeff_homotethie
        #print("#######################################")
    
    k_min = np.nanmin(liste_k)
    coeff_tot = coeff_sec*k_min
    ma_VP_corr = np.ma.array(coeff_tot*liste_eigs, mask = mask_VP, fill_value = 100.0)
    if get_coeff:
        return coeff_tot, ma_VP_corr
    return ma_VP_corr


def d_min_contour(eigs_corr, contour_stab_res):
    "eigs_corr : masked array (True : dans la zone de stabilité, False: en dehors)"
    angle_VP_corr = np.angle(eigs_corr)
    angle_min = angle_VP_corr.min()
    angle_max = angle_VP_corr.max()
    if angle_min <= angle_max:
        masque_cone = (contour_stab_res.angle >= angle_min) & (contour_stab_res.angle <= angle_max)
    else:
        masque_cone = (contour_stab_res.angle >= angle_min) | (contour_stab_res.angle <= angle_max)
    abs_ctr_cone = contour_stab_res.abs[masque_cone]
    if len(abs_ctr_cone) >0:
        d_min_ctr = np.min(abs_ctr_cone)
    else:
        d_min_ctr = max( np.min(contour_stab_res.abs), 1.0e-6)#gère le cas où TOUTES les VP ne peuvent pas etre dans la zone de stabilité 
    abs_VP_max = np.abs(eigs_corr).max()
    return d_min_ctr, abs_VP_max


class StabilityAnalysis:
    def __init__(self, TimeScheme):
        self.TimeScheme = TimeScheme
        self.func_stab = self.TimeScheme.func_stab
        self.contour_stab_res = contour_stab(self.func_stab)

    def Compute_stab_operators(self, liste_eigs, coeff_sec = 0.9, ecarts_angles_max = 0.1):
        if (not isinstance(liste_eigs, list)) or (not isinstance(liste_eigs, np.ndarray)) or (not isinstance(liste_eigs, tuple)):
            liste_eigs = (liste_eigs)
        ResStab = collections.namedtuple('ResStab', ['eigs', 'eigs_corr','max_abs_eigs','max_abs_eigs_corr', 'd_min_stab', 'coeff_corr'])
        liste_res = [None for i in range(0, len(liste_eigs))]
        for i in range(0, len(liste_eigs)):
            coeff_corr_i, VP_corr_i = corr_VP(liste_eigs[i], self.contour_stab_res, coeff_sec=coeff_sec, ecarts_angles_max =ecarts_angles_max, get_coeff = True)
            d_min_ctr_i, max_abs_VP_i = d_min_contour(VP_corr_i, self.contour_stab_res)
            ResStab_i = ResStab(liste_eigs[i], VP_corr_i, np.abs(liste_eigs[i]).max(), max_abs_VP_i, d_min_ctr_i, coeff_corr_i)
            liste_res[i] = ResStab_i
        if len(liste_res)==1:
            return liste_res[0]
        return tuple(liste_res)