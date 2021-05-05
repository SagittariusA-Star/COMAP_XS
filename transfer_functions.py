#the module that gather all three transfer functions together - pipeline, angular resolution, spectral resolution

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ioff() #turn of the interactive plotting
import matplotlib as matplotlib
import numpy.fft as fft
import corner
import h5py
import sys
import scipy.interpolate
import itertools as itr
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable


#Instrumental beam transfer function (from limited angular resolution)
N_beam = 10 #number of signal map realizations used
beam_ps_smooth_1D = np.load('ps_smooth_1D_newest.npy')
beam_ps_original_1D = np.load('ps_original_1D_newest.npy')
beam_k_1D = np.load('k_arr_1D_newest.npy')

beam_ps_smooth_2D = np.load('ps_smooth_newest.npy')
beam_ps_original_2D = np.load('ps_original_newest.npy')
beam_k_2D = np.load('k_arr_newest.npy')

beam_TF_1D = np.zeros_like(beam_ps_smooth_1D)
beam_TF_2D = np.zeros_like(beam_ps_smooth_2D)

for i in range(N_beam):
   beam_TF_1D[i] = beam_ps_smooth_1D[i]/beam_ps_original_1D[i]
   beam_TF_2D[i] = beam_ps_smooth_2D[i]/beam_ps_original_2D[i]
beam_TF_1D = np.mean(beam_TF_1D, axis=0)
beam_TF_2D = np.mean(beam_TF_2D, axis=0)

beam_TF_1D_func = scipy.interpolate.interp1d(beam_k_1D, beam_TF_1D)
beam_k_perp = beam_k_2D[0]
beam_k_par = beam_k_2D[1]
beam_TF_2D_func = scipy.interpolate.interp2d(beam_k_perp, beam_k_par, beam_TF_2D)


#Pixel window transfer function (from limited spectral resolution)
N_freq = 10 #number of signal map realizations used
freq_ps_low_1D = np.load('ps_low_res_1D_v4.npy')
freq_ps_high_1D = np.load('ps_high_res_1D_v4.npy')
freq_k_1D = np.load('k_arr_1D_v4.npy')

freq_ps_low_2D = np.load('ps_low_res_v4.npy')
freq_ps_high_2D = np.load('ps_high_res_v4.npy')
freq_k_2D = np.load('k_arr_v4.npy')

freq_TF_1D = np.zeros_like(freq_ps_low_1D)
freq_TF_2D = np.zeros_like(freq_ps_low_2D)

for i in range(N_freq):
   freq_TF_1D[i] = freq_ps_low_1D[i]/freq_ps_high_1D[i]
   freq_TF_2D[i] = freq_ps_low_2D[i]/freq_ps_high_2D[i]
freq_TF_1D = np.mean(freq_TF_1D, axis=0)
freq_TF_2D = np.mean(freq_TF_2D, axis=0)

freq_TF_1D_func = scipy.interpolate.interp1d(freq_k_1D, freq_TF_1D)
freq_k_perp = freq_k_2D[0]
freq_k_par = freq_k_2D[1]
freq_TF_2D_func = scipy.interpolate.interp2d(freq_k_perp, freq_k_par, freq_TF_2D)


#Pipeline transfer function (from various procedures in low level data processing)

#Read the h5 files
def read_pipeline_TF(filename, dim):
   if dim == 1:
      with h5py.File(filename, mode="r") as my_file:
         k = np.array(my_file['k'][:]) 
         TF_1D = np.array(my_file['TF'][:]) 
      return k, TF_1D
   if dim == 2:
      with h5py.File(filename, mode="r") as my_file:
         k_perp = np.array(my_file['k'][0]) 
         k_par = np.array(my_file['k'][1]) 
         TF_2D = np.array(my_file['TF'][:]) 
      return k_perp, k_par, TF_2D

#mix of Liss and CES scans
mix_k_1D, mix_TF_1D = read_pipeline_TF('TF_1d.h5', 1)
mix_TF_1D_func = scipy.interpolate.interp1d(mix_k_1D, mix_TF_1D) 

mix_k_perp, mix_k_par, mix_TF_2D = read_pipeline_TF('TF_2d.h5', 2)
mix_TF_2D_func = scipy.interpolate.interp2d(mix_k_perp, mix_k_par, mix_TF_2D)

#Liss scans only
liss_k_1D, liss_TF_1D = read_pipeline_TF('1D_TF_Liss.h5', 1)
liss_TF_1D_func = scipy.interpolate.interp1d(liss_k_1D, liss_TF_1D) 

liss_k_perp, liss_k_par, liss_TF_2D = read_pipeline_TF('2D_TF_Liss.h5', 2)
liss_TF_2D_func = scipy.interpolate.interp2d(liss_k_perp, liss_k_par, liss_TF_2D)

#CES scans only
CES_k_1D, CES_TF_1D = read_pipeline_TF('1D_TF_CES.h5', 1)
CES_TF_1D_func = scipy.interpolate.interp1d(CES_k_1D, CES_TF_1D) 

CES_k_perp, CES_k_par, CES_TF_2D = read_pipeline_TF('2D_TF_CES.h5', 2)
CES_TF_2D_func = scipy.interpolate.interp2d(CES_k_perp, CES_k_par, CES_TF_2D)


#gather all functions together - mix of Liss and CES scans
def TF_beam_freq_mix_1D(k):
   return beam_TF_1D_func(k)*freq_TF_1D_func(k)*mix_TF_1D_func(k)

def TF_beam_freq_mix_2D(k_perp, k_par):
   return beam_TF_2D_func(k_perp, k_par)*freq_TF_2D_func(k_perp, k_par)*mix_TF_2D_func(k_perp, k_par)

#gather all functions together - Liss scans only
def TF_beam_freq_liss_1D(k):
   return beam_TF_1D_func(k)*freq_TF_1D_func(k)*liss_TF_1D_func(k)

def TF_beam_freq_liss_2D(k_perp, k_par):
   return beam_TF_2D_func(k_perp, k_par)*freq_TF_2D_func(k_perp, k_par)*liss_TF_2D_func(k_perp, k_par)

#gather all functions together - CES scans only
def TF_beam_freq_CES_1D(k):
   return beam_TF_1D_func(k)*freq_TF_1D_func(k)*CES_TF_1D_func(k)

def TF_beam_freq_CES_2D(k_perp, k_par):
   return beam_TF_2D_func(k_perp, k_par)*freq_TF_2D_func(k_perp, k_par)*CES_TF_2D_func(k_perp, k_par)


