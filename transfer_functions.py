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
beam_k_1D = np.load('k_arr_1D_newest.npy')[0]

beam_ps_smooth_2D = np.load('ps_smooth_newest.npy')
beam_ps_original_2D = np.load('ps_original_newest.npy')
beam_k_2D = np.load('k_arr_newest.npy')[0]

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
freq_k_1D = np.load('k_arr_1D_v4.npy')[0]

freq_ps_low_2D = np.load('ps_low_res_v4.npy')
freq_ps_high_2D = np.load('ps_high_res_v4.npy')
freq_k_2D = np.load('k_arr_v4.npy')[0]

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


#Plotting
k_bin_edges = np.logspace(-2.0, np.log10(1.5), 15) 
k_bin_edges_par = np.logspace(-2.0, np.log10(1.0), 15)
k_bin_edges_perp = np.logspace(-2.0 + np.log10(2), np.log10(1.5), 15)


def plot_TF_1D(k, TF_1D, type_of_TF):

   plt.plot(k, TF_1D(k), color='indianred')
   plt.xscale('log')
   plt.ylabel(type_of_TF, fontsize=14)
   plt.xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   plt.xlim([k.min(),k.max()])
   if type_of_TF == r'$\mathrm{T^{beam}(k)}$':
      plt.ylim([0,2])
   else:
      plt.ylim([0,1.05])
   plt.xticks(fontsize=12)
   plt.yticks(fontsize=12)
   plt.tight_layout()
   plt.show()

def log2lin(x, k_edges):
    loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
    logx = np.log10(x) - np.log10(k_edges[0])
    return logx / loglen

def plot_TF_2D(k_perp, k_par, TF_2D, type_of_TF):
    
      fig, ax = plt.subplots(1,1,figsize=(5.6,5.6))
      fig.tight_layout()
     
      norm = mpl.colors.Normalize(vmin=0, vmax=1.5) 
      img1 = ax.imshow(TF_2D(k_perp,k_par), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='RdBu',norm=norm)
      fig.colorbar(img1, ax=ax,fraction=0.046, pad=0.04)

      ticks = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,0.1,
              0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1., 1.1, 1.2, 1.3]

      majorticks = [ 0.03,0.1, 0.3,1]
      majorlabels = [ '0.03','0.1', '0.3','1']

      xbins = k_bin_edges_par

      ticklist_x = log2lin(ticks[:-3], xbins)
      majorlist_x = log2lin(majorticks, xbins)

      ybins = k_bin_edges_perp

      ticklist_y = log2lin(ticks, ybins)
      majorlist_y = log2lin(majorticks, ybins)

      ax.set_title(type_of_TF, fontsize=16)
      ax.set_xticks(ticklist_x, minor=True)
      ax.set_xticks(majorlist_x, minor=False)
      ax.set_xticklabels(majorlabels, minor=False, fontsize=16)
      ax.set_yticks(ticklist_y, minor=True)
      ax.set_yticks(majorlist_y, minor=False)
      ax.set_yticklabels(majorlabels, minor=False, fontsize=16)
      ax.set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]',fontsize=16)
      ax.set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}$]',fontsize=16)
    
      plt.tight_layout()
      plt.show()

plot_TF_1D(beam_k_1D, beam_TF_1D_func, r'$\mathrm{T^{beam}(k)}$')
plot_TF_1D(freq_k_1D, freq_TF_1D_func, r'$\mathrm{T^{freq}(k)}$')
plot_TF_1D(mix_k_1D, mix_TF_1D_func, r'$\mathrm{T^{mix}(k)}$')
plot_TF_1D(liss_k_1D, liss_TF_1D_func, r'$\mathrm{T^{Liss}(k)}$')
plot_TF_1D(CES_k_1D, CES_TF_1D_func, r'$\mathrm{T^{CES}(k)}$')
plot_TF_1D(CES_k_1D, TF_beam_freq_mix_1D, r'$\mathrm{T^{total}(k)}$')

plot_TF_2D(beam_k_perp, beam_k_par, beam_TF_2D_func, r'$ T^{beam}(k_{\bot},k_{\parallel})$')
plot_TF_2D(freq_k_perp, freq_k_par, freq_TF_2D_func, r'$ T^{freq}(k_{\bot},k_{\parallel})$')
plot_TF_2D(mix_k_perp, mix_k_par, mix_TF_2D_func, r'$ T^{mix}(k_{\bot},k_{\parallel})$')
plot_TF_2D(liss_k_perp, liss_k_par, liss_TF_2D_func, r'$ T^{Liss}(k_{\bot},k_{\parallel})$')
plot_TF_2D(CES_k_perp, CES_k_par, CES_TF_2D_func, r'$ T^{CES}(k_{\bot},k_{\parallel})$')
plot_TF_2D(CES_k_perp, CES_k_par, TF_beam_freq_mix_2D, r'$ T^{total}(k_{\bot},k_{\parallel})$')





