#the module that gather all three transfer functions together - pipeline, angular resolution, spectral resolution

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#plt.ioff() #turn of the interactive plotting
import matplotlib as matplotlib
import numpy.fft as fft
import corner
import matplotlib.colors as colors
import h5py
import sys
import scipy.interpolate
import itertools as itr
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

location = 'transfer_functions/' #the directory where all the TF files are stored

#Instrumental beam transfer function (from limited angular resolution)
N_beam = 100 #number of signal map realizations used
beam_ps_smooth_1D = np.load(location + 'ps_smooth_1D_newest.npy')
beam_ps_original_1D = np.load(location + 'ps_original_1D_newest.npy')
beam_k_1D = np.load(location + 'k_arr_1D_newest.npy')[0]

beam_ps_smooth_2D = np.load(location + 'ps_smooth_newest.npy')
beam_ps_original_2D = np.load(location + 'ps_original_newest.npy')
beam_k_2D = np.load(location + 'k_arr_newest.npy')[0]

beam_TF_1D = np.mean(beam_ps_smooth_1D, axis=0)/np.mean(beam_ps_original_1D, axis=0)
beam_TF_2D = np.mean(beam_ps_smooth_2D, axis=0)/np.mean(beam_ps_original_2D, axis=0)

beam_TF_1D_func = scipy.interpolate.interp1d(beam_k_1D, beam_TF_1D)
beam_k_perp = beam_k_2D[0]
beam_k_par = beam_k_2D[1]
beam_TF_2D_func = scipy.interpolate.interp2d(beam_k_perp, beam_k_par, beam_TF_2D)


#Pixel window transfer function (from limited spectral resolution)
N_freq = 100 #number of signal map realizations used
freq_ps_low_1D = np.load(location + 'ps_low_res_1D_v4.npy')
freq_ps_high_1D = np.load(location + 'ps_high_res_1D_v4.npy')
freq_k_1D = np.load(location + 'k_arr_1D_v4.npy')[0]

freq_ps_low_2D = np.load(location + 'ps_low_res_v4.npy')
freq_ps_high_2D = np.load(location + 'ps_high_res_v4.npy')
freq_k_2D = np.load(location + 'k_arr_v4.npy')[0]

freq_TF_1D = np.mean(freq_ps_low_1D, axis=0)/np.mean(freq_ps_high_1D, axis=0)
freq_TF_2D = np.mean(freq_ps_low_2D, axis=0)/np.mean(freq_ps_high_2D, axis=0)

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
mix_k_1D, mix_TF_1D = read_pipeline_TF(location + 'TF_1d.h5', 1)
mix_TF_1D_func = scipy.interpolate.interp1d(mix_k_1D, mix_TF_1D) 

mix_k_perp, mix_k_par, mix_TF_2D = read_pipeline_TF(location + 'TF_2d.h5', 2)
mix_TF_2D_func = scipy.interpolate.interp2d(mix_k_perp, mix_k_par, mix_TF_2D)

#Liss scans only
liss_k_1D, liss_TF_1D = read_pipeline_TF(location + '1D_TF_Liss.h5', 1)
liss_TF_1D_func = scipy.interpolate.interp1d(liss_k_1D, liss_TF_1D) 

liss_k_perp, liss_k_par, liss_TF_2D = read_pipeline_TF(location + '2D_TF_Liss.h5', 2)
liss_TF_2D_func = scipy.interpolate.interp2d(liss_k_perp, liss_k_par, liss_TF_2D)

#CES scans only
CES_k_1D, CES_TF_1D = read_pipeline_TF(location + '1D_TF_CES.h5', 1)
CES_TF_1D_func = scipy.interpolate.interp1d(CES_k_1D, CES_TF_1D) 

CES_k_perp, CES_k_par, CES_TF_2D = read_pipeline_TF(location + '2D_TF_CES.h5', 2)
CES_TF_2D_func = scipy.interpolate.interp2d(CES_k_perp, CES_k_par, CES_TF_2D)


#gather all functions together - mix of Liss and CES scans
def TF_beam_freq_mix_1D(k):
   return beam_TF_1D_func(k)*freq_TF_1D_func(k)*mix_TF_1D_func(k)




#print 'k', 'beam', 'freq', 'pipeline', 'total'
#for i in range(len(CES_k_1D)):
 #  print CES_k_1D[i], beam_TF_1D_func(CES_k_1D)[i], freq_TF_1D_func(CES_k_1D)[i], mix_TF_1D_func(CES_k_1D)[i], TF_beam_freq_mix_1D(CES_k_1D)[i]

def TF_beam_freq_mix_2D(k_perp, k_par):
   return beam_TF_2D_func(k_perp, k_par)*freq_TF_2D_func(k_perp, k_par)*mix_TF_2D_func(k_perp, k_par)

def TF_beam_mix_2D(k_perp, k_par):
   return beam_TF_2D_func(k_perp, k_par)*mix_TF_2D_func(k_perp, k_par)

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


def plot_TF_1D(k, TF_1D, type_of_TF, name_to_save):

   plt.plot(k, TF_1D(k), color='indianred')
   #plt.plot(k, TF_1D, color='indianred')
   plt.xscale('log')
   #plt.yscale('log')
   plt.ylabel(type_of_TF, fontsize=14)
   plt.xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   plt.xlim([k.min(),k.max()])
   plt.ylim([0,1.2])
   plt.xticks(fontsize=12)
   plt.yticks(fontsize=12)
   plt.tight_layout()
   plt.savefig(name_to_save)
   plt.show()

def log2lin(x, k_edges):
    loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
    logx = np.log10(x) - np.log10(k_edges[0])
    return logx / loglen

def plot_TF_2D(k_perp, k_par, TF_2D, type_of_TF, name_to_save):
    
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
      plt.savefig(name_to_save)
      plt.show()

'''
plot_TF_1D(beam_k_1D, beam_TF_1D_func, r'$\mathrm{T^{beam}(k)}$', 'beam_TF_1D.png')
plot_TF_1D(freq_k_1D, freq_TF_1D_func, r'$\mathrm{T^{freq}(k)}$', 'freq_TF_1D.png')
plot_TF_1D(mix_k_1D, mix_TF_1D_func, r'$\mathrm{T^{mix}(k)}$', 'mix_TF_1D.png')
plot_TF_1D(liss_k_1D, liss_TF_1D_func, r'$\mathrm{T^{Liss}(k)}$','liss_TF_1D.png')
plot_TF_1D(CES_k_1D, CES_TF_1D_func, r'$\mathrm{T^{CES}(k)}$', 'CES_TF_1D.png')
plot_TF_1D(CES_k_1D, TF_beam_freq_mix_1D, r'$\mathrm{T^{total, mix}(k)}$', 'total_mix_TF_1D.png')
plot_TF_1D(CES_k_1D, TF_beam_freq_liss_1D, r'$\mathrm{T^{total, Liss}(k)}$', 'total_liss_TF_1D.png')
plot_TF_1D(CES_k_1D, TF_beam_freq_CES_1D, r'$\mathrm{T^{total, CES}(k)}$', 'total_CES_TF_1D.png')

plot_TF_2D(beam_k_perp, beam_k_par, beam_TF_2D_func, r'$ T^{beam}(k_{\bot},k_{\parallel})$','beam_TF_2D.png') 
plot_TF_2D(freq_k_perp, freq_k_par, freq_TF_2D_func, r'$ T^{freq}(k_{\bot},k_{\parallel})$', 'freq_TF_2D.png')
plot_TF_2D(mix_k_perp, mix_k_par, mix_TF_2D_func, r'$ T^{mix}(k_{\bot},k_{\parallel})$', 'mix_TF_2D.png')
plot_TF_2D(liss_k_perp, liss_k_par, liss_TF_2D_func, r'$ T^{Liss}(k_{\bot},k_{\parallel})$', 'liss_TF_2D.png')
plot_TF_2D(CES_k_perp, CES_k_par, CES_TF_2D_func, r'$ T^{CES}(k_{\bot},k_{\parallel})$', 'CES_TF_2D.png')
plot_TF_2D(CES_k_perp, CES_k_par, TF_beam_freq_mix_2D, r'$ T^{total, mix}(k_{\bot},k_{\parallel})$', 'total_mix_TF_2D.png')
plot_TF_2D(CES_k_perp, CES_k_par, TF_beam_freq_liss_2D, r'$ T^{total, Liss}(k_{\bot},k_{\parallel})$', 'total_liss_TF_2D.png')
plot_TF_2D(CES_k_perp, CES_k_par, TF_beam_freq_CES_2D, r'$ T^{total, CES}(k_{\bot},k_{\parallel})$', 'total_CES_TF_2D.png')
'''

def plot_ps(ps_2d, titlename):
   fig, ax = plt.subplots(1,1)
   norm = mpl.colors.Normalize(vmin=7, vmax=16.5) 
   img = ax.imshow(np.log10(ps_2d), interpolation='none', origin='lower', extent=[0,1,0,1], norm=norm)
   #plt.imshow(np.log10(nmodes), interpolation='none', origin='lower')
   cbar = fig.colorbar(img)
  
   cbar.set_label(r'$\log_{10}(\tilde{P}_{\parallel, \bot}(k))$ [$\mu$K${}^2$ (Mpc)${}^3$]')
  
   def log2lin(x, k_edges):
       loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
       logx = np.log10(x) - np.log10(k_edges[0])
       return logx / loglen
   

   # ax.set_xscale('log')
   minorticks = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
              0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
              0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
              0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
              2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
              20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
              200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]

   majorticks = [1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02]
   majorlabels = ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$']

   xbins = k_bin_edges_par

   ticklist_x = log2lin(minorticks, xbins)
   majorlist_x = log2lin(majorticks, xbins)

   ybins = k_bin_edges_perp

   ticklist_y = log2lin(minorticks, ybins)
   majorlist_y = log2lin(majorticks, ybins)


   ax.set_xticks(ticklist_x, minor=True)
   ax.set_xticks(majorlist_x, minor=False)
   ax.set_xticklabels(majorlabels, minor=False)
   ax.set_yticks(ticklist_y, minor=True)
   ax.set_yticks(majorlist_y, minor=False)
   ax.set_yticklabels(majorlabels, minor=False)

   plt.xlabel(r'$k_{\parallel}$')
   plt.ylabel(r'$k_{\bot}$')
   plt.xlim(0, 1)
   plt.ylim(0, 1)
   #plt.savefig('ps_par_vs_perp_nmodes.png')
   plt.title(titlename, fontsize=12)
   #plt.savefig(titlename)
   plt.show()

#plot_ps(np.mean(beam_ps_smooth_2D, axis=0), 'Beam smoothed PS, mean from 100 signal realizations')
#plot_ps(np.mean(beam_ps_original_2D, axis=0), 'Not smoothed PS, mean from 100 signal realizations')
#plot_TF_1D(beam_k_1D, np.mean(beam_ps_smooth_1D, axis=0), 'Beam smoothed PS', '100realiz_beamsmooth_ps_1d.png')
#plot_TF_1D(beam_k_1D, np.mean(beam_ps_original_1D, axis=0), 'Not smoothed PS', '100realiz_beamnotsmooth_ps_1d.png')






def beam_1D_plot_whole(ps1,ps2,k,tf):
   fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,5))
   #fig = plt.figure()
   #ax[0] = fig.add_subplot(121)
   ax[0].plot(k, ps1*1e-12, label='Smooth', color='darkorchid')
   ax[0].plot(k, ps2*1e-12, label ='Original', color='coral')
   ax[1].plot(k, tf, color='black')
   ax[0].set_xscale('log')
   ax[1].set_xscale('log')
   ax[0].set_yscale('log')
   ax[0].set_ylabel(r'$\tilde{P}_{k}$ [$\mu$K${}^2$ (Mpc)${}^3$]', fontsize=17)
   ax[1].set_ylabel(r'$\tilde{T}^{\mathrm{beam}}_{k}$', fontsize=18)
   ax[0].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   ax[1].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   ax[0].tick_params(axis='x', labelsize=14)
   ax[1].tick_params(axis='x', labelsize=14)
   ax[0].tick_params(axis='y', labelsize=14)
   ax[1].tick_params(axis='y', labelsize=14)
   ax[0].grid(alpha=0.2)
   ax[1].grid(alpha=0.2)
   ax[0].set_xlim(k.min(), k.max())
   ax[1].set_xlim(k.min(), k.max())
   ax[1].set_ylim(0, 1.15)
   ax[0].legend(ncol=2, fontsize=17)
   labnums = [ 0.03,0.1, 0.3,1]
   ax[0].set_xticks(labnums)
   ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   ax[1].set_xticks(labnums)
   ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   plt.tight_layout()
   plt.show()


#beam_1D_plot_whole(np.mean(beam_ps_smooth_1D, axis=0),np.mean(beam_ps_original_1D, axis=0),beam_k_1D,beam_TF_1D)

def beam_2D_plot_whole(figure_name,ps_1, ps_2,TF_2D, k_perp, k_par):
      fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(14,6.5))
      #fig.tight_layout(h_pad=0.005, w_pad=1)
      fig.subplots_adjust(hspace=0.15, wspace=0.15)
      #cbaxes = fig.add_axes([0.072, 0.9, 0.25, 0.04]) 
      norm_tf = mpl.colors.Normalize(vmin=0, vmax=1.2) 
      #norm_ps = mpl.colors.Normalize(vmin=10, vmax=17) 
      norm_ps = colors.LogNorm(vmin=10**(-2), vmax=10**5)
      img1 = ax[0].imshow(ps_1*1e-12, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_ps)
      fig.colorbar(img1, orientation='horizontal', ax=ax[0],fraction=0.046, pad=0.13, ticks=[10**(-2), 10**(-1), 10**0, 10**1, 10**2,10**3,10**4,10**5]).set_label(r'$\langle\tilde{P}^{\mathrm{smooth}}_{k_{\bot}, k_{\parallel}}\rangle$ [$\mu$K${}^2$ (Mpc)${}^3$]', size=18)
  
      img2 = ax[1].imshow(ps_2*1e-12, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_ps)
      fig.colorbar(img2,orientation='horizontal', ax=ax[1], fraction=0.046, pad=0.13, ticks=[10**(-2), 10**(-1), 10**0, 10**1, 10**2,10**3,10**4,10**5]).set_label(r'$\langle\tilde{P}^{\mathrm{original}}_{k_{\bot}, k_{\parallel}}\rangle$ [$\mu$K${}^2$ (Mpc)${}^3$]', size=18)
      img3 = ax[2].imshow(TF_2D(k_perp,k_par), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_tf)
      fig.colorbar(img3,orientation='horizontal', ax=ax[2], fraction=0.046, pad=0.13).set_label(r'$\tilde{T}^{\mathrm{beam}}_{k_{\bot}, k_{\parallel}}$', size=18)
      
     
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
      
      #ax[0].set_title(r'$\langle\tilde{P}^{\mathrm{smooth}}_{k_{\bot}, k_{\parallel}}\rangle$', fontsize=16)
      #ax[1].set_title(r'$\langle\tilde{P}^{\mathrm{original}}_{k_{\bot}, k_{\parallel}}\rangle$', fontsize=16)
      #ax[2].set_title(r'$\tilde{T}^{\mathrm{beam}}_{k_{\bot}, k_{\parallel}}$' , fontsize=16)

      for i in range(3):
         ax[i].set_xticks(ticklist_x, minor=True)
         ax[i].set_xticks(majorlist_x, minor=False)
         ax[i].set_xticklabels(majorlabels, minor=False, fontsize=14)
         ax[i].set_yticks(ticklist_y, minor=True)
         ax[i].set_yticks(majorlist_y, minor=False)
         ax[i].set_yticklabels(majorlabels, minor=False, fontsize=14)
         ax[i].tick_params(labelsize=14)
         #ax[i].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}]$',fontsize=12)
         ax[i].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}]$',fontsize=14)

      ax[0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}]$',fontsize=14)
      plt.tight_layout()
      plt.show()
      plt.savefig(figure_name, bbox_inches='tight',pad_inches = 0) 



#beam_2D_plot_whole('for_beam.png',np.mean(beam_ps_smooth_2D, axis=0), np.mean(beam_ps_original_2D, axis=0),beam_TF_2D_func, beam_k_perp, beam_k_par)



def freq_1D_plot_whole(ps1,ps2,k,tf):
   fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,5))
   #fig = plt.figure()
   #ax[0] = fig.add_subplot(121)
   ax[0].plot(k, ps1*1e-12, label='Low resolution', color='mediumvioletred')
   ax[0].plot(k, ps2*1e-12, label ='High resolution', color='lightsalmon')
   ax[1].plot(k, tf, color='black')
   ax[0].set_xscale('log')
   ax[1].set_xscale('log')
   ax[0].set_yscale('log')
   ax[0].set_ylabel(r'$\tilde{P}_{k}$ [$\mu$K${}^2$ (Mpc)${}^3$]', fontsize=17)
   ax[1].set_ylabel(r'$\tilde{T}^{\mathrm{freq}}_{k}$', fontsize=18)
   ax[0].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   ax[1].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   ax[0].tick_params(axis='x', labelsize=14)
   ax[1].tick_params(axis='x', labelsize=14)
   ax[0].tick_params(axis='y', labelsize=14)
   ax[1].tick_params(axis='y', labelsize=14)
   ax[0].grid(alpha=0.2)
   ax[1].grid(alpha=0.2)
   ax[0].set_xlim(k.min(), k.max())
   ax[1].set_xlim(k.min(), k.max())
   ax[1].set_ylim(0, 1.1)
   ax[0].legend(ncol=2, fontsize=17)
   labnums = [ 0.03,0.1, 0.3,1]
   ax[0].set_xticks(labnums)
   ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   ax[1].set_xticks(labnums)
   ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   plt.tight_layout()
   plt.show()


#freq_1D_plot_whole(np.mean(freq_ps_low_1D, axis=0),np.mean(freq_ps_high_1D, axis=0),freq_k_1D,freq_TF_1D)


def freq_2D_plot_whole(figure_name,ps_1, ps_2,TF_2D, k_perp, k_par):
      fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(14,6.5))
      #fig.tight_layout(h_pad=0.005, w_pad=1)
      fig.subplots_adjust(hspace=0.15, wspace=0.15)
      #cbaxes = fig.add_axes([0.072, 0.9, 0.25, 0.04]) 
      norm_tf = mpl.colors.Normalize(vmin=0, vmax=1.8) 
      norm_ps = colors.LogNorm(vmin=10**(-2), vmax=10**5)
    
      img1 = ax[0].imshow(ps_1*1e-12, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_ps)
      fig.colorbar(img1, orientation='horizontal', ax=ax[0],fraction=0.046, pad=0.13, ticks=[10**(-2), 10**(-1), 10**0, 10**1, 10**2,10**3,10**4,10**5]).set_label(r'$\langle\tilde{P}^{\mathrm{low}}_{k_{\bot}, k_{\parallel}}\rangle$ [$\mu$K${}^2$ (Mpc)${}^3$]', size=18)
  
      img2 = ax[1].imshow(ps_2*1e-12, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_ps)
      fig.colorbar(img2,orientation='horizontal', ax=ax[1], fraction=0.046, pad=0.13,ticks=[10**(-2), 10**(-1), 10**0, 10**1, 10**2,10**3,10**4,10**5]).set_label(r'$\langle\tilde{P}^{\mathrm{high}}_{k_{\bot}, k_{\parallel}}\rangle$ [$\mu$K${}^2$ (Mpc)${}^3$]', size=18)
      img3 = ax[2].imshow(TF_2D(k_perp,k_par), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_tf)
      fig.colorbar(img3,orientation='horizontal', ax=ax[2], fraction=0.046, pad=0.13).set_label(r'$\tilde{T}^{\mathrm{freq}}_{k_{\bot}, k_{\parallel}}$', size=18)
      
     
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
      
      #ax[0].set_title(r'$\langle\tilde{P}^{\mathrm{smooth}}_{k_{\bot}, k_{\parallel}}\rangle$', fontsize=16)
      #ax[1].set_title(r'$\langle\tilde{P}^{\mathrm{original}}_{k_{\bot}, k_{\parallel}}\rangle$', fontsize=16)
      #ax[2].set_title(r'$\tilde{T}^{\mathrm{beam}}_{k_{\bot}, k_{\parallel}}$' , fontsize=16)

      for i in range(3):
         ax[i].set_xticks(ticklist_x, minor=True)
         ax[i].set_xticks(majorlist_x, minor=False)
         ax[i].set_xticklabels(majorlabels, minor=False, fontsize=14)
         ax[i].set_yticks(ticklist_y, minor=True)
         ax[i].set_yticks(majorlist_y, minor=False)
         ax[i].set_yticklabels(majorlabels, minor=False, fontsize=14)
         ax[i].tick_params(labelsize=14)
         #ax[i].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}]$',fontsize=12)
         ax[i].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}]$',fontsize=14)

      ax[0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}]$',fontsize=14)
      plt.tight_layout()
      plt.show()
      plt.savefig(figure_name, bbox_inches='tight',pad_inches = 0) 



#freq_2D_plot_whole('for_freq.png',np.mean(freq_ps_low_2D, axis=0),np.mean(freq_ps_high_2D, axis=0),freq_TF_2D_func, freq_k_perp, freq_k_par)

def pipeline_1D_plot_whole(liss,ces,k,mix):
   fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,5))
   #fig = plt.figure()
   #ax[0] = fig.add_subplot(121)
   ax[0].plot(k, liss, label='Lissajous', color='salmon')
   ax[0].plot(k, ces, label ='CES', color='indigo')
   ax[1].plot(k, mix, color='black', label='Lissajous and CES')
   ax[0].set_xscale('log')
   ax[1].set_xscale('log')
   #ax[0].set_yscale('log')
   ax[0].set_ylabel(r'$\tilde{T}^{\mathrm{pipeline}}_{k}$', fontsize=18)
   ax[1].set_ylabel(r'$\tilde{T}^{\mathrm{pipeline}}_{k}$', fontsize=18)
   ax[0].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   ax[1].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   ax[0].tick_params(axis='x', labelsize=14)
   ax[1].tick_params(axis='x', labelsize=14)
   ax[0].tick_params(axis='y', labelsize=14)
   ax[1].tick_params(axis='y', labelsize=14)
   ax[0].grid(alpha=0.2)
   ax[1].grid(alpha=0.2)
   ax[0].set_xlim(k.min(), k.max())
   ax[1].set_xlim(k.min(), k.max())
   ax[0].set_ylim(0, 1)
   ax[1].set_ylim(0, 1)
   ax[0].legend(ncol=2, fontsize=17)
   ax[1].legend(ncol=2, fontsize=17)
   labnums = [ 0.03,0.1, 0.3,1]
   ax[0].set_xticks(labnums)
   ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   ax[1].set_xticks(labnums)
   ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   plt.tight_layout()
   plt.show()

#pipeline_1D_plot_whole(liss_TF_1D,CES_TF_1D,liss_k_1D,mix_TF_1D)

def pipeline_2D_plot_whole(figure_name,liss, ces,mix):
      fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(14,6.5))
      #fig.tight_layout(h_pad=0.005, w_pad=1)
      fig.subplots_adjust(hspace=0.15, wspace=0.15)
      #cbaxes = fig.add_axes([0.072, 0.9, 0.25, 0.04]) 
      norm_tf = mpl.colors.Normalize(vmin=0, vmax=1.2) 
      
    
      img1 = ax[0].imshow(liss, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_tf)
      fig.colorbar(img1, orientation='horizontal', ax=ax[0],fraction=0.046, pad=0.13).set_label(r'$\tilde{T}^{\mathrm{pipeline}}_{k_{\bot}, k_{\parallel}}$, Lissajous', size=18)
  
      img2 = ax[1].imshow(ces, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_tf)
      fig.colorbar(img2,orientation='horizontal', ax=ax[1], fraction=0.046, pad=0.13).set_label(r'$\tilde{T}^{\mathrm{pipeline}}_{k_{\bot}, k_{\parallel}}$, CES', size=18)
      img3 = ax[2].imshow(mix, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_tf)
      fig.colorbar(img3,orientation='horizontal', ax=ax[2], fraction=0.046, pad=0.13).set_label(r'$\tilde{T}^{\mathrm{pipeline}}_{k_{\bot}, k_{\parallel}}$, Lissajous and CES', size=18)
      
     
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
      
      #ax[0].set_title(r'$\langle\tilde{P}^{\mathrm{smooth}}_{k_{\bot}, k_{\parallel}}\rangle$', fontsize=16)
      #ax[1].set_title(r'$\langle\tilde{P}^{\mathrm{original}}_{k_{\bot}, k_{\parallel}}\rangle$', fontsize=16)
      #ax[2].set_title(r'$\tilde{T}^{\mathrm{beam}}_{k_{\bot}, k_{\parallel}}$' , fontsize=16)

      for i in range(3):
         ax[i].set_xticks(ticklist_x, minor=True)
         ax[i].set_xticks(majorlist_x, minor=False)
         ax[i].set_xticklabels(majorlabels, minor=False, fontsize=14)
         ax[i].set_yticks(ticklist_y, minor=True)
         ax[i].set_yticks(majorlist_y, minor=False)
         ax[i].set_yticklabels(majorlabels, minor=False, fontsize=14)
         ax[i].tick_params(labelsize=14)
         #ax[i].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}]$',fontsize=12)
         ax[i].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}]$',fontsize=14)

      ax[0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}]$',fontsize=14)
      plt.tight_layout()
      plt.show()
      plt.savefig(figure_name, bbox_inches='tight',pad_inches = 0) 

#pipeline_2D_plot_whole('for_pipeline.png',liss_TF_2D,CES_TF_2D, mix_TF_2D)


def total_1D_plot_whole(liss,ces,k,mix):
   fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,5))
   #fig = plt.figure()
   #ax[0] = fig.add_subplot(121)
   ax[0].plot(k, liss(k), label='Lissajous', color='lightsalmon')
   ax[0].scatter(k, liss(k), color='lightsalmon')
   ax[0].plot(k, ces(k), label ='CES', color='mediumvioletred')
   ax[0].scatter(k, ces(k), color='mediumvioletred')
   ax[1].plot(k, mix(k), color='black', label='Lissajous and CES')
   ax[1].scatter(k, mix(k), color='black')
   ax[0].set_xscale('log')
   ax[1].set_xscale('log')
   #ax[0].set_yscale('log')
   ax[0].set_ylabel(r'$\tilde{T}^{\mathrm{total}}_{k}$', fontsize=18)
   ax[1].set_ylabel(r'$\tilde{T}^{\mathrm{total}}_{k}$', fontsize=18)
   ax[0].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   ax[1].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   ax[0].tick_params(axis='x', labelsize=14)
   ax[1].tick_params(axis='x', labelsize=14)
   ax[0].tick_params(axis='y', labelsize=14)
   ax[1].tick_params(axis='y', labelsize=14)
   ax[0].grid(alpha=0.2)
   ax[1].grid(alpha=0.2)
   ax[0].set_xlim(k.min()-0.0004, k.max()+0.04)
   ax[1].set_xlim(k.min()-0.0004, k.max()+0.04)
   ax[0].set_ylim(-0.01, 0.75)
   ax[1].set_ylim(-0.01, 0.75)
   ax[0].legend(ncol=2, fontsize=17)
   ax[1].legend(ncol=2, fontsize=17)
   labnums = [ 0.03,0.1, 0.3,1]
   ax[0].set_xticks(labnums)
   ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   ax[1].set_xticks(labnums)
   ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   plt.tight_layout()
   plt.show()

#total_1D_plot_whole(TF_beam_freq_liss_1D,TF_beam_freq_CES_1D,liss_k_1D,TF_beam_freq_mix_1D)

def total_2D_plot_whole(figure_name,liss, ces,mix,k_perp, k_par):
      fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(14,6.5))
      #fig.tight_layout(h_pad=0.005, w_pad=1)
      fig.subplots_adjust(hspace=0.15, wspace=0.15)
      #cbaxes = fig.add_axes([0.072, 0.9, 0.25, 0.04]) 
      norm_tf = mpl.colors.Normalize(vmin=0, vmax=1) 
      
    
      img1 = ax[0].imshow(liss(k_perp, k_par), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_tf)
      fig.colorbar(img1, orientation='horizontal', ax=ax[0],fraction=0.046, pad=0.13).set_label(r'$\tilde{T}^{\mathrm{total}}_{k_{\bot}, k_{\parallel}}$, Lissajous', size=18)
      #fig.colorbar(img1, orientation='horizontal', ax=ax[0],fraction=0.046, pad=0.13).set_label(r'$\tilde{T}^{\mathrm{total}}_{k_{\bot}, k_{\parallel}}$', size=18)

      img2 = ax[1].imshow(ces(k_perp, k_par), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_tf)
      fig.colorbar(img2,orientation='horizontal', ax=ax[1], fraction=0.046, pad=0.13).set_label(r'$\tilde{T}^{\mathrm{total}}_{k_{\bot}, k_{\parallel}}$, CES', size=18)
      img3 = ax[2].imshow(mix(k_perp, k_par), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm_tf)
      fig.colorbar(img3,orientation='horizontal', ax=ax[2], fraction=0.046, pad=0.13).set_label(r'$\tilde{T}^{\mathrm{total}}_{k_{\bot}, k_{\parallel}}$, Lissajous and CES', size=18)
      
     
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
      
      #ax[0].set_title(r'$\langle\tilde{P}^{\mathrm{smooth}}_{k_{\bot}, k_{\parallel}}\rangle$', fontsize=16)
      #ax[1].set_title(r'$\langle\tilde{P}^{\mathrm{original}}_{k_{\bot}, k_{\parallel}}\rangle$', fontsize=16)
      #ax[2].set_title(r'$\tilde{T}^{\mathrm{beam}}_{k_{\bot}, k_{\parallel}}$' , fontsize=16)

      for i in range(3):
         ax[i].set_xticks(ticklist_x, minor=True)
         ax[i].set_xticks(majorlist_x, minor=False)
         ax[i].set_xticklabels(majorlabels, minor=False, fontsize=14)
         ax[i].set_yticks(ticklist_y, minor=True)
         ax[i].set_yticks(majorlist_y, minor=False)
         ax[i].set_yticklabels(majorlabels, minor=False, fontsize=14)
         ax[i].tick_params(labelsize=14)
         #ax[i].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}]$',fontsize=12)
         ax[i].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}]$',fontsize=14)

      ax[0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}]$',fontsize=14)
      plt.tight_layout()
      plt.show()
      plt.savefig(figure_name, bbox_inches='tight',pad_inches = 0) 

#total_2D_plot_whole('total2.png',TF_beam_mix_2D, TF_beam_freq_CES_2D,TF_beam_freq_mix_2D,mix_k_perp, mix_k_par)

total_2D_plot_whole('for_total.png',TF_beam_freq_liss_2D, TF_beam_freq_CES_2D,TF_beam_freq_mix_2D,mix_k_perp, mix_k_par)

