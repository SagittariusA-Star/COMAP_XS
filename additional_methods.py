import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy.fft as fft
import corner
import h5py
import sys
import scipy.interpolate

import tools
import map_cosmo
import xs_class
import PS_function
from scipy.optimize import curve_fit
#theory spectrum
k_th = np.load('k.npy')
ps_th = np.load('ps.npy')
ps_th_nobeam = np.load('psn.npy') #instrumental beam, less sensitive to small scales line broadening, error bars go up at high k, something with the intrinsic resolution of the telescope (?)

#in 2D
ps_2d_smooth = np.load('ps_2d_smooth.npy')
ps_2d_notsmooth = np.load('ps_2d_notsmooth.npy')
#ps_2d_smooth = np.load('smooth_mean.npy')
#ps_2d_notsmooth = np.load('notsmooth_mean.npy')
#ps_2d_smooth = np.load('ps_smooth_single.npy') #'ps_2dfrom3d.npy'
#ps_2d_notsmooth = np.load('ps_notsmooth_single.npy')

k_smooth = np.load('k_smooth.npy')
#k_notsmooth = np.load('k_notsmooth.npy')

#print (ps_2d_smooth/ps_2d_notsmooth)

k_perp_sim = k_smooth[0]
k_par_sim = k_smooth[1]

transfer_sim_2D = scipy.interpolate.interp2d(k_perp_sim, k_par_sim, ps_2d_smooth/ps_2d_notsmooth)
#values from COPPS
ps_copps = 8.746e3 * ps_th / ps_th_nobeam #shot noise level
ps_copps_nobeam = 8.7e3

transfer = scipy.interpolate.interp1d(k_th, ps_th / ps_th_nobeam) #transfer(k) always < 1, values at high k are even larger and std as well
P_theory = scipy.interpolate.interp1d(k_th,ps_th_nobeam)

#Read the transfer function associated with effects of filtering
def filtering_TF(filename, dim):
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

k_filtering_1D, TF_filtering_1D = filtering_TF('TF_1d.h5', 1)
transfer_filt = scipy.interpolate.interp1d(k_filtering_1D, TF_filtering_1D) 

k_perp_filt, k_par_filt, TF_filtering_2D = filtering_TF('TF_2d.h5', 2)
transfer_filt_2D = scipy.interpolate.interp2d(k_perp_filt, k_par_filt, TF_filtering_2D)


def read_h5_arrays(filename, two_dim=False):
   with h5py.File(filename, mode="r") as my_file:
       k = np.array(my_file['k'][:]) 
       xs_mean = np.array(my_file['xs_mean'][:]) 
       xs_sigma = np.array(my_file['xs_sigma'][:]) 
       if two_dim == True:
          k_edges_perp = np.array(my_file['k_edges_perp'][:]) 
          k_edges_par = np.array(my_file['k_edges_par'][:]) 
          return k, xs_mean, xs_sigma, k_edges_perp, k_edges_par
       else:
          return k, xs_mean, xs_sigma

k2, xs_mean2, xs_sigma2 = read_h5_arrays('co2_map_signal_1D_arrays.h5')
k6, xs_mean6, xs_sigma6 = read_h5_arrays('co6_map_signal_1D_arrays.h5')
k7, xs_mean7, xs_sigma7 = read_h5_arrays('co7_map_signal_1D_arrays.h5')
print (np.load('co7_map_signal_1D_names.npy'))
'''
[u'xs_mean_co7_map_dayn_cesc0.pdf' u'xs_mean_co7_map_dayn_cesc1.pdf'
 u'xs_mean_co7_map_elev_cesc0.pdf' u'xs_mean_co7_map_elev_cesc1.pdf']
'''

def xs_with_model_3fields(figure_name, k, xs_mean2, xs_mean6, xs_mean7, xs_sigma2, xs_sigma6, xs_sigma7, scan_strategy):
  
   if scan_strategy == 'ces':
      titlename = 'CES scans'
   if scan_strategy == 'liss':
      titlename = 'Lissajous scans'
   
   k_offset = k*0.025
   k6 = k - k_offset
   k7 = k + k_offset
   lim = np.mean(np.abs(xs_mean2[4:-2] * k[4:-2])) * 8
   fig = plt.figure()
   #fig.set_figwidth(8)
   ax1 = fig.add_subplot(211)
  
   ax1.errorbar(k6, k * xs_mean6 / (transfer(k)*transfer_filt(k)), k * xs_sigma6 / (transfer(k)*transfer_filt(k)), fmt='o', label=r'co6', color='teal', zorder=3)
   ax1.errorbar(k7, k * xs_mean7 / (transfer(k)*transfer_filt(k)), k * xs_sigma7 / (transfer(k)*transfer_filt(k)), fmt='o', label=r'co7', color='purple', zorder=2)
   ax1.errorbar(k, k * xs_mean2 / (transfer(k)*transfer_filt(k)), k * xs_sigma2 / (transfer(k)*transfer_filt(k)), fmt='o', label=r'co2', color='indianred', zorder=4)
   #ax1.errorbar(k, k * xs_mean, k * xs_sigma, fmt='o', label=r'$k\tilde{C}_{data}(k)$')
   ax1.plot(k, 0 * xs_mean2, 'k', alpha=0.4, zorder=1)
   #ax1.plot(k, k*PS_function.PS_f(k)/ transfer(k), label='k*PS of the input signal')
   #ax1.plot(k, k*PS_function.PS_f(k), label='k*PS of the input signal')
   #ax1.plot(k_th, k_th * ps_th_nobeam * 10, '--', label=r'$10 \times kP_{Theory}(k)$', color='dodgerblue')
   #ax1.plot(k_th, k_th * ps_copps_nobeam * 5, 'g--', label=r'$5 \times kP_{COPPS}$ (shot)')
   ax1.set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]', fontsize=14)
   if scan_strategy == 'ces':
      ax1.set_ylim(-lim*3, lim*3)              # ax1.set_ylim(0, 0.1)
   if scan_strategy == 'liss':
      ax1.set_ylim(-lim, lim)              # ax1.set_ylim(0, 0.1)
   ax1.set_xlim(0.04,0.7)
   ax1.set_xscale('log')
   ax1.set_title(titlename)
   ax1.grid()
   #ax1.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   labnums = [0.05,0.1, 0.2, 0.5]
   ax1.set_xticks(labnums)
   ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   #plt.legend(bbox_to_anchor=(0, 0.61))
   ax1.legend(ncol=3)
   
   ax2 = fig.add_subplot(212)
   #ax2.plot(k, diff_mean / error, fmt='o', label=r'$\tilde{C}_{diff}(k)$', color='black')
   
   ax2.errorbar(k6, xs_mean6 / xs_sigma6, xs_sigma6/xs_sigma6, fmt='o', label=r'co6', color='teal', zorder=3)
   ax2.errorbar(k7, xs_mean7 / xs_sigma7, xs_sigma7/xs_sigma7, fmt='o', label=r'co7', color='purple', zorder=2)
   ax2.errorbar(k, xs_mean2 / xs_sigma2, xs_sigma2/xs_sigma2, fmt='o', label=r'co2', color='indianred', zorder=4)
   #ax2.errorbar(k, sum_mean / error, error /error, fmt='o', label=r'$\tilde{C}_{sum}(k)$', color='mediumorchid')
   ax2.plot(k, 0 * xs_mean2, 'k', alpha=0.4, zorder=1)
   #ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
   ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$', fontsize=14)
   ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   ax2.set_ylim(-5, 5)
   ax2.set_xlim(0.04,0.7)
   ax2.set_xscale('log')
   ax2.grid()
   ax2.legend(ncol=3)
   ax2.set_xticks(labnums)
   ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   
   plt.tight_layout()
   #plt.legend()
   plt.savefig(figure_name, bbox_inches='tight')
   #plt.show()

xs_with_model_3fields('ces_all_fields_map_signal.pdf', k2[3],xs_mean2[3], xs_mean6[3], xs_mean7[3], xs_sigma2[3], xs_sigma6[3], xs_sigma7[3], 'ces')
