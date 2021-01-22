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

import tools
import map_cosmo
import xs_class
import PS_function
import itertools as itr
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
np.save('k_co2_ces.npy',np.array(k2[1]))
np.save('xs_co2_ces.npy',np.array(xs_mean2[1]))
np.save('sigma_co2_ces.npy',np.array(xs_sigma2[1]))
print (k2[1],xs_mean2[1],xs_sigma2[1])
'''
[0.01215163 0.0173808  0.02486021 0.03555821 0.05085983 0.07274615
 0.10405072 0.14882648 0.21287041 0.30447412 0.4354973  0.6229032
 0.89095476 1.27435592] [-111394.55879619  -35713.77249337  -63660.34236317   -2403.03453446
  -10983.39995201  -62925.38996002  -51229.31200701  -26009.15152815
   -5208.28158103    7848.84887646   -7672.5316699     -149.65956843
    3036.73274278    -303.45343263] [502402.56502822 260465.31564856 191957.69052262 134950.36658772
  93414.04731901  63488.46854868  41459.50760101  27083.94180769
  17198.45857079  10336.38453305   6368.44938107   3865.63125151
   2795.62660359   5222.61565755]
'''
k6, xs_mean6, xs_sigma6 = read_h5_arrays('co6_map_signal_1D_arrays.h5')
k7, xs_mean7, xs_sigma7 = read_h5_arrays('co7_map_signal_1D_arrays.h5')
print (np.load('co2_map_signal_1D_names.npy'))
'''
['xs_mean_co7_map_elev_cesc0.pdf' 'xs_mean_co7_map_elev_cesc1.pdf'
 'xs_mean_co7_map_dayn_cesc0.pdf' 'xs_mean_co7_map_dayn_cesc1.pdf'
 'xs_mean_co7_map_sidr_cesc0.pdf' 'xs_mean_co7_map_sidr_cesc1.pdf'
 'xs_mean_co7_map_ambt_cesc0.pdf' 'xs_mean_co7_map_ambt_cesc1.pdf'
 'xs_mean_co7_map_wind_cesc0.pdf' 'xs_mean_co7_map_wind_cesc1.pdf'
 'xs_mean_co7_map_wint_cesc0.pdf' 'xs_mean_co7_map_wint_cesc1.pdf'
 'xs_mean_co7_map_rise_cesc0.pdf' 'xs_mean_co7_map_rise_cesc1.pdf']
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
   #ax1.plot(k_th, k_th * ps_th_nobeam * 10, '--', label=r'$10\times kP_{Theory}(k)$', color='dodgerblue')
   #ax1.plot(k_th, k_th * ps_copps_nobeam * 5, 'g--', label=r'$5 \times kP_{COPPS}$ (shot)')
   ax1.set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]', fontsize=14)
   if scan_strategy == 'ces':
      ax1.set_ylim(-lim*3, lim*3)              # ax1.set_ylim(0, 0.1)
   if scan_strategy == 'liss':
      ax1.set_ylim(-lim, lim)              # ax1.set_ylim(0, 0.1)
   ax1.set_xlim(0.04,0.7)
   ax1.set_xscale('log')
   #ax1.set_title(titlename, fontsize=16)
   ax1.grid()
   #ax1.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   labnums = [0.05,0.1, 0.2, 0.5]
   ax1.set_xticks(labnums)
   ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   #plt.legend(bbox_to_anchor=(0, 0.61))
   ax1.legend(ncol=4)
   
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

xs_with_model_3fields('liss_all_fields_map_signal.pdf', k2[0],xs_mean2[0], xs_mean6[0], xs_mean7[0], xs_sigma2[0], xs_sigma6[0], xs_sigma7[0], 'liss')

def log2lin(x, k_edges):
    loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
    logx = np.log10(x) - np.log10(k_edges[0])
    return logx / loglen

def xs_2D_plot(figure_name, k,k_bin_edges_par, k_bin_edges_perp, xs_mean2,xs_mean6,xs_mean7, xs_sigma2,xs_sigma6,xs_sigma7, titlename):
      #k,k_bin_edges_par, k_bin_edges_perp, xs_mean, xs_sigma =  k[3:],k_bin_edges_par[3:], k_bin_edges_perp[3:], xs_mean[3:], xs_sigma[3:]
      fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(15.5,8))
      #fig.tight_layout(h_pad=0.005, w_pad=1)
      fig.subplots_adjust(hspace=-0.5, wspace=0.0)
      #fig.suptitle(titlename, fontsize=16)
      norm = mpl.colors.Normalize(vmin=1.3*np.amin(xs_mean7), vmax=-1.3*np.amin(xs_mean7))  
      norm1 = mpl.colors.Normalize(vmin=1.3*np.amin(xs_mean7/xs_sigma7), vmax=-1.3*np.amin(xs_mean7/xs_sigma7)) 

      img1 = ax[0][0].imshow(xs_mean2/(transfer_filt_2D(k[0],k[1])*transfer_sim_2D(k[0],k[1])), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img1, ax=ax[0][0],fraction=0.046, pad=0.04)
  
      img2 = ax[0][1].imshow(xs_mean6/(transfer_filt_2D(k[0],k[1])*transfer_sim_2D(k[0],k[1]))/transfer_filt_2D(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img2, ax=ax[0][1], fraction=0.046, pad=0.04)
      img3 = ax[0][2].imshow(xs_mean7/(transfer_filt_2D(k[0],k[1])*transfer_sim_2D(k[0],k[1])), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img2, ax=ax[0][2], fraction=0.046, pad=0.04).set_label(r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$]', size=16)
      
      img4 = ax[1][0].imshow(xs_mean2/xs_sigma2, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm1)
      fig.colorbar(img4, ax=ax[1][0],fraction=0.046, pad=0.04)
  
      img5 = ax[1][1].imshow(xs_mean6/xs_sigma6, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm1)
      fig.colorbar(img5, ax=ax[1][1], fraction=0.046, pad=0.04)
      img6 = ax[1][2].imshow(xs_mean7/xs_sigma7, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm1)
      fig.colorbar(img6, ax=ax[1][2], fraction=0.046, pad=0.04).set_label(r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)/\sigma_{\tilde{C}}$', size=16)
      
     
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
      
      ax[0][0].set_title(r'CO2', fontsize=16)
      ax[0][1].set_title(r'CO6', fontsize=16)
      ax[0][2].set_title(r'CO7', fontsize=16)

      for i in range(3):
         for j in range(2):
            ax[j][i].set_xticks(ticklist_x, minor=True)
            ax[j][i].set_xticks(majorlist_x, minor=False)
            ax[j][i].set_xticklabels(majorlabels, minor=False, fontsize=16)
            ax[j][i].set_yticks(ticklist_y, minor=True)
            ax[j][i].set_yticks(majorlist_y, minor=False)
            ax[j][i].set_yticklabels(majorlabels, minor=False, fontsize=16)
      
      ax[1][0].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]',fontsize=16)
      ax[0][0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}$]',fontsize=16)
      ax[1][0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}$]',fontsize=16)
      ax[1][1].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]', fontsize=16)
      ax[1][2].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]', fontsize=16)
      
      plt.tight_layout()
      plt.savefig(figure_name) 
 
'''
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
'''
k2, xs_mean2, xs_sigma2, k_edges_perp2, k_edges_par2 = read_h5_arrays('co2_map_signal_2D_arrays.h5', two_dim=True)
print (xs_mean2[0])
k6, xs_mean6, xs_sigma6, k_edges_perp6, k_edges_par6 = read_h5_arrays('co6_map_signal_2D_arrays.h5', two_dim=True)
k7, xs_mean7, xs_sigma7, k_edges_perp7, k_edges_par7 = read_h5_arrays('co7_map_signal_2D_arrays.h5', two_dim=True)
xs_2D_plot('liss_3fields_2D.pdf', k2[0],k_edges_par2[0], k_edges_perp2[0], xs_mean2[0],xs_mean6[0],xs_mean7[0], xs_sigma2[0],xs_sigma6[0],xs_sigma7[0], 'Liss cans')
xs_2D_plot('ces_3fields_2D.pdf', k2[1],k_edges_par2[1], k_edges_perp2[1], xs_mean2[1],xs_mean6[1],xs_mean7[1], xs_sigma2[1],xs_sigma6[1],xs_sigma7[1], 'CES cans')
    
print (np.load('co6_map_null_1D_names.npy'))
['xs_mean_co6_map_elev_ambtsubtr_cesc0.pdf'
 'xs_mean_co6_map_elev_ambtsubtr_cesc1.pdf'
 'xs_mean_co6_map_elev_windsubtr_cesc0.pdf'
 'xs_mean_co6_map_elev_windsubtr_cesc1.pdf'
 'xs_mean_co6_map_elev_wintsubtr_cesc0.pdf'
 'xs_mean_co6_map_elev_wintsubtr_cesc1.pdf'
 'xs_mean_co6_map_elev_risesubtr_cesc0.pdf'
 'xs_mean_co6_map_elev_risesubtr_cesc1.pdf'
 'xs_mean_co6_map_elev_halfsubtr_cesc0.pdf'
 'xs_mean_co6_map_elev_halfsubtr_cesc1.pdf'
 'xs_mean_co6_map_elev_oddesubtr_cesc0.pdf'
 'xs_mean_co6_map_elev_oddesubtr_cesc1.pdf'
 'xs_mean_co6_map_elev_fpolsubtr_cesc0.pdf'
 'xs_mean_co6_map_elev_fpolsubtr_cesc1.pdf'
 'xs_mean_co6_map_elev_daynsubtr_cesc0.pdf'
 'xs_mean_co6_map_elev_daynsubtr_cesc1.pdf']



def plot_sub_fig(field,jk_we_want,ax,lim,cesc):
   if field == 'CO2':
      k, xs_mean, xs_sigma = read_h5_arrays('co6_map_null_1D_arrays.h5') #replace with co2 when it will be done
   if field == 'CO6':
      k, xs_mean, xs_sigma = read_h5_arrays('co6_map_null_1D_arrays.h5')
   if field == 'CO7':
      k, xs_mean, xs_sigma = read_h5_arrays('co7_map_null_1D_arrays.h5')
   ax.plot(k[0] 0 * xs_mean[0], 'k', alpha=0.4)
   for index in jk_we_want:
      if index == 4 or index == 5:
         label_name = 'wint'
         color_name = 'teal'
      if index == 8 or index == 9:
         label_name = 'half'
         color_name = 'indianred'
      if index == 10 or index == 11:
         label_name = 'odde'
         color_name = 'purple'
      if index == 14 or index == 14:
         label_name = 'dayn'
         color_name = 'forestgreen'
      ax.errorbar(k[index], k[index] * xs_mean[index] / (transfer(k[index])*transfer_filt(k[index])), k[index] * xs_sigma[index] / (transfer(k[index])*transfer_filt(k[index])), fmt='o', label=label_name, color=color_name)
   if ax == ax1:
      ax.set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]', fontsize=14)

   if cesc == '1':
      ax.set_ylim(-lim*3, lim*3)          
   if cesc == '0':
      ax.set_ylim(-lim, lim)         
   ax.set_xlim(0.04,0.7)
   ax.set_xscale('log')
   ax.set_title(field, fontsize=16)
   ax.grid()
   ax.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   labnums = [0.05,0.1, 0.2, 0.5]
   ax.set_xticks(labnums)
   ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   #plt.legend(bbox_to_anchor=(0, 0.61))
   ax.legend(ncol=4)


def plot_nulltest(cesc):
   k7, xs_mean7, xs_sigma7 = read_h5_arrays('co7_map_null_1D_arrays.h5') 
   xs_mean7 = xs_mean7[8]
   lim = np.mean(np.abs(xs_mean7[4:-2] * k7[4:-2])) * 8
   if cesc = '0':
      jk_we_want = [4,8,10,14] #indices of jk we want to use: wint, half, odde, dayn
   if cesc = '1':
      jk_we_want = [5,9,11,15]

   ax1 = fig.add_subplot(311)
   plot_sub_fig('CO2',jk_we_want,ax1,lim,cesc)
   ax2 = fig.add_subplot(321)
   plot_sub_fig('CO6',jk_we_want,ax2,lim,cesc)
   ax3 = fig.add_subplot(331)
   plot_sub_fig('CO7',jk_we_want,ax3,lim,cesc)
   plt.tight_layout()
   if cesc = '0':
      plt.savefig('nulltests_3fields_liss.pdf')
    if cesc = '1':
      plt.savefig('nulltests_3fields_ces.pdf')

plot_nulltest('0')
plot_nulltest('1')

