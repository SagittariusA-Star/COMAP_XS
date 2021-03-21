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
import pixel_window_TF
#import matplotlib.colors as colors

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


def log2lin(x, k_edges):
    loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
    logx = np.log10(x) - np.log10(k_edges[0])
    return logx / loglen


def xs_2D_plot_null(figure_name, k,k_bin_edges_par, k_bin_edges_perp, xs_mean1,xs_mean2,xs_mean3,xs_mean4, xs_mean5,xs_mean6, titlename, test1, test2, test3):
      #k,k_bin_edges_par, k_bin_edges_perp, xs_mean, xs_sigma =  k[3:],k_bin_edges_par[3:], k_bin_edges_perp[3:], xs_mean[3:], xs_sigma[3:]
      fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(15.5,8))
      #fig.tight_layout(h_pad=0.005, w_pad=1)
      fig.subplots_adjust(hspace=-0.5, wspace=0.0)
      #fig.suptitle(titlename, fontsize=16)
      #norm = mpl.colors.Normalize(vmin=1.3*np.amin(xs_mean7), vmax=-1.3*np.amin(xs_mean7))  
      #norm1 = mpl.colors.Normalize(vmin=1.3*np.amin(xs_mean7/xs_sigma7), vmax=-1.3*np.amin(xs_mean7/xs_sigma7)) 
      norm = mpl.colors.Normalize(vmin=-1000000, vmax=1000000)  #here it was 800000
      norm1 = mpl.colors.Normalize(vmin=-5, vmax=5) 

    
      img1 = ax[0][0].imshow(xs_mean1/(transfer_filt_2D(k[0],k[1])*transfer_sim_2D(k[0],k[1])), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img1, ax=ax[0][0],fraction=0.046, pad=0.04)
  
      img2 = ax[0][1].imshow(xs_mean2/(transfer_filt_2D(k[0],k[1])*transfer_sim_2D(k[0],k[1])), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img2, ax=ax[0][1], fraction=0.046, pad=0.04)
      img3 = ax[0][2].imshow(xs_mean3/(transfer_filt_2D(k[0],k[1])*transfer_sim_2D(k[0],k[1])), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img3, ax=ax[0][2], fraction=0.046, pad=0.04).set_label(r'CES', size=14)
      

 
      img4 = ax[1][0].imshow(xs_mean4/(transfer_filt_2D(k[0],k[1])*transfer_sim_2D(k[0],k[1])), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img4, ax=ax[1][0],fraction=0.046, pad=0.04)
  
      img5 = ax[1][1].imshow(xs_mean5/(transfer_filt_2D(k[0],k[1])*transfer_sim_2D(k[0],k[1])), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img5, ax=ax[1][1], fraction=0.046, pad=0.04)
      img6 = ax[1][2].imshow(xs_mean6/(transfer_filt_2D(k[0],k[1])*transfer_sim_2D(k[0],k[1])), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img6, ax=ax[1][2], fraction=0.046, pad=0.04).set_label(r'Liss', size=14)
      
     
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
      
      ax[0][0].set_title(test1, fontsize=16)
      ax[0][1].set_title(test2, fontsize=16)
      ax[0][2].set_title(test3, fontsize=16)

      for i in range(3):
         for j in range(2):
            ax[j][i].set_xticks(ticklist_x, minor=True)
            ax[j][i].set_xticks(majorlist_x, minor=False)
            ax[j][i].set_xticklabels(majorlabels, minor=False, fontsize=12)
            ax[j][i].set_yticks(ticklist_y, minor=True)
            ax[j][i].set_yticks(majorlist_y, minor=False)
            ax[j][i].set_yticklabels(majorlabels, minor=False, fontsize=12)
            ax[j][i].tick_params(labelsize=12)
      
      ax[1][0].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]',fontsize=14)
      ax[0][0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}$]',fontsize=14)
      ax[1][0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}$]',fontsize=14)
      ax[1][1].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]', fontsize=14)
      ax[1][2].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]', fontsize=14)
      
      plt.tight_layout()
      plt.savefig(figure_name) 
    
#print (np.load('co6_map_null_1D_names.npy'))
'''
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
'''


def plot_null_for_field(field):
   k2, xs_mean2, xs_sigma2, k_edges_perp2, k_edges_par2 = read_h5_arrays(field + '_map_null_2D_arrays.h5', two_dim=True)
   xs_2D_plot_null(field + '_2D_null1.pdf', k2[0],k_edges_par2[0], k_edges_perp2[0], xs_mean2[1],xs_mean2[3], xs_mean2[5],xs_mean2[0],xs_mean2[2],xs_mean2[4],'CO2', 'ambt', 'wind', 'wint')

   xs_2D_plot_null(field + '_2D_null2.pdf', k2[0],k_edges_par2[0], k_edges_perp2[0], xs_mean2[7],xs_mean2[9], xs_mean2[11],xs_mean2[6],xs_mean2[8],xs_mean2[10],'CO2', 'rise', 'half', 'odde')

   xs_2D_plot_null(field + '_2D_null3.pdf', k2[0],k_edges_par2[0], k_edges_perp2[0], xs_mean2[11],xs_mean2[13], xs_mean2[15],xs_mean2[10],xs_mean2[12],xs_mean2[14],'CO2', 'odde', 'fpol', 'dayn')

plot_null_for_field('co2')
plot_null_for_field('co6')
plot_null_for_field('co7')
