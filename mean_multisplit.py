import numpy as np
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


def read_number_of_splits(mapfile, jk):
   with h5py.File(mapfile, mode="r") as my_file:
       my_map = np.array(my_file['/jackknives/map_' + jk])
       sh = my_map.shape   
       number_of_splits = sh[0]   
   return number_of_splits

def xs_feed_feed_grid(map_file):
   #n_sim = 100
   n_k = 14
   n_feed = 19
   n_sum = 0
   xs_sum = np.zeros(n_k)
   #rms_xs_sum = np.zeros((n_k, n_sim))
   xs_div = np.zeros(n_k)
   map_file = 'split_maps/' + map_file
   name_of_map = map_file.split('/')[-1] #get rid of the path, leave only the name of the map
   name_of_map = name_of_map.split('.')[0] #get rid of the ".h5" part
   name_of_map_list = name_of_map.split('_') #co6_map_snup_elev_0_cesc_0'
   field = name_of_map_list[0]
   ff_jk = name_of_map_list[2]
   split_names = []
   split_numbers = []
   for m in range(3, len(name_of_map_list)-1,2):
      split_names.append(name_of_map_list[m])
   for n in range(4, len(name_of_map_list),2):
      split_numbers.append(name_of_map_list[n])
   n_of_splits = read_number_of_splits(map_file, ff_jk)
   n_list = list(range(n_of_splits))
   all_different_possibilities = list(itr.combinations(n_list, 2)) #for n_of_splits = 3, it gives [(0, 1), (0, 2), (1, 2)]
   how_many_combinations = len(all_different_possibilities)
   for u in range(how_many_combinations): #go through all the split combinations
      current_combo = all_different_possibilities[u]    
      split1 = str(current_combo[0])
      split2 = str(current_combo[1])
      path_to_xs = 'spectra/xs_' + name_of_map + '_split' + split1 + '_feed%01i_and_' + name_of_map + '_split' + split2 + '_feed%01i.h5'
      
      xs = np.zeros((n_feed, n_feed, n_k))
      rms_xs_std = np.zeros_like(xs)
      chi2 = np.zeros((n_feed, n_feed))
      k = np.zeros(n_k)
      noise = np.zeros_like(chi2)
      for i in range(n_feed): #go through all the feed combinations
         for j in range(n_feed):
           # if i != 7 and j != 7:
              try:
                  filepath = path_to_xs %(i+1, j+1)
                  with h5py.File(filepath, mode="r") as my_file:
                      #print ("finds file", i, j)
                      xs[i, j] = np.array(my_file['xs'][:])
                      #print (xs[i,j])
                      rms_xs_std[i, j] = np.array(my_file['rms_xs_std'][:])
                      #print (rms_xs_std[i,j])
                      k[:] = np.array(my_file['k'][:])
              except:
                  xs[i, j] = np.nan
                  rms_xs_std[i, j] = np.nan
            
              w = np.sum(1 / rms_xs_std[i,j])
              noise[i,j] = 1 / np.sqrt(w)
              chi3 = np.sum((xs[i,j] / rms_xs_std[i,j]) ** 3) #we need chi3 to take the sign into account - positive or negative correlation

              chi2[i, j] = np.sign(chi3) * abs((np.sum((xs[i,j] / rms_xs_std[i,j]) ** 2) - n_k) / np.sqrt(2 * n_k)) #magnitude (how far from white noise)
            
              
              if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and i != j:  #if excess power is smaller than 5 sigma, chi2 is not nan, not on diagonal
                  xs_sum += xs[i,j] / rms_xs_std[i,j] ** 2
                  #print ("if test worked")
                  xs_div += 1 / rms_xs_std[i,j] ** 2
                  n_sum += 1

      tools.ensure_dir_exists('chi2_grids')
      figure_name = 'chi2_grids/xs_grid_' + name_of_map + '_splits' + split1 + split2 + '.pdf'
      plt.figure()
      vmax = 15
      plt.imshow(chi2, interpolation='none', vmin=-vmax, vmax=vmax, extent=(0.5, n_feed + 0.5, n_feed + 0.5, 0.5))
      new_tick_locations = np.array(range(n_feed)) + 1
      plt.xticks(new_tick_locations)
      plt.yticks(new_tick_locations)
      plt.xlabel('Feed of ' + ff_jk + ' ' + split1 + '-split')
      plt.ylabel('Feed of ' + ff_jk + ' ' + split2 + '-split')
      cbar = plt.colorbar()
      cbar.set_label(r'$|\chi^2| \times$ sign($\chi^3$)')
      plt.savefig(figure_name, bbox_inches='tight')
      
      #plt.show()
      #print ("xs_div:", xs_div)
   return k, xs_sum / xs_div, 1. / np.sqrt(xs_div), field, ff_jk, split_names, split_numbers



def xs_with_model(figure_name, k, xs_mean, xs_sigma, titlename, scan_strategy):
  
   if scan_strategy == 'ces':
      plotcolor = 'indianred'
   if scan_strategy == 'liss':
      plotcolor = 'teal'

   lim = np.mean(np.abs(xs_mean[4:-2] * k[4:-2])) * 8
   fig = plt.figure()
   #fig.set_figwidth(8)
   ax1 = fig.add_subplot(211)
  
   ax1.errorbar(k, k * xs_mean / (transfer(k)*transfer_filt(k)), k * xs_sigma / (transfer(k)*transfer_filt(k)), fmt='o', color=plotcolor)
   #ax1.errorbar(k, k * xs_mean, k * xs_sigma, fmt='o', label=r'$k\tilde{C}_{data}(k)$')
   ax1.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   #ax1.plot(k, k*PS_function.PS_f(k)/ transfer(k), label='k*PS of the input signal') #for simulated map
   #ax1.plot(k, k*PS_function.PS_f(k), label='k*PS of the input signal')
   #ax1.plot(k_th, k_th * ps_th_nobeam * 10, '--', label=r'$10 \times kP_{Theory}(k)$', color='dodgerblue')
   #ax1.plot(k_th, k_th * ps_copps_nobeam * 5, 'g--', label=r'$5 \times kP_{COPPS}$ (shot)')
   ax1.set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]', fontsize=14)
   if not np.isnan(lim):
      if scan_strategy == 'ces':
         ax1.set_ylim(-lim*3, lim*3)              # ax1.set_ylim(0, 0.1)
      if scan_strategy == 'liss':
         ax1.set_ylim(-lim*2, lim*2)              # ax1.set_ylim(0, 0.1)
   ax1.set_xlim(0.04,1.)
   ax1.set_xscale('log')
   ax1.set_title(titlename)
   ax1.grid()
   #ax1.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   labnums = [0.05,0.1, 0.2, 0.5,1.]
   ax1.set_xticks(labnums)
   ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   #plt.legend(bbox_to_anchor=(0, 0.61))
   #ax1.legend(ncol=3)
   
   ax2 = fig.add_subplot(212)
   #ax2.plot(k, diff_mean / error, fmt='o', label=r'$\tilde{C}_{diff}(k)$', color='black')
  
   ax2.errorbar(k, xs_mean / xs_sigma, xs_sigma/xs_sigma, fmt='o', color=plotcolor)
   #ax2.errorbar(k, sum_mean / error, error /error, fmt='o', label=r'$\tilde{C}_{sum}(k)$', color='mediumorchid')
   ax2.plot(k, 0 * xs_mean, 'k', alpha=0.4)
   #ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
   ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$', fontsize=14)
   ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   ax2.set_ylim(-5, 5)
   ax2.set_xlim(0.04,1.)
   ax2.set_xscale('log')
   ax2.grid()
   #ax2.legend(ncol=3)
   ax2.set_xticks(labnums)
   ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   
   plt.tight_layout()
   #plt.legend()
   tools.ensure_dir_exists('xs_mean_figures')
   plt.savefig('xs_mean_figures/' + figure_name, bbox_inches='tight')
   plt.close(fig)
   #plt.show()


def log2lin(x, k_edges):
    loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
    logx = np.log10(x) - np.log10(k_edges[0])
    return logx / loglen

def xs_feed_feed_2D(map_file):
  
   n_k = 14
   n_feed = 19
   n_sum = 0
   xs_sum = np.zeros((n_k,n_k))

   xs_div = np.zeros((n_k,n_k))
   map_file = 'split_maps/' + map_file
   name_of_map = map_file.split('/')[-1] #get rid of the path, leave only the name of the map
   name_of_map = name_of_map.split('.')[0] #get rid of the ".h5" part
   name_of_map_list = name_of_map.split('_') #co6_map_snup_elev_0_cesc_0'
   field = name_of_map_list[0]
   ff_jk = name_of_map_list[2]
   split_names = []
   split_numbers = []
   for m in range(3, len(name_of_map_list)-1,2):
      split_names.append(name_of_map_list[m])
   for n in range(4, len(name_of_map_list),2):
      split_numbers.append(name_of_map_list[n])
   n_of_splits = read_number_of_splits(map_file, ff_jk)
   n_list = list(range(n_of_splits))
   all_different_possibilities = list(itr.combinations(n_list, 2)) #for n_of_splits = 3, it gives [(0, 1), (0, 2), (1, 2)]
   how_many_combinations = len(all_different_possibilities)
   for u in range(how_many_combinations): #go through all the split combinations
      current_combo = all_different_possibilities[u]    
      split1 = str(current_combo[0])
      split2 = str(current_combo[1])
      path_to_xs = 'spectra_2D/xs_2D_' + name_of_map + '_split' + split1 + '_feed%01i_and_' + name_of_map + '_split' + split2 + '_feed%01i.h5'
      
      k_bin_edges_par = np.zeros(n_k+1)
      k_bin_edges_perp = np.zeros(n_k+1)
      xs = np.zeros((n_feed, n_feed, n_k, n_k))
      rms_xs_std = np.zeros_like(xs)
      chi2 = np.zeros((n_feed, n_feed))
      k = np.zeros((2,n_k))
      noise = np.zeros_like(chi2)
      for i in range(n_feed): #go through all the feed combinations
         for j in range(n_feed):
           # if i != 7 and j != 7:
              try:
                  filepath = path_to_xs %(i+1, j+1)
                  with h5py.File(filepath, mode="r") as my_file:
                      #print ("finds file", i, j)
                      xs[i, j] = np.array(my_file['xs_2D'][:])
                      #print (xs[i,j])
                      rms_xs_std[i, j] = np.array(my_file['rms_xs_std_2D'][:])
                      #print (rms_xs_std[i,j])
                      k[:] = np.array(my_file['k'][:])
                      k_bin_edges_par[:] = np.array(my_file['k_bin_edges_par'][:])
                      k_bin_edges_perp[:] = np.array(my_file['k_bin_edges_perp'][:])
              except:
                  xs[i, j] = np.nan
                  rms_xs_std[i, j] = np.nan
            
              #w = np.sum(1 / rms_xs_std[i,j])
              #noise[i,j] = 1 / np.sqrt(w)
              chi3 = np.sum((xs[i,j] / rms_xs_std[i,j]) ** 3) #we need chi3 to take the sign into account - positive or negative correlation

              chi2[i, j] = np.sign(chi3) * abs((np.sum((xs[i,j] / rms_xs_std[i,j]) ** 2) - n_k*n_k) / np.sqrt(2 * n_k*n_k)) #magnitude (how far from white noise)
             
              if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and i != j:  #if excess power is smaller than 5 sigma, chi2 is not nan, not on diagonal
              
              #if not np.isnan(xs[i,j,0,0]) and i != j:
                  xs_sum += xs[i,j] / rms_xs_std[i,j] ** 2
                  #xs_sum += xs[i,j]
                  #print rms_xs_std[i,j]
                  #print xs[i,j] / rms_xs_std[i,j] ** 2
                 # print 'xs', xs[i,j]
                  #print 'rms', rms_xs_std[i,j]
                  #print ("if test worked")
                  xs_div += 1 / rms_xs_std[i,j] ** 2
                  n_sum += 1

      xs_mean = xs_sum / xs_div
      #xs_mean = xs_sum/n_sum
      #print xs_mean
      xs_sigma =  1. / np.sqrt(xs_div)
     
   return k,k_bin_edges_par, k_bin_edges_perp, xs_mean, xs_sigma, field, ff_jk, split_names, split_numbers


def xs_2D_plot(figure_name, k,k_bin_edges_par, k_bin_edges_perp, xs_mean, xs_sigma, titlename):
      fig, (ax1, ax2) = plt.subplots(1, 2)
      

      img1 = ax1.imshow(xs_mean, interpolation='none', origin='lower',extent=[0,1,0,1])
      cbar1 = fig.colorbar(img1, ax=ax1)
      #cbar1.set_label(r'$\tilde{C}_{\parallel, \bot}(k)$ [$\mu$K${}^2$ (Mpc)${}^3$]')
  
      img2 = ax2.imshow(xs_mean/transfer_filt_2D(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1])
      
      
      cbar2 = fig.colorbar(img2, ax=ax2)
      cbar2.set_label(r'$\tilde{C}_{\parallel, \bot}(k)$ [$\mu$K${}^2$ (Mpc)${}^3$]')
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
      
      ax1.set_title(r'$\overline{xs}$')
      ax1.set_xticks(ticklist_x, minor=True)
      ax1.set_xticks(majorlist_x, minor=False)
      ax1.set_xticklabels(majorlabels, minor=False)
      ax1.set_yticks(ticklist_y, minor=True)
      ax1.set_yticks(majorlist_y, minor=False)
      ax1.set_yticklabels(majorlabels, minor=False)
      
      ax2.set_title(r'$\overline{xs}/TF$')
      ax2.set_xticks(ticklist_x, minor=True)
      ax2.set_xticks(majorlist_x, minor=False)
      ax2.set_xticklabels(majorlabels, minor=False)
      ax2.set_yticks(ticklist_y, minor=True)
      ax2.set_yticks(majorlist_y, minor=False)
      ax2.set_yticklabels(majorlabels, minor=False)
      
      ax1.set_xlabel(r'$k_{\parallel}$')
      ax1.set_ylabel(r'$k_{\bot}$')
    
      ax2.set_xlabel(r'$k_{\parallel}$')
      #ax2.set_ylabel(r'$k_{\bot}$')
     
 
      fig.suptitle(titlename)
      tools.ensure_dir_exists('xs_2D_mean_figures')
      plt.savefig('xs_2D_mean_figures/' + figure_name)
    


#k, xs_mean, xs_sigma, field, ff_jk, split_names, split_numbers = xs_feed_feed_2D('co6_map_snup_elev_0_cesc_0.h5')
#print k[0], k[1]
'''
[0.02361234 0.03214194 0.04375272 0.05955771 0.081072   0.11035801
 0.15022313 0.20448891 0.27835735 0.37890963 0.51578486 0.70210415
 0.95572839 1.30097046] [0.01194748 0.01660097 0.02306697 0.03205145 0.04453534 0.06188166
 0.08598428 0.11947477 0.16600966 0.23066968 0.32051448 0.44535342
 0.61881657 0.85984284]
'''


