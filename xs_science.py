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

#import transfer_functions as TF
from transfer_functions import TF_beam_freq_liss_1D as TF_liss_1D
from transfer_functions import TF_beam_freq_CES_1D as TF_CES_1D
from transfer_functions import TF_beam_freq_liss_2D as TF_liss_2D
from transfer_functions import TF_beam_freq_CES_2D as TF_CES_2D
from transfer_functions import TF_beam_freq_mix_1D as TF_mix_1D

#the function to extract data from pre-computed mean pseudo cross spectra
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

#-----------------------NULL TESTS--------------------------------------------------------------
#Print names to know which indices correspond to various data splits
#print (np.load('co6_map_null_1D_names.npy'))
#['xs_mean_co6_map_elev_ambtsubtr_cesc0.pdf' 0
# 'xs_mean_co6_map_elev_ambtsubtr_cesc1.pdf' 1
# 'xs_mean_co6_map_elev_windsubtr_cesc0.pdf' 2
# 'xs_mean_co6_map_elev_windsubtr_cesc1.pdf' 3
# 'xs_mean_co6_map_elev_wintsubtr_cesc0.pdf' 4
# 'xs_mean_co6_map_elev_wintsubtr_cesc1.pdf' 5
# 'xs_mean_co6_map_elev_risesubtr_cesc0.pdf' 6
# 'xs_mean_co6_map_elev_risesubtr_cesc1.pdf' 7
# 'xs_mean_co6_map_elev_halfsubtr_cesc0.pdf' 8
# 'xs_mean_co6_map_elev_halfsubtr_cesc1.pdf' 9
# 'xs_mean_co6_map_elev_oddesubtr_cesc0.pdf' 10
# 'xs_mean_co6_map_elev_oddesubtr_cesc1.pdf' 11
# 'xs_mean_co6_map_elev_fpolsubtr_cesc0.pdf' 12
# 'xs_mean_co6_map_elev_fpolsubtr_cesc1.pdf' 13 
# 'xs_mean_co6_map_elev_daynsubtr_cesc0.pdf' 14
# 'xs_mean_co6_map_elev_daynsubtr_cesc1.pdf'] 15


def plot_sub_fig(field,jk_we_want,ax_i,lim,cesc,ax, TF, scan):
   if field == 'CO2':
      k, xs_mean, xs_sigma = read_h5_arrays('co2_map_null_1D_arrays.h5')
      
   if field == 'CO6':
      k, xs_mean, xs_sigma = read_h5_arrays('co6_map_null_1D_arrays.h5')
      
   if field == 'CO7':
      k, xs_mean, xs_sigma = read_h5_arrays('co7_map_null_1D_arrays.h5')
     
   ax[ax_i].plot(k[0], 0 * xs_mean[0], 'k', alpha=0.4)
   
   for index in jk_we_want:
      if index == 4 or index == 5:
         kt = -0.015
         
         label_name = 'wint'
         color_name = 'teal'
         l1 = ax[ax_i].errorbar(k[index]+k[index]*kt, k[index] * xs_mean[index] /TF(k[index]), k[index] * xs_sigma[index] / TF(k[index]), fmt='o', label=label_name, color=color_name)
      if index == 8 or index == 9:
         
         kt = -0.005
         label_name = 'half'
         color_name = 'indianred'
         l2 = ax[ax_i].errorbar(k[index]+k[index]*kt, k[index] * xs_mean[index] /TF(k[index]), k[index] * xs_sigma[index] / TF(k[index]), fmt='o', label=label_name, color=color_name)
      if index == 10 or index == 11:
         
         kt = 0.005
         label_name = 'odde'
         color_name = 'purple'
         l3 = ax[ax_i].errorbar(k[index]+k[index]*kt, k[index] * xs_mean[index] / TF(k[index]), k[index] * xs_sigma[index] / TF(k[index]), fmt='o', label=label_name, color=color_name)
      if index == 14 or index == 15:
         kt = 0.015
         label_name = 'dayn'
         color_name = 'forestgreen'
       
         l4 = ax[ax_i].errorbar(k[index]+k[index]*kt, k[index] * xs_mean[index] / TF(k[index]), k[index] * xs_sigma[index] / TF(k[index]), fmt='o', label=label_name, color=color_name)
      if index == 0 or index == 1:
         kt = 0.02
         label_name = 'ambt'
         color_name = 'royalblue'
       
         l5 = ax[ax_i].errorbar(k[index]+k[index]*kt, k[index] * xs_mean[index] / TF(k[index]), k[index] * xs_sigma[index] / TF(k[index]), fmt='o', label=label_name, color=color_name)
      if index == 2 or index == 3:
         kt = -0.02
         label_name = 'wind'
         color_name = 'gold'
       
         l6 = ax[ax_i].errorbar(k[index]+k[index]*kt, k[index] * xs_mean[index] / TF(k[index]), k[index] * xs_sigma[index] / TF(k[index]), fmt='o', label=label_name, color=color_name)


   
   ax[ax_i].set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]', fontsize=22)

   if cesc == '1':
      ax[ax_i].set_ylim(-lim*6, lim*6)          
   if cesc == '0':
      ax[ax_i].set_ylim(-lim*6, lim*6)  
   
   if field == 'CO2':
      #ax[ax_i].xaxis.set_label_position('top')
      #ax[ax_i].xaxis.tick_top()
      if cesc == '0':
         ax[ax_i].set_title('Lissajous scans', fontsize=22, pad=50)
      if cesc == '1':
         ax[ax_i].set_title('CES scans', fontsize=22, pad=50)  
   ax[ax_i].text(.5,.9,field,horizontalalignment='center',transform=ax[ax_i].transAxes, fontsize=22)     
   ax[ax_i].set_xlim(0.04,0.7)
   ax[ax_i].set_xscale('log')
   #ax[ax_i].set_title(field, fontsize=16)
   ax[ax_i].grid()
   ax[ax_i].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=22)
   labnums = [0.05,0.1, 0.2, 0.5]
   ax[ax_i].tick_params(labelsize=22)
   ax[ax_i].set_xticks(labnums)
   ax[ax_i].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   #plt.legend(bbox_to_anchor=(0, 0.61))
   #ax[ax_i].legend(ncol=4)
   return l1,l2,l3,l4, l5, l6

def plot_nulltest(cesc):
   k7, xs_mean7, xs_sigma7 = read_h5_arrays('co7_map_null_1D_arrays.h5') 
   xs_mean7 = xs_mean7[8]
   k7 = k7[8]
   lim = np.mean(np.abs(xs_mean7[4:-2] * k7[4:-2])) * 8
   if cesc == '0':
      TF = TF_liss_1D
      jk_we_want = [4,8,10,14,0,2] #indices of jk we want to use: wint, half, odde, dayn, ambt, wind
      scan = 'Lissajous'
   if cesc == '1':
      jk_we_want = [5,9,11,15,1,3]
      TF = TF_CES_1D
      scan = 'CES'


   fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(11,19))
   
  
   l1,l2,l3,l4, l5, l6 = plot_sub_fig('CO2',jk_we_want,0,lim,cesc,ax, TF,scan)
  
   l1,l2,l3,l4, l5, l6 = plot_sub_fig('CO6',jk_we_want,1,lim,cesc,ax, TF, scan)
  
   l1,l2,l3,l4, l5, l6 = plot_sub_fig('CO7',jk_we_want,2,lim,cesc,ax, TF, scan)
   plt.figlegend((l1,l2,l3,l4, l5, l6), ('Winter/Summer', 'Half-mission', 'Odd/Even ObsID', 'Day/Night', 'Ambient temperature', 'Wind speed'),loc='upper center',bbox_to_anchor=(0.52,0.987), ncol=6, fontsize=24)
   plt.tight_layout()
   if cesc == '0':
      #plt.title('Lissajous scans', fontsize=16, loc='right')
      plt.savefig('nl.png', bbox_inches='tight')
   if cesc == '1':
      #plt.title('CES scans', fontsize=16, loc='right')
      plt.savefig('nc.png', bbox_inches='tight')
   

#plot_nulltest('0')
#plot_nulltest('1')


#now plot one of these in 2D

def log2lin(x, k_edges):
    loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
    logx = np.log10(x) - np.log10(k_edges[0])
    return logx / loglen

def xs_2D_plot_null(index_liss, index_ces, figure_name):
      fields = ['co2', 'co6', 'co7']
      k_c2, xs_mean_c2, xs_sigma_c2, k_edges_perp_c2, k_edges_par_c2 = read_h5_arrays(fields[0] + '_map_null_2D_arrays.h5', two_dim=True)
      k_c6, xs_mean_c6, xs_sigma_c6, k_edges_perp_c6, k_edges_par_c6 = read_h5_arrays(fields[1] + '_map_null_2D_arrays.h5', two_dim=True)
      k_c7, xs_mean_c7, xs_sigma_c7, k_edges_perp_c7, k_edges_par_c7 = read_h5_arrays(fields[2] + '_map_null_2D_arrays.h5', two_dim=True)
      k = k_c2[0] #these are all the same
      k_bin_edges_perp = k_edges_perp_c2[0] #these are all the same
      k_bin_edges_par = k_edges_par_c2[0] #these are all the same
      xs_mean1 = xs_mean_c2[index_liss]   
      xs_mean2 = xs_mean_c6[index_liss]
      xs_mean3 = xs_mean_c7[index_liss]
      xs_mean4 = xs_mean_c2[index_ces]   
      xs_mean5 = xs_mean_c6[index_ces]
      xs_mean6 = xs_mean_c7[index_ces]

      fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(16,16))
      #fig.tight_layout(h_pad=0.005, w_pad=1)
      fig.subplots_adjust(hspace=-0.5, wspace=0.0)
      #fig.suptitle(titlename, fontsize=16)
      #norm = mpl.colors.Normalize(vmin=1.3*np.amin(xs_mean7), vmax=-1.3*np.amin(xs_mean7))  
      #norm1 = mpl.colors.Normalize(vmin=1.3*np.amin(xs_mean7/xs_sigma7), vmax=-1.3*np.amin(xs_mean7/xs_sigma7)) 
      norm = mpl.colors.Normalize(vmin=-3e6, vmax=3e6)  #here it was 800000
      norm1 = mpl.colors.Normalize(vmin=-5, vmax=5) 

    
      img1 = ax[0][0].imshow(xs_mean1/TF_liss_2D(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img1, ax=ax[0][0],fraction=0.046, pad=0.1, orientation='horizontal').set_label(r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$], Liss', size=18)
  
      img2 = ax[0][1].imshow(xs_mean2/TF_liss_2D(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img2, ax=ax[0][1], fraction=0.046, pad=0.1, orientation='horizontal').set_label(r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$], Liss', size=18)
      img3 = ax[0][2].imshow(xs_mean3/TF_liss_2D(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img3, ax=ax[0][2], fraction=0.046, pad=0.1, orientation='horizontal').set_label(r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$], Liss', size=18)
      

 
      img4 = ax[1][0].imshow(xs_mean4/TF_CES_2D(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img4, ax=ax[1][0],fraction=0.046, pad=0.1, orientation='horizontal').set_label(r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$], CES', size=18)
      
  
      img5 = ax[1][1].imshow(xs_mean5/TF_CES_2D(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img5, ax=ax[1][1], fraction=0.046, pad=0.1, orientation='horizontal').set_label(r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$], CES', size=18)
      
      img6 = ax[1][2].imshow(xs_mean6/TF_CES_2D(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img6, ax=ax[1][2], fraction=0.046, pad=0.1, orientation='horizontal').set_label(r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$], CES', size=18)
      
     
     
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
      
      ax[0][0].set_title('CO2', fontsize=20)
      ax[0][1].set_title('CO6', fontsize=20)
      ax[0][2].set_title('CO7', fontsize=20)

      for i in range(3):
         for j in range(2):
            ax[j][i].set_xticks(ticklist_x, minor=True)
            ax[j][i].set_xticks(majorlist_x, minor=False)
            ax[j][i].set_xticklabels(majorlabels, minor=False, fontsize=16)
            ax[j][i].set_yticks(ticklist_y, minor=True)
            ax[j][i].set_yticks(majorlist_y, minor=False)
            ax[j][i].set_yticklabels(majorlabels, minor=False, fontsize=16)
            ax[j][i].tick_params(labelsize=16)
            ax[j][i].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]', fontsize=18)
      
      #ax[1][0].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]',fontsize=18)
      ax[0][0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}$]',fontsize=18)
      ax[1][0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}$]',fontsize=18)
      #ax[1][1].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]', fontsize=18)
      #ax[1][2].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]', fontsize=18)
      
      plt.tight_layout()
      plt.savefig(figure_name) 


#xs_2D_plot_null(14,15, 'dayn_2d.png') #dayn


#Main results  -  mean from FPXS

#def xs_2D_plot(figure_name, k,k_bin_edges_par, k_bin_edges_perp, xs_mean2,xs_mean6,xs_mean7, xs_sigma2,xs_sigma6,xs_sigma7, titlename):
def xs_2D_plot(figure_name, index, scan_type):
      fields = ['co2', 'co6', 'co7']
      k_c2, xs_mean_c2, xs_sigma_c2, k_edges_perp_c2, k_edges_par_c2 = read_h5_arrays(fields[0] + '_map_signal_2D_arrays.h5', two_dim=True)
      k_c6, xs_mean_c6, xs_sigma_c6, k_edges_perp_c6, k_edges_par_c6 = read_h5_arrays(fields[1] + '_map_signal_2D_arrays.h5', two_dim=True)
      k_c7, xs_mean_c7, xs_sigma_c7, k_edges_perp_c7, k_edges_par_c7 = read_h5_arrays(fields[2] + '_map_signal_2D_arrays.h5', two_dim=True)
      k = k_c2[0] #these are all the same
      k_bin_edges_perp = k_edges_perp_c2[0] #these are all the same
      k_bin_edges_par = k_edges_par_c2[0] #these are all the same
      xs_mean2 = xs_mean_c2[index]   
      xs_mean6 = xs_mean_c6[index]
      xs_mean7 = xs_mean_c7[index]
      xs_sigma2 = xs_sigma_c2[index]   
      xs_sigma6 = xs_sigma_c6[index]
      xs_sigma7 = xs_sigma_c7[index]
      if scan_type == 'liss':
         TF = TF_liss_2D
         first_label = r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$]'
         second_label = r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)/\sigma_{\tilde{C}}$'
      if scan_type == 'CES':
         TF = TF_CES_2D
         first_label = r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$]'
         second_label = r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)/\sigma_{\tilde{C}}$'


      




      
      fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(14,8))
   
      fig.subplots_adjust(hspace=-0.5, wspace=0.0)

     
     
      norm = mpl.colors.Normalize(vmin=-800000, vmax=800000)  #here it was 800000
      norm1 = mpl.colors.Normalize(vmin=-5, vmax=5) 

    
      img1 = ax[0][0].imshow(xs_mean2/TF(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      #fig.colorbar(img1, ax=ax[0][0],fraction=0.046, pad=0.04, ticks=[-800000, -400000, 0, 400000, 800000])
  
      img2 = ax[0][1].imshow(xs_mean6/TF(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      #fig.colorbar(img2, ax=ax[0][1], fraction=0.046, pad=0.04,  ticks=[-800000, -400000, 0, 400000, 800000])
      img3 = ax[0][2].imshow(xs_mean7/TF(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img3, ax=ax[0][2], fraction=0.046, pad=0.04,  ticks=[-800000, -400000, 0, 400000, 800000]).set_label(first_label, size=18)
     
      img4 = ax[1][0].imshow(xs_mean2/xs_sigma2, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm1)
      #fig.colorbar(img4, ax=ax[1][0],fraction=0.046, pad=0.04)
  
      img5 = ax[1][1].imshow(xs_mean6/xs_sigma6, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm1)
      #fig.colorbar(img5, ax=ax[1][1], fraction=0.046, pad=0.04)
      img6 = ax[1][2].imshow(xs_mean7/xs_sigma7, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm1)
      fig.colorbar(img6, ax=ax[1][2], fraction=0.046,pad=0.04).set_label(second_label, size=18)
      
     
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
      
      ax[0][0].set_title(r'CO2', fontsize=20)
      ax[0][1].set_title(r'CO6', fontsize=20)
      ax[0][2].set_title(r'CO7', fontsize=20)

      for i in range(3):
         for j in range(2):
            ax[j][i].set_xticks(ticklist_x, minor=True)
            ax[j][i].set_xticks(majorlist_x, minor=False)
            ax[j][i].set_xticklabels(majorlabels, minor=False, fontsize=16)
            ax[j][i].set_yticks(ticklist_y, minor=True)
            ax[j][i].set_yticks(majorlist_y, minor=False)
            ax[j][i].set_yticklabels(majorlabels, minor=False, fontsize=16)
            ax[j][i].tick_params(labelsize=16)
            ax[j][i].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]', fontsize=18)
      
      #ax[1][0].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]',fontsize=14)
      ax[0][0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}$]',fontsize=18)
      ax[1][0].set_ylabel(r'$k_{\bot}$ [Mpc${}^{-1}$]',fontsize=18)
      #ax[1][1].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]', fontsize=14)
      #ax[1][2].set_xlabel(r'$k_{\parallel}$ [Mpc${}^{-1}$]', fontsize=14)
      
      plt.tight_layout()
      plt.savefig(figure_name) 


xs_2D_plot('liss_2d.png', 0, 'liss')
xs_2D_plot('ces_2d.png', 1, 'CES')


#combine all CES data
def coadd_all_ces():
   k2, xs_mean2, xs_sigma2 = read_h5_arrays('co2_map_signal_1D_arrays.h5')
   k6, xs_mean6, xs_sigma6 = read_h5_arrays('co6_map_signal_1D_arrays.h5')
   k7, xs_mean7, xs_sigma7 = read_h5_arrays('co7_map_signal_1D_arrays.h5')
   k2, xs2, sigma2 = k2[1], xs_mean2[1], xs_sigma2[1] #take CES
   k6, xs6, sigma6 = k6[1], xs_mean6[1], xs_sigma6[1] #take CES
   k7, xs7, sigma7 = k7[1], xs_mean7[1], xs_sigma7[1] #take CES
   xs_sigma_arr = np.array([sigma2, sigma6, sigma7])
   xs_mean_arr = np.array([xs2,xs6,xs7])
   k2 = np.array(k2)
   no_k = len(k2)
   mean_combined = np.zeros(no_k)
   w_sum = np.zeros(no_k)
   
   for i in range(3): 
      w = 1./ xs_sigma_arr[i]**2.
      w_sum += w
      mean_combined += w*xs_mean_arr[i]
   mean_combined1 = mean_combined/w_sum
   sigma_combined1 = w_sum**(-0.5)

   mean_combined = mean_combined1/TF_CES_1D(k2)
   sigma_combined = sigma_combined1/TF_CES_1D(k2)

   return mean_combined, sigma_combined, k2

def coadd_CO7():
   k7, xs_mean7, xs_sigma7 = read_h5_arrays('co7_map_signal_1D_arrays.h5')
   k_liss, xs_liss, sigma_liss = k7[0], xs_mean7[0], xs_sigma7[0] #take Liss
   k_ces, xs_ces, sigma_ces = k7[1], xs_mean7[1], xs_sigma7[1] #take CES
   xs_sigma_arr = np.array([sigma_liss, sigma_ces])
   xs_mean_arr = np.array([xs_liss, xs_ces])
   k_liss = np.array(k_liss)
   no_k = len(k_liss)
   mean_combined = np.zeros(no_k)
   w_sum = np.zeros(no_k)
   
   for i in range(2): 
      w = 1./ xs_sigma_arr[i]**2.
      w_sum += w
      mean_combined += w*xs_mean_arr[i]
   mean_combined1 = mean_combined/w_sum
   sigma_combined1 = w_sum**(-0.5)

   mean_combined = mean_combined1/TF_mix_1D(k_liss)
   sigma_combined = sigma_combined1/TF_mix_1D(k_liss)

   return mean_combined, sigma_combined, k_liss

def coadd_ces_CO7liss():
   k2, xs_mean2, xs_sigma2 = read_h5_arrays('co2_map_signal_1D_arrays.h5')
   k6, xs_mean6, xs_sigma6 = read_h5_arrays('co6_map_signal_1D_arrays.h5')
   k7, xs_mean7, xs_sigma7 = read_h5_arrays('co7_map_signal_1D_arrays.h5')
   k2, xs2, sigma2 = k2[1], xs_mean2[1], xs_sigma2[1] #take CES
   k6, xs6, sigma6 = k6[1], xs_mean6[1], xs_sigma6[1] #take CES
   k7, xs7, sigma7 = k7[1], xs_mean7[1], xs_sigma7[1] #take CES
   k_liss, xs_liss, sigma_liss = k7[0], xs_mean7[0], xs_sigma7[0] #take CO7 Liss
   xs_sigma_arr = np.array([sigma2, sigma6, sigma7, sigma_liss])
   xs_mean_arr = np.array([xs2, xs6,xs7, xs_liss])
   k2 = np.array(k2)
   no_k = len(k2)
   mean_combined = np.zeros(no_k)
   w_sum = np.zeros(no_k)
   
   for i in range(4): 
      w = 1./ xs_sigma_arr[i]**2.
      w_sum += w
      mean_combined += w*xs_mean_arr[i]
   mean_combined1 = mean_combined/w_sum
   sigma_combined1 = w_sum**(-0.5)

   mean_combined = mean_combined1/TF_CES_1D(k2)
   sigma_combined = sigma_combined1/TF_CES_1D(k2)

   return mean_combined, sigma_combined, k2

def coadd_ces67_liss7():
   k2, xs_mean2, xs_sigma2 = read_h5_arrays('co2_map_signal_1D_arrays.h5')
   k6, xs_mean6, xs_sigma6 = read_h5_arrays('co6_map_signal_1D_arrays.h5')
   k7, xs_mean7, xs_sigma7 = read_h5_arrays('co7_map_signal_1D_arrays.h5')
   k2, xs2, sigma2 = k2[1], xs_mean2[1], xs_sigma2[1] #take CES
   k6, xs6, sigma6 = k6[1], xs_mean6[1], xs_sigma6[1] #take CES
   k7, xs7, sigma7 = k7[1], xs_mean7[1], xs_sigma7[1] #take CES
   k_liss, xs_liss, sigma_liss = k7[0], xs_mean7[0], xs_sigma7[0] #take CO7 Liss
   xs_sigma_arr = np.array([sigma6, sigma7, sigma_liss])
   xs_mean_arr = np.array([xs6,xs7, xs_liss])
   k2 = np.array(k2)
   no_k = len(k2)
   mean_combined = np.zeros(no_k)
   w_sum = np.zeros(no_k)
   
   for i in range(3): 
      w = 1./ xs_sigma_arr[i]**2.
      w_sum += w
      mean_combined += w*xs_mean_arr[i]
   mean_combined1 = mean_combined/w_sum
   sigma_combined1 = w_sum**(-0.5)

   mean_combined = mean_combined1/TF_CES_1D(k2)
   sigma_combined = sigma_combined1/TF_CES_1D(k2)

   return mean_combined, sigma_combined, k2


def xs_1D_3fields(figure_name, scan_strategy, index):  
   k2, xs_mean2, xs_sigma2 = read_h5_arrays('co2_map_signal_1D_arrays.h5')
   k6, xs_mean6, xs_sigma6 = read_h5_arrays('co6_map_signal_1D_arrays.h5')
   k7, xs_mean7, xs_sigma7 = read_h5_arrays('co7_map_signal_1D_arrays.h5')
   k = k2[0] 
   xs_mean2 = xs_mean2[index]
   xs_mean6 = xs_mean6[index]
   xs_mean7 = xs_mean7[index]
   xs_sigma2 = xs_sigma2[index]
   xs_sigma6 = xs_sigma6[index]
   xs_sigma7 = xs_sigma7[index]


   if scan_strategy == 'ces':
      titlename = 'CES'
      
      TF = TF_CES_1D
   if scan_strategy == 'liss':
      titlename = 'Lissajous scans'
      TF = TF_liss_1D
   
   k_offset = k*0.025
   k6 = k - k_offset
   k7 = k + k_offset
   k_combo = k + k_offset*2
   lim = np.mean(np.abs(xs_mean2[4:-2] * k[4:-2])) * 8
   fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10.5,8))
   #fig.set_figwidth(8)
   ax[0].set_title(titlename, fontsize=22, pad=44)
   #mean_combo, sigma_combo, k_combo = coadd_all_ces()
   ax[0].errorbar(k6, k * xs_mean6 / TF(k), k * xs_sigma6 / TF(k), fmt='o', label=r'CO6', color='teal', zorder=3)
   ax[0].errorbar(k7, k * xs_mean7 / TF(k), k * xs_sigma7 / TF(k), fmt='o', label=r'CO7', color='purple', zorder=2)
   ax[0].errorbar(k, k * xs_mean2 / TF(k), k * xs_sigma2 / TF(k), fmt='o', label=r'CO2', color='indianred', zorder=4)
   #ax[0].errorbar(k_combo, k_combo * mean_combo, k_combo * sigma_combo , fmt='o', label=r'combo', color='black', zorder=4)
   #ax1.errorbar(k_combo, k * mean_combo / (transfer(k)*transfer_filt(k)), k * sigma_combo / (transfer(k)*transfer_filt(k)), fmt='o', label=r'combo', color='black', zorder=5)
   #ax1.errorbar(k, k * xs_mean, k * xs_sigma, fmt='o', label=r'$k\tilde{C}_{data}(k)$')
   ax[0].plot(k, 0 * xs_mean2, 'k', alpha=0.4, zorder=1)
   #ax1.plot(k, k*PS_function.PS_f(k)/ transfer(k), label='k*PS of the input signal')
   #ax1.plot(k, k*PS_function.PS_f(k), label='k*PS of the input signal')
   #ax1.plot(k_th, k_th * ps_th_nobeam * 10, '--', label=r'$10\times kP_{Theory}(k)$', color='dodgerblue')
   #ax1.plot(k_th, k_th * ps_copps_nobeam * 5, 'g--', label=r'$5 \times kP_{COPPS}$ (shot)')
   ax[0].set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]', fontsize=18)
   if scan_strategy == 'ces':
      ax[0].set_ylim(-lim*3, lim*3)              # ax1.set_ylim(0, 0.1)
   if scan_strategy == 'liss':
      ax[0].set_ylim(-lim, lim)              # ax1.set_ylim(0, 0.1)
   ax[0].set_xlim(0.04,0.7)
   ax[0].set_xscale('log')
   #ax1.set_title(titlename, fontsize=16)
   ax[0].grid()
   #ax1.set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   labnums = [0.05,0.1, 0.2, 0.5]
   ax[0].set_xticks(labnums)
   ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   #plt.legend(bbox_to_anchor=(0, 0.61))
   ax[0].legend(ncol=4, fontsize=18, loc='upper center',bbox_to_anchor=(0.5,1.22))
   
   
   #ax2.plot(k, diff_mean / error, fmt='o', label=r'$\tilde{C}_{diff}(k)$', color='black')
   
   ax[1].errorbar(k6, xs_mean6 / xs_sigma6, xs_sigma6/xs_sigma6, fmt='o', label=r'CO6', color='teal', zorder=3)
   ax[1].errorbar(k7, xs_mean7 / xs_sigma7, xs_sigma7/xs_sigma7, fmt='o', label=r'CO7', color='purple', zorder=2)
   ax[1].errorbar(k, xs_mean2 / xs_sigma2, xs_sigma2/xs_sigma2, fmt='o', label=r'CO2', color='indianred', zorder=4)
   #ax2.errorbar(k_combo, mean_combo / sigma_combo, sigma_combo/sigma_combo, fmt='o', label=r'combo', color='black', zorder=5)
   #ax2.errorbar(k, sum_mean / error, error /error, fmt='o', label=r'$\tilde{C}_{sum}(k)$', color='mediumorchid')
   ax[1].plot(k, 0 * xs_mean2, 'k', alpha=0.4, zorder=1)
   #ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
   ax[1].set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$', fontsize=18)
   ax[1].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   ax[1].set_ylim(-5, 5)
   ax[1].set_xlim(0.04,0.7)
   ax[1].set_xscale('log')
   ax[1].grid()
   #ax[1].legend(ncol=4, fontsize=18)
   ax[1].set_xticks(labnums)
   ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   ax[1].tick_params(labelsize=16)
   ax[0].tick_params(labelsize=16)
   plt.tight_layout()
   #plt.legend()
   plt.savefig(figure_name, bbox_inches='tight')
   #plt.show()



xs_1D_3fields('liss_1d.png', 'liss', 0)
xs_1D_3fields('ces_1d.png', 'ces', 1)

 
 


def plot_combined_and_model(figure_name):
   xs_data, sigma_data, k = coadd_all_ces()
   P_theory_new = np.load('ps_theory_new_1D.npy')
   P_theory_new = 1e-12*np.mean(P_theory_new, axis=0) #this factor accounts for the fact that I wrongly converted units in the simulated maps
   k_th = np.load('k.npy')
   P_theory_old = np.load('psn.npy')

   beam_ps_original_1D = np.load('transfer_functions/' + 'ps_original_1D_newest.npy')
   P_notsmooth = 1e-12*np.mean(beam_ps_original_1D, axis=0)
   lim = np.mean(np.abs(xs_data[4:-2] * k[4:-2])) * 8
   fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,8))
   ax[0].errorbar(k, k * xs_data, k * sigma_data, fmt='o', label=r'$k\tilde{C}(k)$, all CES', color='black', zorder=4)
   ax[0].plot(k_th, k_th * P_theory_old * 10, '--', label=r'$10kP_{Theory}(k)$', color='teal', zorder=3)
   #ax.plot(k_th, P_theory_old, '--', label=r'$\times P_{Theory, old}(k)$', color='dodgerblue')
   ax[0].plot(k, k * P_theory_new  * 10, label=r'$10k\tilde{P}_{Theory, \parallel smooth}(k)$', color='purple') #smoothed in z-direction
   ax[0].plot(k, k * P_notsmooth  * 10, label=r'$10k\tilde{P}_{Theory}(k)$', color='salmon') #not smoothed
   #ax.set_ylim(-lim*3, lim*3) 
   ax[0].set_ylim(-40000, 30000) 
   ax[0].plot(k, 0 * xs_data, 'k', alpha=0.4, zorder=1)
   ax[0].set_ylabel(r'[$\mu$K${}^2$ Mpc${}^2$]', fontsize=18)
   ax[0].legend(ncol=2, fontsize=16, loc='upper center',bbox_to_anchor=(0.5,1.42))
   ax[0].set_xlim(0.04,0.7)
   ax[0].set_xscale('log')
   #ax[0].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   #ax.set_yscale('log')
   ax[0].grid()
   labnums = [0.05,0.1, 0.2, 0.5]
   ax[0].set_xticks(labnums)
   ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   ax[0].tick_params(labelsize=15)
   ax[1].errorbar(k, xs_data / sigma_data, sigma_data/sigma_data, fmt='o', color='black', zorder=4)
   #ax2.errorbar(k_combo, mean_combo / sigma_combo, sigma_combo/sigma_combo, fmt='o', label=r'combo', color='black', zorder=5)
   #ax2.errorbar(k, sum_mean / error, error /error, fmt='o', label=r'$\tilde{C}_{sum}(k)$', color='mediumorchid')
   ax[1].plot(k, 0 * xs_data, 'k', alpha=0.4, zorder=1)
   #ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
   ax[1].set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$', fontsize=18)
   ax[1].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   ax[1].set_ylim(-4, 4)
   ax[1].set_xlim(0.04,0.7)
   ax[1].set_xscale('log')
   ax[1].grid()
   #ax[1].legend(ncol=4, fontsize=18)
   ax[1].set_xticks(labnums)
   ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   ax[1].tick_params(labelsize=15)


   plt.tight_layout()
   plt.savefig(figure_name, bbox_inches='tight')


plot_combined_and_model('theory_ces267.png')

def calculate_A1(k, xs_mean, xs_sigma):
   #mean and sigma already divded by TF
   PS_estimate = 0
   w_sum = 0
   no_of_k = len(k)
   for i in range(5,no_of_k-3): #we exclude 4 first points and 3 last points, previously excluded 2 first, i excluded one more extra now
      print ('k used in estimate:', k[i])
      w = 1./ xs_sigma[i]**2.
      w_sum += w
      PS_estimate += w*xs_mean[i]
   PS_estimate = PS_estimate/w_sum
   PS_error = w_sum**(-0.5)
   return PS_estimate, PS_error

def calculate_A2(k, xs_mean, xs_sigma, P_theory):
   PS_estimate = 0
   w_sum = 0
   no_of_k = len(k)
   #P_theory = scipy.interpolate.interp1d(k_th,ps_th_nobeam)
   xs_mean = xs_mean/P_theory(k)
   xs_sigma = xs_sigma/P_theory(k)
   for i in range(5,no_of_k-3): #we exclude 4 first points and 3 last points, previously excluded 2 first, i excluded one more extra now
      w = 1./ xs_sigma[i]**2.
      w_sum += w
      PS_estimate += w*xs_mean[i]
   PS_estimate = PS_estimate/w_sum
   PS_error = w_sum**(-0.5)
   return PS_estimate, PS_error

def plot_estimates(figure_name):
   xs_data, sigma_data, k = coadd_all_ces()
   P_theory_new = np.load('ps_theory_new_1D.npy')
   P_theory_new = 1e-12*np.mean(P_theory_new, axis=0) #this factor accounts for the fact that I wrongly converted units in the simulated maps
   P_theory_new_func = scipy.interpolate.interp1d(k,P_theory_new)
   #beam_ps_original_1D = np.load('transfer_functions/' + 'ps_original_1D_newest.npy')
   #P_notsmooth = 1e-12*np.mean(beam_ps_original_1D, axis=0)
   #P_notsmooth_func = scipy.interpolate.interp1d(k, P_notsmooth)


   
   A1, A1_error = calculate_A1(k, xs_data, sigma_data)
   A2, A2_error = calculate_A2(k, xs_data, sigma_data, P_theory_new_func)
   print ('A1:', A1, A1_error)
   print ('A2:', A2, A2_error)
  
   lim = np.mean(np.abs(xs_data[4:-2] * k[4:-2])) * 8
   fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,8))
   ax[0].errorbar(k[5:-3], k[5:-3] * xs_data[5:-3], k[5:-3] * sigma_data[5:-3], fmt='o', label=r'$k\tilde{C}(k)$, all CES', color='black', zorder=4)
   ax[0].plot(k, k*A1, label=r'$A_1k$', color='midnightblue')
   ax[0].fill_between(x=k, y1=k*A1-k*A1_error, y2=k*A1+k*A1_error, facecolor='lightsteelblue', edgecolor='lightsteelblue')
   ax[0].plot(k, k * P_theory_new  * 10, label=r'$10k\tilde{P}_{Theory, \parallel smooth}(k)$', color='purple') #smoothed in z-direction

   #ax.set_ylim(-lim*3, lim*3) 
   ax[0].set_ylim(-40000, 30000) 
   ax[1].set_ylim(-40000, 30000)
   ax[0].plot(k, 0 * xs_data, 'k', alpha=0.4, zorder=1)
   ax[0].set_ylabel(r'[$\mu$K${}^2$ Mpc${}^2$]', fontsize=18)
   ax[1].set_ylabel(r'[$\mu$K${}^2$ Mpc${}^2$]', fontsize=18)
   ax[0].legend(ncol=3, fontsize=14, loc='upper center', bbox_to_anchor=(0.45,1.21))
   ax[0].set_xlim(0.04,0.7)
   ax[0].set_xscale('log')
   #ax[0].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   #ax.set_yscale('log')
   ax[0].grid()
   labnums = [0.05,0.1, 0.2, 0.5]
   ax[0].set_xticks(labnums)
   ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   ax[0].tick_params(labelsize=15)
   
   #ax2.errorbar(k_combo, mean_combo / sigma_combo, sigma_combo/sigma_combo, fmt='o', label=r'combo', color='black', zorder=5)
   #ax2.errorbar(k, sum_mean / error, error /error, fmt='o', label=r'$\tilde{C}_{sum}(k)$', color='mediumorchid')
   ax[1].plot(k, 0 * xs_data, 'k', alpha=0.4, zorder=1)
   #ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
   ax[1].errorbar(k[5:-3], k[5:-3] * xs_data[5:-3], k[5:-3] * sigma_data[5:-3], fmt='o', label=r'$k\tilde{C}(k)$, all CES', color='black', zorder=4)
   
   ax[1].plot(k, k*A2*P_theory_new_func(k), label=r'$A_2k\tilde{P}_{Theory, \parallel smooth}(k)$', color='midnightblue')
   ax[1].fill_between(x=k, y1=k*A2*P_theory_new_func(k)-k*A2_error*P_theory_new_func(k), y2=k*A2*P_theory_new_func(k)+k*A2_error*P_theory_new_func(k), facecolor='lightsteelblue', edgecolor='lightsteelblue')
   ax[1].plot(k, k * P_theory_new  * 10, label=r'$10k\tilde{P}_{Theory, \parallel smooth}(k)$', color='purple') #smoothed in z-direction
   ax[1].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=18)
   
   ax[1].set_xlim(0.04,0.7)
   ax[1].set_xscale('log')
   ax[1].grid()
   ax[1].legend(ncol=3, fontsize=14, loc='upper center', bbox_to_anchor=(0.5,1.21))
   ax[1].set_xticks(labnums)
   ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   ax[1].tick_params(labelsize=15)


   plt.tight_layout()
   plt.savefig(figure_name, bbox_inches='tight')

plot_estimates('fits_ces267.png')

'''
comment- excluding CES CO2 made the error bars larger, but at least we have an estimate that is positive (A2), CES CO2 makes whole data more negatively biased so we don't even enclose theory spectrum if we use it -- i tried CO7 Liss + CO7 CES, all CES, all CES + CO7 Liss, and finally CES CO6 + CES CO7 + Liss CO7 with the result: 
A1: -33289.69798458218 15603.74388984571
A2: -3.400396906629835 12.189530628784624

--also I'm using only 6 k-bins in these estimates (the ones that are included on the amplitude plot)
RESULTS:
theory_ces67_liss7.png, fits_ces67_liss7.png
A1: -33289.69798458218 15603.74388984571
A2: -3.400396906629835 12.189530628784624

theory_ces7_liss7.png, fits_ces7_liss7.png
A1: -35130.226481494945 16696.823778928218
A2: -7.069335042177033 12.15553618129899

theory_ces267_liss7.png, fits_ces267_liss7.png
A1: -27456.564938533178 11818.573308872701
A2: -10.286665525112525 8.955355370420415

theory_ces267.png, fits_ces267.png
A1: -26365.787450647218 13798.994374516007
A2: -12.795491218975663 10.305357921662255

maybe make a table with these for the thesis?
'''

