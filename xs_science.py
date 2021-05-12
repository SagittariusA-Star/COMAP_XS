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
         ax[ax_i].set_title('Lissajous scans', fontsize=22, pad=50
)
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
         first_label = r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$], Liss'
         second_label = r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)/\sigma_{\tilde{C}}$, Liss'
      if scan_type == 'CES':
         TF = TF_CES_2D
         first_label = r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$], CES'
         second_label = r'$\tilde{C}\left(k_{\bot},k_{\parallel}\right)/\sigma_{\tilde{C}}$, CES'


      




      
      fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(16,16))
   
      fig.subplots_adjust(hspace=-0.5, wspace=0.0)

     
     
      norm = mpl.colors.Normalize(vmin=-800000, vmax=800000)  #here it was 800000
      norm1 = mpl.colors.Normalize(vmin=-5, vmax=5) 

    
      img1 = ax[0][0].imshow(xs_mean2/TF(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img1, ax=ax[0][0],fraction=0.046, pad=0.1, orientation='horizontal').set_label(first_label, size=18)
  
      img2 = ax[0][1].imshow(xs_mean6/TF(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img2, ax=ax[0][1], fraction=0.046, pad=0.1, orientation='horizontal').set_label(first_label, size=18)
      img3 = ax[0][2].imshow(xs_mean7/TF(k[0],k[1]), interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm)
      fig.colorbar(img3, ax=ax[0][2], fraction=0.046, pad=0.1, orientation='horizontal').set_label(first_label, size=18)
     
      img4 = ax[1][0].imshow(xs_mean2/xs_sigma2, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm1)
      fig.colorbar(img4, ax=ax[1][0],fraction=0.046, pad=0.1, orientation='horizontal').set_label(second_label, size=18)
  
      img5 = ax[1][1].imshow(xs_mean6/xs_sigma6, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm1)
      fig.colorbar(img5, ax=ax[1][1], fraction=0.046, pad=0.1, orientation='horizontal').set_label(second_label, size=18)
      img6 = ax[1][2].imshow(xs_mean7/xs_sigma7, interpolation='none', origin='lower',extent=[0,1,0,1], cmap='magma', norm=norm1)
      fig.colorbar(img6, ax=ax[1][2], fraction=0.046,pad=0.1, orientation='horizontal').set_label(second_label, size=18)
      
     
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








