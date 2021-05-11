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


   if ax_i == 0:
      ax[ax_i].set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]', fontsize=14)

   if cesc == '1':
      ax[ax_i].set_ylim(-lim*6, lim*6)          
   if cesc == '0':
      ax[ax_i].set_ylim(-lim*6, lim*6)  
   
   if field == 'CO2':
      #ax[ax_i].xaxis.set_label_position('top')
      #ax[ax_i].xaxis.tick_top()
      if cesc == '0':
         ax[ax_i].set_title('Lissajous scans', fontsize=16, pad=40)
      if cesc == '1':
         ax[ax_i].set_title('CES scans', fontsize=16, pad=40)  
   ax[ax_i].text(.5,.9,field,horizontalalignment='center',transform=ax[ax_i].transAxes, fontsize=16)     
   ax[ax_i].set_xlim(0.04,0.7)
   ax[ax_i].set_xscale('log')
   #ax[ax_i].set_title(field, fontsize=16)
   ax[ax_i].grid()
   ax[ax_i].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=16)
   labnums = [0.05,0.1, 0.2, 0.5]
   ax[ax_i].tick_params(labelsize=16)
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


   fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(9,17))
   
  
   l1,l2,l3,l4, l5, l6 = plot_sub_fig('CO2',jk_we_want,0,lim,cesc,ax, TF,scan)
  
   l1,l2,l3,l4, l5, l6 = plot_sub_fig('CO6',jk_we_want,1,lim,cesc,ax, TF, scan)
  
   l1,l2,l3,l4, l5, l6 = plot_sub_fig('CO7',jk_we_want,2,lim,cesc,ax, TF, scan)
   plt.figlegend((l1,l2,l3,l4, l5, l6), ('Winter/Summer split', 'Half-mission split', 'Odd/Even split', 'Day/Night split', 'Ambient temperature', 'Wind speed'),loc='upper center',bbox_to_anchor=(0.52,0.6), ncol=6, fontsize=16)
   plt.tight_layout()
   if cesc == '0':
      #plt.title('Lissajous scans', fontsize=16, loc='right')
      plt.savefig('nl.png', bbox_inches='tight')
   if cesc == '1':
      #plt.title('CES scans', fontsize=16, loc='right')
      plt.savefig('nc.png', bbox_inches='tight')
   

plot_nulltest('0')
plot_nulltest('1')

'''
def plot_sub_fig2(field,jk_we_want,ax_i,lim,cesc,ax):
   if field == 'CO2':
      k, xs_mean, xs_sigma = read_h5_arrays('co2_map_null_1D_arrays.h5')
      
   if field == 'CO6':
      k, xs_mean, xs_sigma = read_h5_arrays('co6_map_null_1D_arrays.h5')
      
   if field == 'CO7':
      k, xs_mean, xs_sigma = read_h5_arrays('co7_map_null_1D_arrays.h5')
     
   ax[ax_i].plot(k[0], 0 * xs_mean[0], 'k', alpha=0.4)
   
   for index in jk_we_want:
      if index == 0 or index == 1:
         kt = -0.015
         
         label_name = 'ambt'
         color_name = 'teal'
         l1 = ax[ax_i].errorbar(k[index]+k[index]*kt, k[index] * xs_mean[index] / (transfer(k[index])*transfer_filt(k[index])), k[index] * xs_sigma[index] / (transfer(k[index])*transfer_filt(k[index])), fmt='o', label=label_name, color=color_name)
      if index == 2 or index == 3:
         
         kt = -0.005
         label_name = 'wind'
         color_name = 'indianred'
         l2 = ax[ax_i].errorbar(k[index]+k[index]*kt, k[index] * xs_mean[index] / (transfer(k[index])*transfer_filt(k[index])), k[index] * xs_sigma[index] / (transfer(k[index])*transfer_filt(k[index])), fmt='o', label=label_name, color=color_name)
      if index == 6 or index == 7:
         
         kt = 0.005
         label_name = 'rise'
         color_name = 'purple'
         l3 = ax[ax_i].errorbar(k[index]+k[index]*kt, k[index] * xs_mean[index] / (transfer(k[index])*transfer_filt(k[index])), k[index] * xs_sigma[index] / (transfer(k[index])*transfer_filt(k[index])), fmt='o', label=label_name, color=color_name)
      if index == 12 or index == 13:
         kt = 0.015
         label_name = 'fpol'
         color_name = 'forestgreen'
       
         l4 = ax[ax_i].errorbar(k[index]+k[index]*kt, k[index] * xs_mean[index] / (transfer(k[index])*transfer_filt(k[index])), k[index] * xs_sigma[index] / (transfer(k[index])*transfer_filt(k[index])), fmt='o', label=label_name, color=color_name)
   if ax_i == 0:
      ax[ax_i].set_ylabel(r'$k\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^2$]', fontsize=14)

   if cesc == '1':
      ax[ax_i].set_ylim(-lim*6, lim*6)          
   if cesc == '0':
      ax[ax_i].set_ylim(-lim*6, lim*6)  
   
   if field == 'CO6':
      #ax[ax_i].xaxis.set_label_position('top')
      #ax[ax_i].xaxis.tick_top()
      if cesc == '0':
         ax[ax_i].set_title('Lissajous scans', fontsize=16, pad=40)
      if cesc == '1':
         ax[ax_i].set_title('CES scans', fontsize=16, pad=40)  
   ax[ax_i].text(.5,.9,field,horizontalalignment='center',transform=ax[ax_i].transAxes, fontsize=16)     
   ax[ax_i].set_xlim(0.04,0.7)
   ax[ax_i].set_xscale('log')
   #ax[ax_i].set_title(field, fontsize=16)
   ax[ax_i].grid()
   ax[ax_i].set_xlabel(r'$k$ [Mpc${}^{-1}$]', fontsize=14)
   labnums = [0.05,0.1, 0.2, 0.5]
   ax[ax_i].set_xticks(labnums)
   ax[ax_i].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   #plt.legend(bbox_to_anchor=(0, 0.61))
   #ax[ax_i].legend(ncol=4)
   return l1,l2,l3,l4

def plot_nulltest2(cesc):
   k7, xs_mean7, xs_sigma7 = read_h5_arrays('co7_map_null_1D_arrays.h5') 
   xs_mean7 = xs_mean7[8]
   k7 = k7[8]
   lim = np.mean(np.abs(xs_mean7[4:-2] * k7[4:-2])) * 8
   if cesc == '0':
      jk_we_want = [0,2,6,12] #indices of jk we want to use: wint, half, odde, dayn
   if cesc == '1':
      jk_we_want = [1,3,7,13]


   fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(17,4))
   
  
   l1,l2,l3,l4 = plot_sub_fig2('CO2',jk_we_want,0,lim,cesc,ax)
  
   l1,l2,l3,l4 = plot_sub_fig2('CO6',jk_we_want,1,lim,cesc,ax)
  
   l1,l2,l3,l4 = plot_sub_fig2('CO7',jk_we_want,2,lim,cesc,ax)
   plt.figlegend((l1,l2,l3,l4), ('Ambient temp', 'Wind speed', 'Rise', 'Fpol'),loc='upper center',bbox_to_anchor=(0.52,0.9), ncol=4, fontsize=14)
   plt.tight_layout()
   if cesc == '0':
      #plt.title('Lissajous scans', fontsize=16, loc='right')
      plt.savefig('nulltests_3fields_liss.pdf', bbox_inches='tight')
   if cesc == '1':
      #plt.title('CES scans', fontsize=16, loc='right')
      plt.savefig('nulltests_3fields_ces.pdf', bbox_inches='tight')
   

plot_nulltest2('0')
plot_nulltest2('1')
'''

