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


def read_number_of_splits(mapfile, jk):
   with h5py.File(mapfile, mode="r") as my_file:
       my_map = np.array(my_file['/jackknives/map_' + jk])
       sh = my_map.shape   
       number_of_splits = sh[0]   
   return number_of_splits

def xs_feed_feed_grid(map_file):
   went_through_first_cut = 0
   went_through_sigma_cut = 0
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
            #if i != 7 and j != 7:
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

              chi2[i, j] = np.sign(chi3) * abs((np.sum((xs[i,j] / rms_xs_std[i,j]) ** 2) - n_k) / np.sqrt(2 * n_k))      
              #print (abs(chi2[i,j]))
              #if abs(chi2[i,j]) < 5. and not np.isnan(chi2[i,j]) and i != j:  #if excess power is smaller than 5 sigma, chi2 is not nan, not on diagonal
              if not np.isnan(chi2[i,j]) and i != j:
                  went_through_first_cut += 1
                  if abs(chi2[i,j]) < 5.:
                     went_through_sigma_cut += 1
                     xs_sum += xs[i,j] / rms_xs_std[i,j] ** 2
                     #print ("if test worked")
                     xs_div += 1 / rms_xs_std[i,j] ** 2
                     n_sum += 1

      #tools.ensure_dir_exists('chi2_grids')
      figure_name = 'xs_grid_' + name_of_map + '_splits' + split1 + split2 + '.png'
      print (figure_name)
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
   return chi2

map_files = ['co2_map_elev_cesc_1.h5', 'co2_map_elev_cesc_0.h5', 'co6_map_elev_cesc_1.h5', 'co6_map_elev_cesc_0.h5', 'co7_map_elev_cesc_1.h5','co7_map_elev_cesc_0.h5']
chi2_array = []
for map_file in map_files:
   chi2 = xs_feed_feed_grid(map_file)
   chi2_array.append(chi2)

chi2_array = np.array(chi2_array)
np.save('chi2_array.npy', chi2_array)


'''
def check_sigma_cut(map_file):
   k, xs_mean, xs_sigma, field, ff_jk, split_names, split_numbers, went1, went2 = xs_feed_feed_grid(map_file)
   print ('Mapfile:', map_file)
   print ('Went through first cut:', went1)
   print ('Went through 5 sigma cut:', went2)
   print ('Difference:', went1-went2)
   print ('Surviving fraction:', 100-(100*(went1-went2)/went1))
   print ('---------------------------')
  
check_sigma_cut('co2_map_elev_cesc_1.h5')
check_sigma_cut('co2_map_elev_cesc_0.h5')

check_sigma_cut('co6_map_elev_cesc_1.h5')
check_sigma_cut('co6_map_elev_cesc_0.h5')

check_sigma_cut('co7_map_elev_cesc_1.h5')
check_sigma_cut('co7_map_elev_cesc_0.h5')

Mapfile: co2_map_elev_cesc_1.h5
Went through first cut: 240
Went through 5 sigma cut: 232
Difference: 8
Surviving fraction: 96.66666666666667
---------------------------
Mapfile: co2_map_elev_cesc_0.h5
Went through first cut: 240
Went through 5 sigma cut: 190
Difference: 50
Surviving fraction: 79.16666666666667
---------------------------
Mapfile: co6_map_elev_cesc_1.h5
Went through first cut: 240
Went through 5 sigma cut: 216
Difference: 24
Surviving fraction: 90.0
---------------------------
Mapfile: co6_map_elev_cesc_0.h5
Went through first cut: 240
Went through 5 sigma cut: 200
Difference: 40
Surviving fraction: 83.33333333333333
---------------------------
Mapfile: co7_map_elev_cesc_1.h5
Went through first cut: 240
Went through 5 sigma cut: 219
Difference: 21
Surviving fraction: 91.25
---------------------------
Mapfile: co7_map_elev_cesc_0.h5
Went through first cut: 240
Went through 5 sigma cut: 207
Difference: 33
Surviving fraction: 86.25
---------------------------
'''


