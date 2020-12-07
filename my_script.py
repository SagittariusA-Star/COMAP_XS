import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import corner
import h5py
import sys

import tools
import map_cosmo
import my_class
import multiprocessing
import read_multisplit
import mean_multisplit

def run_all_methods(feed1,feed2, n_of_splits, two_dimensions=False):
   my_xs = my_class.CrossSpectrum_nmaps(mapfile,jk,feed1, feed2, n_of_splits)

   calculated_xs = my_xs.get_information() #gives the xs, k, rms_sig, rms_mean index with corresponding map-pair

   if two_dimensions == False:
      xs, k, nmodes = my_xs.calculate_xs()
      rms_mean, rms_sig = my_xs.run_noise_sims(10) #these rms's are arrays of 14 elements, that give error bars (number of bin edges minus 1)

      #plot all cross-spectra that have been calculated
      my_xs.plot_xs(k, xs, rms_sig, rms_mean, save=True)
      my_xs.make_h5()
   if two_dimensions == True:
      print ('Not implemented yet!')
      #write all of these functions in 2D as well

def all_feed_combo_xs(p):
   i = p // 19 + 1 #floor division, divides and returns the integer value of the quotient (it dumps the digits after the decimal)
   j = p % 19 + 1 #modulus, divides and returns the value of the remainder
    
   if i == 4 or i == 6 or i == 7: #avoid these feeds (were turned off for most of the mission)
       return p
   if j == 4 or j == 6 or j == 7: #avoid these feeds (were turned off for most of the mission)
       return p
   run_all_methods(i,j, n_of_splits, two_dimensions)
   return p

def read_number_of_splits(mapfile, jk):
   with h5py.File(mapfile, mode="r") as my_file:
       my_map = np.array(my_file['/jackknives/map_' + jk])
       sh = my_map.shape   
       number_of_splits = sh[0]   
   return number_of_splits

def read_field_jklist(mappath):
   map_name = mappath.rpartition('/')[-1] #get rid of the path, leave only the name of the map
   map_name = map_name.rpartition('.')[0] #get rid of the ".h5" part
   field_name = map_name.split('_')[0]
   last_part = map_name.split('_')[-1]
   jk_list = '/mn/stornext/d16/cmbco/comap/protodir/auxiliary/jk_list_' + last_part + '.txt'
   print ('Field:', field_name)
   print ('List of split-variables:', jk_list)
   return field_name, jk_list


#read from the command:
#sys.argv[-1] = mappath
mappath = '/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/successive_split_test/co6_map_good_splittest.h5'

xs_2D = input("Cross-spectra in two dimensions? (yes/no) ")
if xs_2D == 'yes':
   two_dimensions = True
if xs_2D == 'no':
   two_dimensions = False

field, jk_list = read_field_jklist(mappath)

control_variables, test_variables, feed_feed_variables, all_variables = read_multisplit.read_jk(jk_list)

map_files = read_multisplit.read_map(mappath, field, control_variables, test_variables, feed_feed_variables, all_variables)

number_of_maps = len(map_files)
number_of_ff_variables = len(feed_feed_variables)
maps_per_jk = int(number_of_maps/number_of_ff_variables)
feed_combos = list(range(19*19)) #number of combinations between feeds
'''
print ('STAGE 3: Calculating cross-spectra for all feed-feed combinations.')
for g in range(number_of_ff_variables):
   for h in range(maps_per_jk):
      jk = feed_feed_variables[g]
      mapname = map_files[g*maps_per_jk+h]
      print ('Split for FPXS: ' + jk + '. Map: ' + mapname + '.')
      mapfile = 'split_maps/' + mapname
      n_of_splits = read_number_of_splits(mapfile, jk)
      #make xs for all feed-combinations
      pool = multiprocessing.Pool(8) #here number of cores
      np.array(pool.map(all_feed_combo_xs, feed_combos))
'''
print ('STAGE 4: Calculating the mean of cross-spectra from all feed-feed combinations.')
k_arr = []
xs_mean_arr = []
xs_sigma_arr = []
field_arr = []
ff_jk_arr = []
split_names_arr = []
split_numbers_arr = []
for mn in range(number_of_maps):
   if two_dimensions == True:
      print ('Not implemented yet!')
   if two_dimensions == False:
      k, xs_mean, xs_sigma, field, ff_jk, split_names, split_numbers = mean_multisplit.xs_feed_feed_grid(map_files[mn]) #saves the chi2 grid for each split-combo
   
   k_arr.append(k)
   xs_mean_arr.append(xs_mean)
   xs_sigma_arr.append(xs_sigma)
   field_arr.append(field)
   ff_jk_arr.append(ff_jk)
   split_names_arr.append(split_names)
   split_numbers_arr.append(split_numbers)
how_many_different_splits = len(split_names)

#group maps with respect to scanning strategy
index_cesc = split_names_arr[0].index('cesc')

# plot xs mean
for mn in range(number_of_maps):
   last_name_part = '_'
   other = ' '
   for ds in range(how_many_different_splits):
      last_part = split_names_arr[mn][ds] + split_numbers_arr[mn][ds] + '_'
      other_part = split_names_arr[mn][ds] + '-' + split_numbers_arr[mn][ds] + ', '
      last_name_part += last_part
      other += other_part
   figure_name = 'xs_mean_' + field_arr[mn] + '_map_' + ff_jk_arr[mn] + last_name_part + '.pdf'
   figure_title = 'Field: ' + field_arr[mn] + '; Feed-feed variable: ' + ff_jk_arr[mn] + '; Other splits:' + other
   
   if split_numbers_arr[mn][index_cesc] == '0': #cesc=0
      scan_strategy = 'liss'
   if split_numbers_arr[mn][index_cesc] == '1': #cesc=0
      scan_strategy = 'ces'
   if two_dimensions == False:
      print ('Saving the figure ' + figure_name)
      mean_multisplit.xs_with_model(figure_name, k_arr[mn], xs_mean_arr[mn], xs_sigma_arr[mn], figure_title, scan_strategy)


    


   



