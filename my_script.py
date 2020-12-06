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

def run_all_methods(feed1,feed2, n_of_splits, two_dimensions=False):
  
   my_xs = my_class.CrossSpectrum_nmaps(map_name,jk,feed1, feed2, n_of_splits)

   calculated_xs = my_xs.get_information() #gives the xs, k, rms_sig, rms_mean index with corresponding map-pair
   print ('Created xs between ' + calculated_xs[0][1] + ' and ' + calculated_xs[0][2] + '.')
   if two_dimensions == False:
      xs, k, nmodes = my_xs.calculate_xs()
      rms_mean, rms_sig = my_xs.run_noise_sims(10) #these rms's are arrays of 14 elements, that give error bars (number of bin edges minus 1)

      #plot all cross-spectra that have been calculated
      my_xs.plot_xs(k, xs, rms_sig, rms_mean, 0, save=True)
      my_xs.make_h5(0)
   if two_dimenstions == True:
      print ('Not implemented yet!')
      #write all of these functions in 2D as well

def all_feed_combo_xs(p):
   i = p // 19 + 1 #floor division, divides and returns the integer value of the quotient (it dumps the digits after the decimal)
   j = p % 19 + 1 #modulus, divides and returns the value of the remainder
    
   if i == 4 or i == 6 or i == 7: #avoid these feeds (were turned off for most of the mission)
       return p
   if j == 4 or j == 6 or j == 7: #avoid these feeds (were turned off for most of the mission)
       return p
   run_all_methods(feed1=i,feed2=j, n_of_splits, two_dimensions)
   return p

def read_number_of_splits(map_name, jk):
   with h5py.File(map_name, mode="r") as my_file:
       my_map = np.array(my_file['/jackknives/map_' + jk])
       sh = my_map.shape   
       number_of_splits = sh[0]   
       print ('n of splits', number_of_splits)
   return number_of_splits

def read_field(mappath):
   map_name = mappath.rpartition('/')[-1] #get rid of the path, leave only the name of the map
   map_name = map_name.rpartition('.')[0] #get rid of the ".h5" part
   field_name = map_name.split('_')[0]
   print ('field:', field_name)
   return field_name

#read two things from the command:
#sys.argv[-2] = mappath
#sys.argv[-1] = two_dimensions

map_name = 'co6_map_good_splittest.h5'

mappath = '/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/successive_split_test/' + map_name
jk_list = '/mn/stornext/d16/cmbco/comap/protodir/auxiliary/jk_list_splittest.txt'
two_dimensions = False
field = read_field(mappath)

control_variables, test_variables, feed_feed_variables, all_variables = read_multisplit.read_jk(jk_list)

map_files = read_multisplit.read_map(mappath, field, control_variables, test_variables, feed_feed_variables, all_variables)

number_of_maps = len(map_files)
number_of_ff_variables = len(feed_feed_variables)
maps_per_jk = int(number_of_maps/number_of_ff_variables)
feed_combos = list(range(19*19)) #number of combinations between feeds

print ('STAGE 3: Calculating cross-spectra for all feed-feed combinations.')
for g in range(number_of_ff_variables):
   for h in range(maps_per_jk):
      jk = feed_feed_variables[g]
      map_name = map_files[g*maps_per_jk+h]
      print ('Split for FPXS: ' + jk + '. Map: ' + map_name + '.')
      map_name = 'split_maps/' + map_name
      n_of_splits = read_number_of_splits(map_name, jk)
      #make xs for all feed-combinations
      #pool = multiprocessing.Pool(8) #here number of cores
      #np.array(pool.map(all_feed_combo_xs, feed_combos))

