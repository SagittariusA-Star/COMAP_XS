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

def run_all_methods(feed,feed1,feed2):
  
   my_xs = my_class.CrossSpectrum_nmaps(map_name,jk, feed, feed1, feed2)

   calculated_xs = my_xs.get_information() #gives the xs, k, rms_sig, rms_mean index with corresponding map-pair
   print ('Created xs between ' + calculated_xs[0][1] + ' and ' + calculated_xs[0][2] + '.')
   xs, k, nmodes = my_xs.calculate_xs(print_show=False)
   rms_mean, rms_sig = my_xs.run_noise_sims(10) #these rms's are arrays of 14 elements, that give error bars (number of bin edges minus 1)

   #plot all cross-spectra that have been calculated
   my_xs.plot_xs(k, xs, rms_sig, rms_mean, 0, save=True)
   my_xs.make_h5(0)
   

def all_feed_combo_xs(p):
    i = p // 19 + 1 #floor division, divides and returns the integer value of the quotient (it dumps the digits after the decimal)
    j = p % 19 + 1 #modulus, divides and returns the value of the remainder
    
    if i == 4 or i == 6 or i == 7: #avoid these feeds (were turned off for most of the mission)
        return p
    if j == 4 or j == 6 or j == 7: #avoid these feeds (were turned off for most of the mission)
        return p
    run_all_methods(None, feed1=i,feed2=j)
    return p


#sys.argv[-1] = mappath
#sys.argv[-1] = field
map_name = 'co6_map_good_splittest.h5'
field = 'co6'
mappath = '/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/successive_split_test/' + map_name
jk_list = '/mn/stornext/d16/cmbco/comap/protodir/auxiliary/jk_list_splittest.txt'

control_variables, test_variables, feed_feed_variables, all_variables = read_multisplit.read_jk(jk_list)
map_files, jk_collection = read_multisplit.read_map(mappath, field, control_variables, test_variables, feed_feed_variables, all_variables)
 
number_of_maps = len(map_files)
number_of_test_variables = len(jk_collection)
maps_per_jk = int(number_of_maps/number_of_test_variables)
feed_combos = list(range(19*19)) #number of combinations between feeds

for g in range(number_of_test_variables):
   for h in range(maps_per_jk):
      jk = jk_collection[g]
      map_name = maps_files[g*maps_per_jk+h]
      map_name = 'split_maps/' + map_name
      print (jk, map_name)
      #make xs for all feed-combinations
      #pool = multiprocessing.Pool(8) #here number of cores
      #np.array(pool.map(all_feed_combo_xs, feed_combos))

