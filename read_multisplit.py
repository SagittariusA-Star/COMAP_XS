import h5py
import numpy as np
import itertools as itr
import tools

'''
map_name = 'co6_map_good_splittest.h5'
field = 'co6'
mappath = '/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/successive_split_test/' + map_name

'''

#jk_list = '/mn/stornext/d16/cmbco/comap/protodir/auxiliary/jk_list_splittest.txt'
#jk_list = 'jk_list_splittest.txt'


# --- READ jk_list ---

# (important with jk_list - keep 1s first, then 2s, then 3s, and two first lines irrelevant)
# marked with 3 - control variables - produce all the different combinations of these 
# marked with 2 - test variables - look at only one of these, while the rest is co-added - can be found in the map file
# marked with extra 1 - the variable used for feed-feed cross spectra

def read_jk(filename):
   print ('STAGE 1: Reading the list of variables associated with the map.')
   jk_file = open(filename, 'r')
   all_lines = jk_file.readlines()
   jk_file.close()
   all_lines = all_lines[2:] #skip the first two lines (number of different jk and accr)
   control_variables = [] #marked with 3
   test_variables = [] #marked with 2
   feed_feed_variables = [] #extra 1
   all_variables = []
   index = -1
   for line in all_lines:
      index += 1
      split_line = line.split()
      variable = split_line[0]
      number = split_line[1]
      extra = split_line[2]
      all_variables.append(variable)

      if number == '3' and extra != '1':
         control_variables.append(variable)
        
      if number == '2' and extra != '1':
         test_variables.append(variable)
         
      if extra == '1':
         feed_feed_variables.append(variable) 
         
   return control_variables, test_variables, feed_feed_variables, all_variables


def read_map(mappath,field, control_variables, test_variables, feed_feed_variables, all_variables):
   print ('STAGE 2: Splitting the map into subsets with different split combinations.')
   input_map = h5py.File(mappath, 'r')

   x = np.array(input_map['x'][:]) #common part for all maps
   y = np.array(input_map['y'][:]) #common part for all maps
   multisplits = input_map['multisplits']
   maps_created = []
   for_feed_feed = [] #append the variables used for feed-feed cross-spectra
   for test_variable in test_variables:
      for_feed_feed.append(test_variable)
      map_split = np.array(multisplits['map_' + test_variable][:])
      rms_split = np.array(multisplits['rms_' + test_variable][:])
      shp = map_split.shape
      how_many_twos = len(all_variables) - len(test_variables) + 1 #how to reshape the map with respect to splits
      new_shape = []
      for i in range(how_many_twos):
         new_shape.append(2)
      new_shape.append(shp[1]) #feed
      new_shape.append(shp[2]) #sideband
      new_shape.append(shp[3]) #freq
      new_shape.append(shp[4]) #x
      new_shape.append(shp[5]) #y
      map_split = map_split.reshape(new_shape)
      rms_split = rms_split.reshape(new_shape)
      split_names = [] #collect the names of the spits in the correct order for the new shape
      split_names.append(test_variable)
      for i in range(len(control_variables)):
         split_names.append(control_variables[-1-i])
      for i in range(len(feed_feed_variables)):
         split_names.append(feed_feed_variables[-1-i])
      how_many_to_combine = len(split_names) - 1
      all_different_possibilities = list(itr.product(range(2), repeat=how_many_to_combine)) #find all the combinations of 'how_many_to_combine' 0s and 1s
      slc = [slice(None)]*len(new_shape) #includes all elements
      
      for i in range(len(all_different_possibilities)): #this many maps will be created
         for_naming = [] #identify which combination of splits the current map is using
         
         for j in range(how_many_to_combine):
            slc[j+1] = all_different_possibilities[i][j] #choose 0 or 1 for this split
            for_naming.append(split_names[j+1])
            for_naming.append(all_different_possibilities[i][j])
           #print (split_names[j+1], all_different_possibilities[i][j])
         my_map = map_split[tuple(slc)] #slice the map for the current combination of splits
         my_rms = rms_split[tuple(slc)] #slice the rms-map for the current combination of splits
         name = field + '_' + 'map' + '_' + split_names[0] 
         for k in range(len(for_naming)):
            name += '_'
            name += str(for_naming[k])
         name += '.h5'
         maps_created.append(name) #add the name of the current map to the list
         print ('Creating HDF5 file for the map ' + name + '.')
         tools.ensure_dir_exists('split_maps')
         outname = 'split_maps/' + name

         f = h5py.File(outname, 'w') #create HDF5 file with the sliced map
         f.create_dataset('x', data=x)
         f.create_dataset('y', data=y)
         f.create_dataset('/jackknives/map_' + split_names[0], data=my_map)
         f.create_dataset('/jackknives/rms_' + split_names[0], data=my_rms)
         f.close()
   return maps_created, for_feed_feed

'''
(['co6_map_elev_cesc_0_snup_0.h5', 'co6_map_elev_cesc_0_snup_1.h5', 'co6_map_elev_cesc_1_snup_0.h5', 'co6_map_elev_cesc_1_snup_1.h5', 'co6_map_ambt_cesc_0_snup_0.h5', 'co6_map_ambt_cesc_0_snup_1.h5', 'co6_map_ambt_cesc_1_snup_0.h5', 'co6_map_ambt_cesc_1_snup_1.h5', 'co6_map_half_cesc_0_snup_0.h5', 'co6_map_half_cesc_0_snup_1.h5', 'co6_map_half_cesc_1_snup_0.h5', 'co6_map_half_cesc_1_snup_1.h5'], ['elev', 'ambt', 'half'])
'''


