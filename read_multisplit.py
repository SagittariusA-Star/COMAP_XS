#THE VERSION WHERE I ACCOUNT FOR MANY FEED-FEED VARIABLES

import h5py
import numpy as np
import itertools as itr
import tools


# --- READ jk_list ---

#(! keep test variables after control variables in jk_list)
# marked with 3 - control variables - produce all the different combinations of these 
# marked with 2 - test variables - look at only one of these, while the rest is co-added - can be found in the map file
# marked with extra 1 - the variable used for feed-feed cross spectra

def read_jk(filename):
   print ('STAGE 1/4: Reading the list of variables associated with the map.')
   jk_file = open(filename, 'r')
   all_lines = jk_file.readlines()
   jk_file.close()
   all_lines = all_lines[2:] #skip the first two lines (number of different jk and accr)
   control_variables = [] #marked with 3
   test_variables = [] #marked with 2
   feed_feed_variables = [] #extra 1
   all_variables = []
   for line in all_lines:
      split_line = line.split()
      variable = split_line[0]
      number = split_line[1]
      if len(split_line) > 2:
         extra = split_line[2]
      if len(split_line) < 2 or len(split_line) == 2:
         extra = '#'
      all_variables.append(variable)

      if number == '3':
         control_variables.append(variable)
        
      if number == '2':
         test_variables.append(variable)
         
      if extra == '1':
         feed_feed_variables.append(variable) 

   #find all feed-feed variables that are also test variables or control variables
   feed_and_test = []
   feed_and_control = [] 
   for variable in all_variables:
      if variable in test_variables and variable in feed_feed_variables:
         feed_and_test.append(variable)
      if variable in feed_feed_variables and variable in control_variables:
         feed_and_control.append(variable)
 
   return control_variables, test_variables, feed_feed_variables, all_variables, feed_and_test, feed_and_control

def read_map(mappath,field, control_variables, test_variables, feed_feed_variables, all_variables, feed_and_test, feed_and_control):
   print ('STAGE 2/4: Splitting the map into subsets with different split combinations.')
   input_map = h5py.File(mappath, 'r')

   x = np.array(input_map['x'][:]) #common part for all maps
   y = np.array(input_map['y'][:]) #common part for all maps
   multisplits = input_map['multisplits']
   maps_created = []
   if len(feed_and_test) != 0: #if some test variables are simultaneously feed-feed variables
      for test_variable in feed_and_test:
     
            map_split = np.array(multisplits['map_' + test_variable][:])
            rms_split = np.array(multisplits['rms_' + test_variable][:])
            shp = map_split.shape
            how_many_twos = len(control_variables) + 1 #how many parts to reshape the map with respect to splits
            new_shape = []
            for i in range(how_many_twos):
               new_shape.append(2) #because we split in 2 - needs to be changed if more splits are implemented
            new_shape.append(shp[1]) #feed
            new_shape.append(shp[2]) #sideband
            new_shape.append(shp[3]) #freq
            new_shape.append(shp[4]) #x
            new_shape.append(shp[5]) #y
            map_split = map_split.reshape(new_shape)
            rms_split = rms_split.reshape(new_shape)
            split_names = [] #collect the names of the spits in the correct order for the new shape
            split_names.append(test_variable)
            for i in range(how_many_twos-1):
               split_names.append(control_variables[-1-i])
             
            how_many_to_combine = len(split_names) -1 #all control variables
            all_different_possibilities = list(itr.product(range(2), repeat=how_many_to_combine)) #find all the combinations of 'how_many_to_combine' 0s and 1s  
            index_of_ff_variable = 0
   
            all_axes_to_combine = list(range(0,how_many_to_combine+1))
            
            all_axes_to_combine.remove(index_of_ff_variable) #all axes for different combinations of splits, include both splits for the feed-feed variable
      
            slc = [slice(None)]*len(new_shape) #includes all elements
      
            for i in range(len(all_different_possibilities)): #this many maps will be created
               for_naming = [] #identify which combination of splits the current map is using
                
               for j in range(how_many_to_combine):
                  axis_index = all_axes_to_combine[j]
                  slc[axis_index] = all_different_possibilities[i][j] #choose 0 or 1 for this split
                  for_naming.append(split_names[axis_index])
                  for_naming.append(all_different_possibilities[i][j])
                 
               my_map = map_split[tuple(slc)] #slice the map for the current combination of splits
               my_rms = rms_split[tuple(slc)] #slice the rms-map for the current combination of splits
               name = field + '_' + 'map' + '_' + test_variable
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
               f.create_dataset('/jackknives/map_' + test_variable, data=my_map)
               f.create_dataset('/jackknives/rms_' + test_variable, data=my_rms)
               f.close()
   if len(feed_and_control) != 0: #if some feed-feed variables are control variables
      for ff_variable in feed_and_control:
         for test_variable in test_variables:
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
            for i in range(how_many_twos-1):
               split_names.append(all_variables[-len(test_variables)-1-i])
             
            how_many_to_combine = len(split_names) -1 #test variable + all control variables, except for the ff_variable
            all_different_possibilities = list(itr.product(range(2), repeat=how_many_to_combine)) #find all the combinations of 'how_many_to_combine' 0s and 1s  
            index_of_ff_variable = split_names.index(ff_variable)
   
            all_axes_to_combine = list(range(0,how_many_to_combine+1))
            
            all_axes_to_combine.remove(index_of_ff_variable) #all axes for different combinations of splits, include both splits for the feed-feed variable
      
            slc = [slice(None)]*len(new_shape) #includes all elements
      
            for i in range(len(all_different_possibilities)): #this many maps will be created
               for_naming = [] #identify which combination of splits the current map is using
                
               for j in range(how_many_to_combine):
                  axis_index = all_axes_to_combine[j]
                  slc[axis_index] = all_different_possibilities[i][j] #choose 0 or 1 for this split
                  for_naming.append(split_names[axis_index])
                  for_naming.append(all_different_possibilities[i][j])
                 
               my_map = map_split[tuple(slc)] #slice the map for the current combination of splits
               my_rms = rms_split[tuple(slc)] #slice the rms-map for the current combination of splits
               name = field + '_' + 'map' + '_' + ff_variable
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
               f.create_dataset('/jackknives/map_' + ff_variable, data=my_map)
               f.create_dataset('/jackknives/rms_' + ff_variable, data=my_rms)
               f.close()
   
   return maps_created


def null_test_subtract(maps_created):
   print ('Subtracting test-variable maps for the null-tests.')
   '''
   We want to subract split-maps (split according to test variables) for both cesc = 0 and cesc = 1 and then create new map-files for the FPXS.
   '''
   
   return new_subtracted_maps

def read_field_jklist(mappath):
   map_name = mappath.rpartition('/')[-1] #get rid of the path, leave only the name of the map
   map_name = map_name.rpartition('.')[0] #get rid of the ".h5" part
   field_name = map_name.split('_')[0]
   last_part = map_name.split('_')[-1]
   jk_list = '/mn/stornext/d16/cmbco/comap/protodir/auxiliary/jk_list_' + last_part + '.txt'
   print ('Field:', field_name)
   print ('List of split-variables:', jk_list)
   return field_name, jk_list, map_name

mappath = '/mn/stornext/d16/cmbco/comap/protodir/maps/co6_map_null.h5'
field_name, jk_list, map_name = read_field_jklist(mappath)
control_variables, test_variables, feed_feed_variables, all_variables, feed_and_test, feed_and_control = read_jk(jk_list)
maps_created = read_map(mappath,field_name, control_variables, test_variables, feed_feed_variables, all_variables, feed_and_test, feed_and_control)
print (maps_created)


'''
EXAMPLES:

--jk_list:
6        # number of different jack-knives (including acceptlist)
accr     # accept/reject (reject=0)
snup     3 1
cesc     3 #
elev     2 #
ambt     2 #
half     2 # sad
--maps:
['co2_map_snup_elev_0_cesc_0.h5', 'co2_map_snup_elev_0_cesc_1.h5', 'co2_map_snup_elev_1_cesc_0.h5', 'co2_map_snup_elev_1_cesc_1.h5', 'co2_map_snup_ambt_0_cesc_0.h5', 'co2_map_snup_ambt_0_cesc_1.h5', 'co2_map_snup_ambt_1_cesc_0.h5', 'co2_map_snup_ambt_1_cesc_1.h5', 'co2_map_snup_half_0_cesc_0.h5', 'co2_map_snup_half_0_cesc_1.h5', 'co2_map_snup_half_1_cesc_0.h5', 'co2_map_snup_half_1_cesc_1.h5']


--jk_list_signal.txt:
4        # number of different jack-knives (including acceptlist)
accr     # accept/reject (reject=0)
cesc     3 #
elev     2 1 #
dayn     2 1
--maps:
['co2_map_elev_cesc_0.h5', 'co2_map_elev_cesc_1.h5', 'co2_map_dayn_cesc_0.h5', 'co2_map_dayn_cesc_1.h5']

--jk_list_signal.txt, new file:
9        # number of different jack-knives (including acceptlist)
accr     # accept/reject (reject=0)
cesc     3 #
elev     2 1 #
dayn     2 1
sidr     2 1
ambt     2 1
wind     2 1
wint     2 1
rise     2 1


--jk_list_null.txt, for the null test
11	 # number of different jack-knives (including acceptlist)
accr     # accept/reject (reject=0)
cesc     3 #
elev     3 1 #
ambt     2
wind     2
wint     2
rise     2
half     2
odde     2
fpol     2
dayn     2

'''


