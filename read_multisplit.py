import h5py
import numpy as np
import itertools as itr


map_name = 'co6_map_good_splittest.h5'
field = 'co6'
mappath = '/mn/stornext/d16/cmbco/comap/nils/COMAP_general/data/maps/successive_split_test/' + map_name



jk_list = '/mn/stornext/d16/cmbco/comap/protodir/auxiliary/jk_list_splittest.txt'
#jk_list = 'jk_list_splittest.txt'


# --- READ jk_list ---

# marked with 3 - control variables - produce all the different combinations of these 
# marked with 2 - test variables - look at only one of these, while the rest is co-added - can be found in the map file
# marked with extra 1 - the variable used for feed-feed cross spectra

def read_jk(filename):
   jk_file = open(filename, 'r')
   all_lines = jk_file.readlines()
   jk_file.close()
   all_lines = all_lines[2:] #skip the first two lines (number of different jk and accr)
   control_variables = [] #marked with 3
   cv_index = []
   test_variables = [] #marked with 2
   tv_index = []
   feed_feed_variables = [] #extra 1
   ffv_index = []
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
         cv_index.append(index)
      if number == '2' and extra != '1':
         test_variables.append(variable)
         tv_index.append(index)
      if extra == '1':
         feed_feed_variables.append(variable) 
         ffv_index.append(index)
  # print control_variables, cv_index
  # print test_variables, tv_index
  # print feed_feed_variables, ffv_index
  # print all_variables
   return control_variables, cv_index, test_variables, tv_index, feed_feed_variables, ffv_index, all_variables

control_variables, cv_index, test_variables, tv_index, feed_feed_variables, ffv_index, all_variables = read_jk(jk_list)


def read_map(mappath, control_variables, test_variables, feed_feed_variables, all_variables):
   input_map = h5py.File(mappath, 'r')
   x = np.array(input_map['x'][:])
   y = np.array(input_map['y'][:])
   multisplits = input_map['multisplits']
   for test_variable in test_variables:
      map_split = np.array(multisplits['map_' + test_variable][:])
      rms_split = np.array(jackknives['rms_' + test_variable][:])
      shp = map_split.shape
      how_many_twos = len(all_variables) - len(test_variables) + 1
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
      split_names = []
      split_names.append(test_variable)
      for i in range(len(control_variables)):
         split_names.append(control_variables[-1-i])
      for i in range(len(feed_feed_variables)):
         split_names.append(feed_feed_variables[-1-i])
      how_many_to_combine = len(split_names) - 1
      all_different_possibilities = list(itr.product(range(2), repeat=how_many_to_combine))
      slc = [slice(None)]*len(new_shape)
      maps_created = []
      for i in range(len(all_different_possibilities)): #this many maps will be created
         for_naming = []
         for j in range(len(how_many_to_combine)):
            slc[j+1] = all_different_possibilities[i][j]
            for_naming.append(split_names[j+1], all_different_possibilities[i][j])
            print (split_names[j+1], all_different_possibilities[i][j])
         my_map = map_split[slc] 
         my_rms = rms_split[slc]
         name = field + '_' + 'map' + split_names[0] 
         for k in range(len(for_naming)):
            name += '_'
            name += str(for_naming[k])
         name += '.h5'
         maps_created.append(name)
         print ('Creating HDF5 file for the map ' + name + '.')
         f = h5py.File(new_mapname, 'w')
         f.create_dataset('x', data=x)
         f.create_dataset('y', data=y)
         f.create_dataset('/jackknives/map_' + split_names[0], data=my_map)
         f.create_dataset('/jackknives/rms_' + split_names[0], data=my_rms)
         f.close()
   return maps_created

print (read_map(mappath, control_variables, test_variables, feed_feed_variables, all_variables))


'''


keys_list = list(input_map.keys())

#print (keys_list)
'''
['feeds', 'freq', 'jackknives', 'map', 'map_coadd', 'mean_az', 'mean_el', 'n_x', 'n_y', 'nhit', 'nhit_coadd', 'njk', 'nside', 'nsim', 'patch_center', 'rms', 'rms_coadd', 'time', 'x', 'y']
'''
print ('Reading common parts.')
data_map = np.array(input_map['map'][:])
rms_map = np.array(input_map['rms'][:])
data_beam_map = np.array(input_map['map_coadd'][:])
rms_beam_map = np.array(input_map['rms_coadd'][:])
x = np.array(input_map['x'][:])
y = np.array(input_map['y'][:])
freq = np.array(input_map['freq'][:])

jackknives = input_map['jackknives']

#print (jackknives.keys())
'''
['jk_def', 'jk_feedmap', 'map_sidr', 'map_split', 'nhit_sidr', 'nhit_split', 'rms_sidr', 'rms_split']
'''
map_split = np.array(jackknives['map_split'][:]) #shape (16, 19, 4, 64, 120, 120)
rms_split = np.array(jackknives['rms_split'][:]) #shape (16, 19, 4, 64, 120, 120)
shp = map_split.shape
map_split = map_split.reshape((2,2,2,2,shp[1],shp[2],shp[3],shp[4],shp[5])) #cesc, snup, sune, elev, feed, sideband, freq, x, y
rms_split = rms_split.reshape((2,2,2,2,shp[1],shp[2],shp[3],shp[4],shp[5]))

def coadd_elev(old_map_split, old_rms_split):
   new_map_shape = (2,2,2,19,4,64,120,120)
   new_map_split = np.zeros(new_map_shape)
   new_rms_split = np.zeros(new_map_shape)
   w_sum = np.zeros(new_map_shape)
   print ('Coadding elev-split.')
   for i in range(2):
      mask = np.zeros(new_map_shape)
      mask[(old_rms_split[:,:,:,i,:,:,:,:,:] != 0.0)] = 1.0
      where = (mask == 1.0) 
      weight = np.zeros(new_map_shape)
      weight[where] = 1./old_rms_split[:,:,:,i,:,:,:,:,:][where]**2.
      w_sum += weight
      new_map_split += weight*old_map_split[:,:,:,i,:,:,:,:,:]
   
   mask2 =  np.zeros(new_map_shape)
   mask2[(w_sum != 0.0)] = 1.0
   where2 = (mask2 == 1.0)
   new_map_split[where2] = new_map_split[where2]/w_sum[where2]
   new_rms_split[where2] = w_sum[where2]**(-0.5)  
   return new_map_split, new_rms_split

def coadd_sune(old_map_split, old_rms_split):
   new_map_shape = (2,2,2,19,4,64,120,120)
   new_map_split = np.zeros(new_map_shape)
   new_rms_split = np.zeros(new_map_shape)
   w_sum = np.zeros(new_map_shape)
   print ('Coadding sune-split.')
   for i in range(2):
      mask = np.zeros(new_map_shape)
      mask[(old_rms_split[:,:,i,:,:,:,:,:,:] != 0.0)] = 1.0
      where = (mask == 1.0) 
      weight = np.zeros(new_map_shape)
      weight[where] = 1./old_rms_split[:,:,i,:,:,:,:,:,:][where]**2.
      w_sum += weight
      new_map_split += weight*old_map_split[:,:,i,:,:,:,:,:,:]
   
   mask2 =  np.zeros(new_map_shape)
   mask2[(w_sum != 0.0)] = 1.0
   where2 = (mask2 == 1.0)
   new_map_split[where2] = new_map_split[where2]/w_sum[where2]
   new_rms_split[where2] = w_sum[where2]**(-0.5)  
   return new_map_split, new_rms_split

map_split_coadded_elev, rms_split_coadded_elev = coadd_elev(map_split, rms_split) #cesc, snup, sune, feed, sideband, freq, x, y

map_split_coadded_sune, rms_split_coadded_sune = coadd_sune(map_split, rms_split) #cesc, snup, elev, feed, sideband, freq, x, y

mapnames_created = [] 
def create_output_map(cesc, snup, field, map_out, rms_out):
    #create the name
    part0 = field + '_elmap_' #cause I write El split as dayn
    my_map = map_out[cesc,snup,:,:,:,:,:,:]
    my_rms = rms_out[cesc,snup,:,:,:,:,:,:]
    if cesc == 0:
       part2 = 'liss.h5' #this is liss, I fixed it for this version
    if cesc == 1:
       part2 = 'ces.h5' 
    if snup == 0:
       part1 = 'night_'
    if snup == 1:
       part1 = 'day_'
    
    new_mapname = part0 + part1 + part2
    print ('Creating HDF5 file for the map ' + new_mapname + '.')
    mapnames_created.append(new_mapname)

    f = h5py.File(new_mapname, 'w')
    f.create_dataset('rms', data=rms_map)
    f.create_dataset('map', data=data_map)
    f.create_dataset('rms_coadd', data=rms_beam_map) 
    f.create_dataset('map_coadd', data=data_beam_map) 
    f.create_dataset('x', data=x)
    f.create_dataset('y', data=y)
    f.create_dataset('freq', data=freq)
    f.create_dataset('/jackknives/map_dayn', data=my_map)
    f.create_dataset('/jackknives/rms_dayn', data=my_rms)
    f.close()




create_output_map(0,1,field, map_split_coadded_sune, rms_split_coadded_sune)
create_output_map(1,1,field, map_split_coadded_sune, rms_split_coadded_sune)

create_output_map(0,0,field, map_split_coadded_sune, rms_split_coadded_sune)
create_output_map(1,0,field, map_split_coadded_sune, rms_split_coadded_sune)

#create_output_map(0,0,field, map_split_coadded_elev, rms_split_coadded_elev)
#create_output_map(1,0,field, map_split_coadded_elev, rms_split_coadded_elev)

print ('All the maps created: ', mapnames_created)


'''
