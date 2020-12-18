import h5py
import numpy as np
import tools

def read_map(mappath,field, control_variables, test_variables, feed_feed_variables, all_variables):
   print ('STAGE 2/4: Splitting the map into subsets with different split combinations.')
   input_map = h5py.File(mappath, 'r')

   x = np.array(input_map['x'][:]) #common part for all maps
   y = np.array(input_map['y'][:]) #common part for all maps
   multisplits = input_map['multisplits']
   map_dayn = np.array(multisplits['map_dayn'][:]) #{4, 19, 4, 64, 120, 120}
   rms_dayn = np.array(multisplits['rms_dayn'][:])
   map_elev = np.array(multisplits['map_elev'][:]) #{4, 19, 4, 64, 120, 120}
   rms_elev = np.array(multisplits['rms_elev'][:])
   shp = map_elev.shape
   map_dayn = map_dayn.reshape((2,2,shp[1],shp[2],shp[3],shp[4],shp[5])) #dayn,cesc,...
   rms_dayn = rms_dayn.reshape((2,2,shp[1],shp[2],shp[3],shp[4],shp[5]))
   map_elev = map_elev.reshape((2,2,shp[1],shp[2],shp[3],shp[4],shp[5])) #elev,cesc,...
   rms_elev = rms_elev.reshape((2,2,shp[1],shp[2],shp[3],shp[4],shp[5]))
   mapdayn1 = map_dayn[:,0,:,:,:,:,:]
   mapdayn2 = map_dayn[:,1,:,:,:,:,:]
   mapelev1 = map_elev[:,0,:,:,:,:,:]
   mapelev2 = map_elev[:,1,:,:,:,:,:]
   rmsdayn1 = rms_dayn[:,0,:,:,:,:,:]
   rmsdayn2 = rms_dayn[:,1,:,:,:,:,:]
   rmselev1 = rms_elev[:,0,:,:,:,:,:]
   rmselev2 = rms_elev[:,1,:,:,:,:,:]
   #maps_list = [mapdayn1, mapdayn2, mapelev1, mapelev2]
   #rms_list = [rmsdayn1, rmsdayn2, rmselev1, rmselev2]
   maps_created = ['co6_map_dayn_cesc_0.h5','co6_map_dayn_cesc_1.h5', 'co6_map_elev_cesc_0.h5', 'co6_map_elev_cesc_1.h5']
   create_h5_file('dayn', mapdayn1, rmsdayn1, maps_created[0],x,y)
   create_h5_file('dayn', mapdayn2, rmsdayn2, maps_created[1],x,y)
   create_h5_file('elev', mapelev1, rmselev1, maps_created[2],x,y)
   create_h5_file('elev', mapelev2, rmselev2, maps_created[3],x,y)
   return maps_created

def create_h5_file(ff_variable, my_map, my_rms, name,x,y):
   print ('Creating HDF5 file for the map ' + name + '.')
   tools.ensure_dir_exists('split_maps')
   outname = 'split_maps/' + name

   f = h5py.File(outname, 'w') #create HDF5 file with the sliced map
   f.create_dataset('x', data=x)
   f.create_dataset('y', data=y)
   f.create_dataset('/jackknives/map_' + ff_variable, data=my_map)
   f.create_dataset('/jackknives/rms_' + ff_variable, data=my_rms)
   f.close()   




'''

this is jk_list_signal.txt
4        # number of different jack-knives (including acceptlist)
accr     # accept/reject (reject=0)
cesc     3 #
elev     2 1 #
dayn     2 1

'''
