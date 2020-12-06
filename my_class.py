#to create feed-feed pseudo-cross-spectra between all feed combinations and all split combinations

import numpy as np
import h5py
import tools
import map_cosmo
import itertools as itr
import matplotlib.pyplot as plt
plt.ioff() #turn of the interactive plotting
import PS_function #P(k) = k**-3
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) #ignore warnings caused by weights cut-off

class CrossSpectrum_nmaps():
    def __init__(self, name_of_my_map, jk=False, feed1=None, feed2=None, n_of_splits=2):
        self.feed_name1 = '_feed' + str(feed1)
        self.feed_name2 = '_feed' + str(feed2)
        
        self.name_of_map = name_of_my_map #the names schould indicate which map and feed we take
        self.names = []
        self.maps = []        

        n_list = list(range(n_of_splits))
        all_different_possibilities = list(itr.combinations(n_list, 2)) #for n_of_splits = 3, it gives [(0, 1), (0, 2), (1, 2)]
        how_many_combinations = len(all_different_possibilities)
        
        self.name_of_map = self.name_of_map.split('/')[-1] #get rid of the path, leave only the name of the map
        self.name_of_map = self.name_of_map.split('.')[0] #get rid of the ".h5" part
        for u in range(how_many_combinations):
           current_combo = all_different_possibilities[u] #there are two splits from mapmaker so far, can be more from simulations
           name1 = self.name_of_map + '_split' + str(current_combo[0]) + self.feed_name1
           name2 = self.name_of_map + '_split' + str(current_combo[1]) + self.feed_name2
           self.names.append(name1)  
           self.names.append(name2)

        for u in range(how_many_combinations):
           current_combo = all_different_possibilities[u] #there are two splits from mapmaker so far, can be more from simulations
           my_map_split_1 = map_cosmo.MapCosmo(name_of_my_map, feed1, jk, current_combo[0])
           my_map_split_2 = map_cosmo.MapCosmo(name_of_my_map, feed2, jk, current_combo[1])
           self.maps.append(my_map_split_1)
           self.maps.append(my_map_split_2)    
         
   
    #NORMALIZE WEIGHTS FOR A GIVEN PAIR OF MAPS
    def normalize_weights(self, i, j):
        norm = np.sqrt(np.mean((self.maps[i].w * self.maps[j].w).flatten()))
        self.maps[i].w = self.maps[i].w / norm
        self.maps[j].w = self.maps[j].w / norm
       
    #REVERSE NORMALIZE_WEIGHTS, TO NORMALIZE EACH PAIR OF MAPS INDEPENDENTLY
    def reverse_normalization(self, i, j):    
        norm = np.sqrt(np.mean((self.maps[i].w * self.maps[j].w).flatten()))
        self.maps[i].w = self.maps[i].w*norm
        self.maps[j].w = self.maps[j].w*norm

    #INFORM WHICH XS INDEX CORRESPONDS TO WHICH MAP-PAIR
    def get_information(self):
        indexes_xs = []
        index = -1 
        for i in range(0,len(self.maps)-1, 2):
          j = i+1
          index += 1
          indexes_xs.append([index,self.names[i],self.names[j]])
        return indexes_xs
             
    #COMPUTE ALL THE XS BETWEEN THE n FEED-AVERAGED MAPS
    def calculate_xs(self, no_of_k_bins=15): #here take the number of k-bins as an argument 
        n_k = no_of_k_bins
        self.k_bin_edges = np.logspace(-2.0, np.log10(1.5), n_k)
        calculated_xs = self.get_information()        

        #store each cross-spectrum and corresponding k and nmodes by appending to these lists:
        self.xs = []
        self.k = []
        self.nmodes = []
        index = 0
        for i in range(0,len(self.maps)-1, 2):
           j = i + 1
           self.normalize_weights(i,j) #normalize weights for given xs pair of maps

           wi = self.maps[i].w
           wj = self.maps[j].w
           wh_i = np.where(np.log10(wi) < -0.5)
           wh_j = np.where(np.log10(wj) < -0.5)
           wi[wh_i] = 0.0
           wj[wh_j] = 0.0
              
           my_xs, my_k, my_nmodes = tools.compute_cross_spec3d(
           (self.maps[i].map * np.sqrt(wi*wj), self.maps[j].map * np.sqrt(wi*wj)),
           self.k_bin_edges, dx=self.maps[i].dx, dy=self.maps[i].dy, dz=self.maps[i].dz)

           self.reverse_normalization(i,j) #go back to the previous state to normalize again with a different map-pair

           self.xs.append(my_xs)
           self.k.append(my_k)
           self.nmodes.append(my_nmodes)
        self.xs = np.array(self.xs)
        self.k = np.array(self.k)
        self.nmodes = np.array(self.nmodes)
        return self.xs, self.k, self.nmodes
   
    #RUN NOISE SIMULATIONS (for all combinations of n maps, to match xs)
    def run_noise_sims(self, n_sims, seed=None):
        self.rms_xs_mean = []
        self.rms_xs_std = []
        for i in range(0,len(self.maps)-1, 2):
           j = i + 1
              
           self.normalize_weights(i,j)
           wi = self.maps[i].w
           wj = self.maps[j].w
           wh_i = np.where(np.log10(wi) < -0.5)
           wh_j = np.where(np.log10(wj) < -0.5)
           wi[wh_i] = 0.0
           wj[wh_j] = 0.0

           if seed is not None:
               if self.maps[i].feed is not None:
                   feeds = np.array([self.maps[i].feed, self.maps[j].feed])
               else:
                   feeds = np.array([1, 1])
            
           rms_xs = np.zeros((len(self.k_bin_edges) - 1, n_sims))
           for g in range(n_sims):
               randmap = [np.zeros(self.maps[i].rms.shape), np.zeros(self.maps[i].rms.shape)]
               for l in range(2):
                   if seed is not None:
                       np.random.seed(seed * (g + 1) * (l + 1) * feeds[l])
                   randmap[l] = np.random.randn(*self.maps[l].rms.shape) * self.maps[l].rms

               rms_xs[:, g] = tools.compute_cross_spec3d(
                   (randmap[0] * np.sqrt(wi*wj), randmap[1] * np.sqrt(wi*wj)),
                   self.k_bin_edges, dx=self.maps[i].dx, dy=self.maps[i].dy, dz=self.maps[i].dz)[0]
                 
           self.reverse_normalization(i,j) #go back to the previous state to normalize again with a different map-pair

           self.rms_xs_mean.append(np.mean(rms_xs, axis=1))
           self.rms_xs_std.append(np.std(rms_xs, axis=1))
        return self.rms_xs_mean, self.rms_xs_std
    
    #MAKE SEPARATE H5 FILE FOR EACH XS
    def make_h5(self,index=0, outname=None):
       
        for i in range(0,len(self.maps)-1, 2):
           j = i+1

           if outname is None:
               tools.ensure_dir_exists('spectra')
               outname = 'spectra/xs_' + self.get_information()[index][1] + '_and_'+ self.get_information()[index][2] + '.h5'          

           f1 = h5py.File(outname, 'w')
           try:
               f1.create_dataset('mappath1', data=self.maps[i].mappath)
               f1.create_dataset('mappath2', data=self.maps[j].mappath)
               f1.create_dataset('xs', data=self.xs[index])
               f1.create_dataset('k', data=self.k[index])
               f1.create_dataset('k_bin_edges', data=self.k_bin_edges)
               f1.create_dataset('nmodes', data=self.nmodes[index])
           except:
               print('No power spectrum calculated.')
               return 
           try:
               f1.create_dataset('rms_xs_mean', data=self.rms_xs_mean[index])
               f1.create_dataset('rms_xs_std', data=self.rms_xs_std[index])
           except:
               pass
                 
           f1.close()

    #PLOT XS
    def plot_xs(self, k_array, xs_array, rms_sig_array, rms_mean_array, index, save=False):
       
       k = k_array[index]
       xs = xs_array[index]
       rms_sig = rms_sig_array[index]
       rms_mean = rms_mean_array[index]
       
       #lim = 200.
       fig = plt.figure()
       fig.suptitle('xs of ' + self.get_information()[index][1] + ' and ' + self.get_information()[index][2])
       ax1 = fig.add_subplot(211)
       ax1.errorbar(k, k*xs, k*rms_sig, fmt='o', label=r'$k\tilde{C}_{data}(k)$') #added k*
       ax1.plot(k, 0 * rms_mean, 'k', label=r'$\tilde{C}_{noise}(k)$', alpha=0.4)
       ax1.plot(k, k*PS_function.PS_f(k), label='k*PS of the input signal')
       ax1.set_ylabel(r'$\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^3$]')
       
       lim = np.mean(np.abs(xs[4:])) * 4
       if not np.isnan(lim):
          ax1.set_ylim(-lim, lim)            

       ax1.set_xscale('log')
       ax1.grid()
       plt.legend()

       ax2 = fig.add_subplot(212)
       ax2.errorbar(k, xs / rms_sig, rms_sig / rms_sig, fmt='o', label=r'$\tilde{C}_{data}(k)$')
       ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
       ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
       ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]')
       ax2.set_ylim(-12, 12)
       ax2.set_xscale('log')
       ax2.grid()
       plt.legend()
       if save==True:
          tools.ensure_dir_exists('figures')
          name_for_figure = 'figures/xs_' + self.get_information()[index][1] + '_and_'+ self.get_information()[index][2] + '.pdf'
          plt.savefig(name_for_figure, bbox_inches='tight')
          print ('Figure saved as', name_for_figure)
       plt.close(fig)
       #plt.show()


