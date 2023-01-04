import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import corner
import h5py
import sys
import re
import scipy.interpolate
import time
from tqdm import tqdm 

import tools
import map_cosmo
import xs_class
import multiprocessing
import read_multisplit
import mean_multisplit

def process_params(param_file):
    """Function reading the parameter file provided by the command line
    argument or passed to class in __init__.
    """
    param_file  = open(param_file, "r")     
    params        = param_file.read()

    targetname = re.search(r"\nTARGET_NAME\s*=\s*'([0-9A-Za-z\_]*)'", params)    # Defining regex pattern to search for name of output map file.
    target_name = str(targetname.group(1))                          

    map_path = re.search(r"\nMAP_DIR\s*=\s*'(\/.*?)'", params) # Regex pattern to search for directory where to put the maps.
    map_path = str(map_path.group(1))                                  

    mapname = re.search(r"\nMAP_NAME\s*=\s*'([0-9A-Za-z\_]*)'", params)    # Defining regex pattern to search for name of output map file.
    map_name = map_path + target_name + "_" + str(mapname.group(1)) + ".h5"                          

    # jk_list file from parameter file
    split_def = re.search(r"\nJK_DEF_FILE\s*=\s*'(\/.*?)'", params)    # Defining regex pattern to search for complete path to split definition file.
    split_def = str(split_def.group(1))                                          
        
    signal_path = re.search(r"\nSIGNAL_PATH\s*=\s*'(\/.*?)'", params)    # Defining regex pattern to search for complete path to split definition file.
    
    if signal_path != None:
        signal_path = str(signal_path.group(1))
    
    param_file.close()

    print ('Field:', target_name)
    print ('List of split-variables:', split_def)

    return map_name, target_name, split_def, signal_path

def run_all_methods(feed1, feed2, n_of_splits, two_dimensions, jk, reproduce_spectra):
    #my_xs = xs_class.CrossSpectrum_nmaps(mapfile, jk, feed1, feed2, n_of_splits)
    my_xs = xs_class.CrossSpectrum_nmaps(mapfile, jk, feed1, feed2, n_of_splits)
    
    calculated_xs = my_xs.get_information() #gives the xs, k, rms_sig, rms_mean index with corresponding map-pair

    """if two_dimensions == False:
        #rms_mean, rms_sig = my_xs.run_noise_sims(50) #these rms's are arrays of 14 elements, that give error bars (number of bin edges minus 1)
        rms_mean_2D, rms_sig_2D = my_xs.run_noise_sims_2d(50) #these rms's are arrays of 14 elements, that give error bars (number of bin edges minus 1)
        #xs, k, nmodes = my_xs.calculate_xs()
        xs, k, nmodes, rms_mean, rms_sig = my_xs.calculate_xs_with_tf()
        np.save('knmodes.npy',np.array(k))
        np.save('nmodes.npy',np.array(nmodes))
      
        #plot all cross-spectra that have been calculated
        my_xs.plot_xs(k, xs, rms_sig, rms_mean, save=True, outdir = "")
        my_xs.make_h5(outdir)  
    """
    names = [my_xs.names[i] + "_X_" + my_xs.names[i + 1] for i in range(0, len(my_xs.names), 2)]

    if not spectra_from_file:
        #if two_dimensions == True:
        xs, k, nmodes = my_xs.calculate_xs_2d()
        #print("2D XS time:", time.time() - t0, "sec")

        rms_mean, rms_sig = my_xs.run_noise_sims_2d(50)                  # CHANGE TO 50 WHEN DONE DEVELOPING
        #print("Noise sim time:", time.time() - t0, "sec")
        my_xs.make_h5_2d(outdir)

        result = {
                "names": names[0],
                "xs_2D": xs,
                "k_bin_edges_par": my_xs.k_bin_edges_par,
                "k_bin_edges_perp": my_xs.k_bin_edges_perp,
                "k": k,
                "nmodes": nmodes,
                "rms_xs_mean_2D": rms_mean,
                "rms_xs_std_2D": rms_sig
                }

    else:
        result = {
             "names": names[0]
             }
        
    return result

def all_feed_combo_xs(p):
    i = p // 19 + 1 #floor division, divides and returns the integer value of the quotient (it dumps the digits after the decimal)
    j = p % 19 + 1 #modulus, divides and returns the value of the remainder
     
    #if i == 4 or i == 6 or i == 7: #avoid these feeds (were turned off for most of the mission)
    #     return p
    #if j == 4 or j == 6 or j == 7: #avoid these feeds (were turned off for most of the mission)
    #     return p

    spectra = run_all_methods(i, j, n_of_splits, two_dimensions, jk, spectra_from_file)
    return spectra

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
    return field_name, jk_list, map_name

def read_jk(single_map_name):
    jk_name = single_map_name.split('_')[2]
    return jk_name

def get_tf():
    #theory spectrum
    k_th = np.load('k.npy')
    ps_th = np.load('ps.npy')
    ps_th_nobeam = np.load('psn.npy') #instrumental beam, less sensitive to small scales line broadening, error bars go up at high k, something with the intrinsic resolution of the telescope (?)

    #in 2D
    #ps_2d_smooth = np.load('ps_2d_smooth.npy')
    #ps_2d_notsmooth = np.load('ps_2d_notsmooth.npy')
    #ps_2d_smooth = np.load('smooth_mean.npy')
    #ps_2d_notsmooth = np.load('notsmooth_mean.npy')
    #ps_2d_smooth = np.load('ps_smooth_single.npy') #'ps_2dfrom3d.npy'
    #ps_2d_notsmooth = np.load('ps_notsmooth_single.npy')

    tf_2d_beam = np.load("transfer_functions/beam_transfer_func_2D.npy")

    tf_2d_pix_win     = np.load("transfer_functions/pixel_window_v4.npy")
    tf_1d_pix_win     = np.load("transfer_functions/pixel_window_1D_v4.npy")
    k_1d_pix_win      = np.load('transfer_functions/k_arr_1D_v4.npy')[0, ...]
    k_smooth = np.load('k_smooth.npy')
    #k_notsmooth = np.load('k_notsmooth.npy')
    
    #print (ps_2d_smooth/ps_2d_notsmooth)

    k_perp_sim = k_smooth[0]
    k_par_sim = k_smooth[1]

    transfer_sim_2D = scipy.interpolate.interp2d(k_perp_sim, k_par_sim, tf_2d_beam)
    transfer_pix_2D = scipy.interpolate.interp2d(k_perp_sim, k_par_sim, tf_2d_pix_win)
    
    transfer_pix_1D = scipy.interpolate.interp1d(k_1d_pix_win, tf_1d_pix_win)

    
    #transfer_sim_2D = scipy.interpolate.interp2d(k_perp_sim, k_par_sim, ps_2d_smooth / ps_2d_notsmooth)
    #values from COPPS
    ps_copps = 8.746e3 * ps_th / ps_th_nobeam #shot noise level
    ps_copps_nobeam = 8.7e3

    transfer = scipy.interpolate.interp1d(k_th, ps_th / ps_th_nobeam) #transfer(k) always < 1, values at high k are even larger and std as well
    P_theory = scipy.interpolate.interp1d(k_th,ps_th_nobeam)

    #Read the transfer function associated with effects of filtering
    def filtering_TF(filename, dim):
        if dim == 1:
                with h5py.File(filename, mode="r") as my_file:
                    k = np.array(my_file['k'][:]) 
                    TF_1D = np.array(my_file['TF'][:]) 
                return k, TF_1D
        if dim == 2:
                with h5py.File(filename, mode="r") as my_file:
                    k_perp = np.array(my_file['k'][0]) 
                    k_par = np.array(my_file['k'][1]) 
                    TF_2D = np.array(my_file['TF'][:]) 
                return k_perp, k_par, TF_2D

    k_filtering_1D, TF_filtering_1D = filtering_TF('transfer_functions/TF_1d_new_pca.h5', 1)
    transfer_filt = scipy.interpolate.interp1d(k_filtering_1D, TF_filtering_1D) 
 
    k_perp_filt, k_par_filt, TF_filtering_2D = filtering_TF('transfer_functions/TF_2d_new_pca.h5', 2)
    transfer_filt_2D = scipy.interpolate.interp2d(k_perp_filt, k_par_filt, TF_filtering_2D)

    k_perp_filt, k_par_filt, TF_filtering_2D_ces = filtering_TF('transfer_functions/TF_2d_new_pca.h5', 2)
    transfer_filt_2D_ces = scipy.interpolate.interp2d(k_perp_filt, k_par_filt, TF_filtering_2D_ces)

    k_perp_filt, k_par_filt, TF_filtering_2D_liss = filtering_TF('transfer_functions/2D_TF_Liss.h5', 2)
    transfer_filt_2D_liss = scipy.interpolate.interp2d(k_perp_filt, k_par_filt, TF_filtering_2D_liss)

    return transfer, transfer_filt, transfer_pix_1D, transfer_sim_2D, transfer_filt_2D, transfer_filt_2D_ces, transfer_filt_2D_liss, transfer_pix_2D 


if __name__ == "__main__":
    t0 = time.time()
    #read from the command:
    mappath, field, jk_list, signal_path = process_params(sys.argv[1])

    #mappath_last_part = sys.argv[-1]
    #mappath = '/mn/stornext/d16/cmbco/comap/protodir/maps/' + mappath_last_part 
    if len(sys.argv) > 2:
        spectra_from_file = sys.argv[2] == "-f"
    else:
        spectra_from_file = False

    #xs_2D = input("Cross-spectra in two dimensions? (yes/no) ")
    #if xs_2D == 'yes':
    two_dimensions = False
    #if xs_2D == 'no':
    #    two_dimensions = False

    print ('The following directories will be created:')
    print ('- split_maps - the HDF5 map-files split according to all combinations of variables')
    if two_dimensions == True:
        print ('- spectra_2D - the HDF5 files with 2D cross-spectra for all split-split and feed-feed combinations')
    if two_dimensions == False:
        print ('- spectra - the HDF5 files with cross-spectra for all split-split and feed-feed combinations')
        print ('- xs_figures - figures of cross-spectra for all split-split and feed-feed combinations')
        print ('- chi2_grids - chi2 grids for all split-split combinations')
        print ('- xs_mean_figures - figures of mean cross-spectra for each combination of variables')

    #field, jk_list, main_map_name = read_field_jklist(mappath)
    main_map_name = mappath.split('/')[-1] #get rid of the path, leave only the name of the map
    main_map_name = main_map_name.split('.')[0] #get rid of the ".h5" part
        

    outdir = main_map_name

    #this jk list was an exeption from the naming convention!
    #jk_list = '/mn/stornext/d16/cmbco/comap/protodir/auxiliary/jk_list_' + 'science' + '.txt'
    control_variables, test_variables, feed_feed_variables, all_variables, feed_and_test, feed_and_control = read_multisplit.read_jk(jk_list)


    print(control_variables)
    print(test_variables)
    print(feed_feed_variables)
    print(all_variables)
    print(feed_and_test)
    print(feed_and_control)


    # maps = read_multisplit.read_map(mappath, control_variables, test_variables, all_variables, feed_and_test, feed_and_control)
    map_files = read_multisplit.read_map(mappath, field, control_variables, test_variables, feed_feed_variables, all_variables, feed_and_test, feed_and_control)
    #Perform null-test
    #new_subtracted_maps = read_multisplit.null_test_subtract(map_files, test_variables, field, outdir)
    #map_files = new_subtracted_maps
    #outdir += "_subtr"
    print(map_files)
    number_of_maps = len(map_files)
    number_of_ff_variables = len(feed_feed_variables)
    maps_per_jk = int(number_of_maps / number_of_ff_variables)
    #feed_combos = list(range(19*19)) #number of combinations between feeds
    Nfeed = 19
    Nproc = 128
    Ncombos = Nfeed ** 2
    feed_combos = list(range(Ncombos)) #number of combinations between feeds
    all_spectra = []
    print ('STAGE 3/4: Calculating cross-spectra for all split-split feed-feed combinations.')
    for g in tqdm(range(number_of_maps)):
    # for g in range(1):
        mapname = mappath#map_files[g]
        #jk = read_jk(mapname)
        jk = map_files[g]
        #print ('Split for FPXS: ' + jk + '. Map: ' + mapname)
        #mapfile = 'split_maps/' + outdir + '/' + mapname
        # n_of_splits = read_number_of_splits(mapfile, jk)
        mapfile = mapname    
        ###############################################################  
        ############# CHANGE LATER IF MORE THAN TWO SPLITS ############
        n_of_splits = 2
        ###############################################################  
        ###############################################################  
        
        # Make xs for all feed-combinations
        
        pool = multiprocessing.Pool(Nproc) # here number of cores
        #for feed_combo in feed_combos:
        #     all_feed_combo_xs(feed_combo)
        result = np.array(pool.map(all_feed_combo_xs, feed_combos))
        all_spectra.append(result.reshape(Nfeed, Nfeed))



    print ('STAGE 4/4: Calculating the mean of cross-spectra from all combinations.')
    k_arr_1d = []
    xs_mean_arr_1d = []
    xs_sigma_arr_1d = []

    field_arr = []
    ff_jk_arr = []
    split_names_arr = []
    split_numbers_arr = []
    figure_names = []

    k_arr_2d = []
    xs_mean_arr_2d = []
    xs_sigma_arr_2d = []

    k_edges_perp = []
    k_edges_par = []

    for mn in range(number_of_maps): 
    # for mn in range(1): 
        if two_dimensions == True:
            k, k_bin_edges_par, k_bin_edges_perp, xs_mean, xs_sigma, ff_jk, split_names, split_numbers = mean_multisplit.xs_feed_feed_2D(map_files[mn], outdir, all_spectra[mn])
            k_edges_perp.append(k_bin_edges_perp)
            k_edges_par.append(k_bin_edges_par)
        
        if two_dimensions == False:
            #k, xs_mean, xs_sigma, field, ff_jk, split_names, split_numbers = mean_multisplit.xs_feed_feed_grid(map_files[mn], outdir) #saves the chi2 grid for each split-combo
            #k, k_bin_edges_par, k_bin_edges_perp, xs_mean, xs_sigma, field, ff_jk, split_names, split_numbers = mean_multisplit.xs_feed_feed_2D(map_files[mn], outdir)
            k_2d, k_bin_edges_par, k_bin_edges_perp, xs_mean_2d, xs_sigma_2d, ff_jk, split_names, split_numbers = mean_multisplit.xs_feed_feed_2D(map_files[mn], outdir, all_spectra[mn], spectra_from_file)
            k_edges_perp.append(k_bin_edges_perp)
            k_edges_par.append(k_bin_edges_par)

            k_arr_2d.append(k_2d)
            xs_mean_arr_2d.append(xs_mean_2d)
            xs_sigma_arr_2d.append(xs_sigma_2d)



            tfs = get_tf()
            print(split_numbers)

            #if "0" in split_numbers[0]:
            #    tf_beam2d, tf_filt2d, tf_pix_win2d = tfs[3], tfs[6], tfs[7]
            #else:
            tf_beam2d, tf_filt2d, tf_pix_win2d = tfs[3], tfs[5], tfs[7] # We only use CES after S1
            
            kx, ky = k_2d
            k_bin_edges = np.logspace(-2.0, np.log10(1.5), len(kx) + 1)
            #k_bin_edges = k[1:] - k[:-1]
            #print(k_bin_edges)
            weights = 1 / (xs_sigma_2d / (tf_beam2d(kx, ky) * tf_filt2d(kx, ky) * tf_pix_win2d(kx, ky))) ** 2

            xs_mean_1d = xs_mean_2d.copy()
            xs_mean_1d /= (tf_beam2d(kx, ky) * tf_filt2d(kx, ky) * tf_pix_win2d(kx, ky))
            xs_mean_1d *= weights
            
            kgrid = np.sqrt(sum(ki ** 2 for ki in np.meshgrid(kx, ky, indexing='ij')))

            Ck_nmodes_1d        = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges, weights=xs_mean_1d[kgrid > 0])[0]
            inv_var_nmodes_1d   = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges, weights=weights[kgrid > 0])[0]
            nmodes_1d           = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges)[0]

            # Ck = Ck_nmodes / nmodes
            k_1d = (k_bin_edges[1:] + k_bin_edges[:-1]) / 2.0
            Ck_1d = np.zeros_like(k_1d)
            rms_1d = np.zeros_like(k_1d)
            Ck_1d[np.where(nmodes_1d > 0)] = Ck_nmodes_1d[np.where(nmodes_1d > 0)] / inv_var_nmodes_1d[np.where(nmodes_1d > 0)]
            rms_1d[np.where(nmodes_1d > 0)] = np.sqrt(1 / inv_var_nmodes_1d[np.where(nmodes_1d > 0)]) 

            xs_mean_1d = Ck_1d 
            xs_sigma_1d = rms_1d

        k_arr_1d.append(k_1d)
        xs_mean_arr_1d.append(xs_mean_1d)
        xs_sigma_arr_1d.append(xs_sigma_1d)
        ff_jk_arr.append(ff_jk)
    
        field_arr.append(field)
        split_names_arr.append(split_names)
        split_numbers_arr.append(split_numbers)

    how_many_different_splits = len(split_names)

    #group maps with respect to scanning strategy
    #index_cesc = split_names_arr[0].index('cesc')

    # plot xs mean
    for mn in range(number_of_maps):
    # for mn in range(1): 

        last_name_part = '_'
        other = ' '
        for ds in range(how_many_different_splits):
            if ds != how_many_different_splits - 1:
                last_part = split_names_arr[mn][ds] + str(split_numbers_arr[mn][ds]) + '_'
                other_part = split_names_arr[mn][ds] + ' ' + str(split_numbers_arr[mn][ds]) + ', '
            if ds == how_many_different_splits - 1:
                last_part = split_names_arr[mn][ds] + str(split_numbers_arr[mn][ds]) 
                other_part = split_names_arr[mn][ds] + ' ' + str(split_numbers_arr[mn][ds]) 
            last_name_part += last_part
            other += other_part
        
        figure_title = 'Field: ' + field_arr[mn] + '; Feed-feed variable: ' + ff_jk_arr[mn] + '; Other splits:' + other
        
        #if split_numbers_arr[mn][index_cesc] == '0': #cesc=0
        #    scan_strategy = 'liss'
        #if split_numbers_arr[mn][index_cesc] == '1': #cesc=0
        scan_strategy = 'ces'   # Since we only use CES after S1

        #if two_dimensions == False:
        figure_name = 'xs_mean_' + field_arr[mn] + '_map_' + ff_jk_arr[mn] + last_name_part + '.pdf'
        figure_names.append(figure_name)
        print ('Saving the figure ' + figure_name) #Saving the figure xs_mean_co6_map_snup_elev0_cesc0.pdf
        print("hei", k_arr_1d[mn], xs_mean_arr_1d[mn], xs_sigma_arr_1d[mn])
        mean_multisplit.xs_with_model(figure_name, k_arr_1d[mn], xs_mean_arr_1d[mn], xs_sigma_arr_1d[mn], figure_title, scan_strategy, outdir, signal_path)

        #if two_dimensions == True:
        figure_name = 'xs_mean_2D_' + field_arr[mn] + '_map_' + ff_jk_arr[mn] + last_name_part + '.pdf'
        figure_names.append(figure_name)
        print ('Saving the figure ' + figure_name)
        mean_multisplit.xs_2D_plot(figure_name, k_arr_2d[mn],k_edges_par[mn], k_edges_perp[mn], xs_mean_arr_2d[mn], xs_sigma_arr_2d[mn], figure_title, outdir)


    figure_names = np.array(figure_names, dtype  = "S")
    #save arrays as a file
    #if two_dimensions == True:
    outname = main_map_name + '_arrays.h5'
    #     npyname_ = main_map_name + '_2D_names.npy'
    # if two_dimensions == False:
    #     outname = main_map_name + '_1D_arrays.h5'
    #     npyname = main_map_name + '_1D_names.npy'

    outname = outdir + "/" + outname
    #npyname = outdir + "/" + npyname
    tools.ensure_dir_exists(outdir)

    print ('Saving data in ' + outname + '.')
    f = h5py.File(outname, 'w') #create HDF5 file with the sliced map
    f.create_dataset('k_1d', data=k_arr_1d)
    f.create_dataset('xs_mean_1d', data=xs_mean_arr_1d)
    f.create_dataset('xs_sigma_1d', data=xs_sigma_arr_1d)
    # if two_dimensions == True:
    f.create_dataset('k_2d', data=k_arr_2d)
    f.create_dataset('xs_mean_2d', data=xs_mean_arr_2d)
    f.create_dataset('xs_sigma_2d', data=xs_sigma_arr_2d)

    f.create_dataset('k_edges_perp', data=k_edges_perp)
    f.create_dataset('k_edges_par', data=k_edges_par)
    f.close()    

    # print ('Saving names of sub-maps in ' + npyname + '.')
    # np.save(npyname,np.array(figure_names))

    print("Total run time:", time.time() -  t0, "sec")
