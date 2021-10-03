import numpy as np 
import h5py
import sys 

def swap_axis(idx_in):
    binary = int(np.binary_repr(idx_in))
    binary = f"{binary:03d}"
    binary = binary[0] + binary[1:][::-1]
    idx_out = 0
    binary = binary[::-1]
    for i in range(len(binary)):        
        idx_out += int(binary[i]) * 2 ** i
    return idx_out
    
#path = "/mn/stornext/d16/cmbco/comap/protodir/maps/co6_map_summer_null_debug2.h5"
path = "/mn/stornext/d16/cmbco/comap/protodir/maps/co2_map_summer_null_backup.h5"

with h5py.File(path, "r+") as file:
    for key in file["multisplits"].keys():    
        if "map" in key:
            print("multisplits/" + key)
            map_data = file["multisplits/" + key]
            map = map_data[()]
            
            map_buffer = np.zeros_like(map)
            for i in range(map.shape[0]):
                idx_out = swap_axis(i)
                map_buffer[idx_out, ...] = map[i, ...]
            map_data[...] = map_buffer

        elif "rms" in key:
            rms_data = file["multisplits/" + key]
            rms = rms_data[()]
            rms_buffer = np.zeros_like(rms)
            for i in range(rms.shape[0]):
                idx_out = swap_axis(i)
                rms_buffer[idx_out, ...] = rms[i, ...]
            
            rms_data[...] = rms_buffer
            print("multisplits/" + key)

        elif "nhit" in key:
            nhit_data = file["multisplits/" + key]
            nhit = nhit_data[()]
            nhit_buffer = np.zeros_like(nhit)
            for i in range(nhit.shape[0]):
                idx_out = swap_axis(i)
                nhit_buffer[idx_out, ...] = nhit[i, ...]
            
            nhit_data[...] = nhit_buffer
            
            print("multisplits/" + key)
"""        
nhit_buffer = np.zeros_like(nhit)
rms_buffer = np.zeros_like(rms)
    
for i in range(7):
    idx_out = swap_axis(i)
    map_buffer[idx_out, ...] = map[i, ...]
    nhit_buffer[idx_out, ...] = nhit[i, ...]
    rms_buffer[idx_out, ...] = rms[i, ...]
    
    print("\n", idx_out)
map  = map_buffer
nhit = nhit_buffer
rms  = rms_buffer
"""