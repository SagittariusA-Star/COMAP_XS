import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

def pw():
   ps_low_res = np.load('ps_low_res.npy')
   ps_high_res = np.load('ps_high_res.npy')
   k = np.load('k_arr.npy')
   pw_tf = np.load('pixel_window.npy')
   k = k[0]

   #pixel_window = ps_low_res[0]/ps_high_res[0]
   pixel_window = pw_tf
   transfer_pixel_window = scipy.interpolate.interp2d(k[0], k[1], pixel_window)
  
   return transfer_pixel_window



