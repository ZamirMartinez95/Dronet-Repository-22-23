import numpy as np
from sim_functions import *

#####################################################
#            Analyze the results of the code        #
#                                                   #
#                                                   #
#####################################################

# Define the default parameters.
T0 = 25 # Environment temperature in [c].
gamma = 1.4
R = 287
T = 273.15 + T0
# speed of sound at T0
a = np.sqrt(gamma * R * T) 
# Sampling rate (in Hz)
fs = 51200  
# Window length (in samples)
win_l = 2500 
# Overlap percentage 
sig_overlap = 0.5 
initial_search_radius=15
res_factor=0.5
res_angle=10
num_points=13

mic_p_l = np.array([[1.52, 13.76, 0],
                        [0.8, 13.76, 0],
                        [1.525, 13.14, 0],
                        [0.94, 13.14, 0]])

# Microphone position of the right sided constellation:
mic_p_r = np.array([[5.7, 13.15, 0],
                        [5.61, 13.78, 0],
                        [6.47, 13.95, 0],
                        [6.44, 13.12, 0]])

# Define the desired noise generator places.
points_vec = np.array([[4, 1.88, 1.56],
                       [3.3, 2.4, 9.56],
                       [1.7, 6.44, 8.56],
                       [4, 6.62, 1.56],
                       [4, 10.77, 6.56],
                       [6.3, 8.73, 1.56],
                       [6.09, 5.92, 7.56]])

selected_params_names=['num_points',
                      'win_l',
                      'sig_overlap',
                      'initial_search_radius',
                      'res_factor',
                      'res_angle'
]
selected_params_values=[np.array(range(4,15)), np.arange(2500, 3500, 100), np.arange(0.1,1,0.1),
                      np.arange(1,26,1), np.arange(0.1,1,0.1), np.arange(0.1,90,10)]


default_data = {
    'num_points': num_points,
    'mic_p_l': mic_p_l,
    'mic_p_r': mic_p_r,
    'win_l': win_l,
    'sig_overlap': sig_overlap,
    'initial_search_radius': initial_search_radius,
    'res_factor': res_factor,
    'res_angle':res_angle,
    'a': a,
    'fs': fs
}

for i in range(0,len(selected_params_names)):
    curr_dictionary=default_data
    del curr_dictionary[selected_params_names[i]]
    curr_resdf=optimize_param(selected_params_values[i],selected_params_names[i],curr_dictionary,points_vec)
    # Save the results to a .csv file. 
    str1='Results - '
    str2='.csv'       
    curr_resdf.to_csv(str1+selected_params_names[i]+str2, index=False)
    # plot the results.
    # Error as a function of param_str (legend each location).
    curr_str='Error as a function of'+selected_params_names[i]
    png_str='.png'
    # Runtime as a function of param_str (legend each location).
    
    
    
    
    
    
    
    
