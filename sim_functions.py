import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
import math
import time

def sample_sim(mic_x, mic_y, mic_z, pos_x, pos_y, pos_z,a,fs):
    """
    Create sample simulation with the microphone location, source position, 
    sampling frequency, speed of sound and frequency.
    
    @param mic_x: (float)  microphone location in x axis. [m]
    @param mic_y: (float)  microphone location in y axis. [m]
    @param mic_z: (float) microphone location in z axis. [m]
    @param pos_x: (float) microphone location in x axis. [m]
    @param pos_x: (float) microphone location in y axis. [m]
    @param pos_x: (float) microphone location in z axis. [m]
    @param a: (float) speed of sound. [m/sec]
    @param fs: (float) sample rate. [Hz]

    """

    
    # Define the duration of the simulation [Sec]
    duration = 2  
    
    # Define the source position (x,y,z) [m]
    source_pos = np.array([pos_x, pos_y, pos_z])
    
    # Define the microphone position (x,y,z) [m]
    mic_pos = np.array([mic_x, mic_y, mic_z]) 
    
    # Pass the speed of sound in air [m/s]
    speed_of_sound = a
    
    # Set the frequency of the sine wave in [Hz]
    freq = 260
     
    # Create the time vector that the microphone would feel.
    t = np.arange(0, duration, 1/fs)
    
    # Calculate the distance between the source and the microphone.
    distance = np.sqrt(np.sum((source_pos - mic_pos)**2))
    
    # Define the desired amplitude
    amplitude=5
    
    # Calculate the time delay due to the distance between the source and the receiver.
    time_delay = distance / speed_of_sound
    
    # Simulate what the microphone would hear at its' position.
    mic_signal = amplitude * np.sin(2 * np.pi * freq * (t+time_delay))

    
    # Return the result.
    return mic_signal


def intersection(x1, y1, z1, A1, B1, x2, y2, z2, A2, B2):
    """
    This function computes the intersection of two location vectors in (x,y,z) using their azimuth and elevation.

    @param x1: (float) first vector x location.
    @param y1: (float) first vector x location.
    @param z1: (float) first vector x location.
    @param A1: (float) first vector azimuth.
    @param B1: (float) first vector elevation.
    @param x2: (float) second vector x location.
    @param y2: (float) second vector y location.
    @param z2: (float) second vector z location.
    @param A2: (float) second vector azimuth.
    @param B2: (float) second vector elevation.
        
    x_intersect, y_intersect, z_intersect are the calculated output point of intersection.
    """
    
    m1 = np.tan(A1)
    m2 = np.tan(A2)
    
    # Check if the lines are parallel
    if m1 == m2:
        x_intersect = 0
        y_intersect = 0
        z_intersect = 0
        return x_intersect, y_intersect, z_intersect
    
    # Compute the intersection point
    x_intersect = (y2 - y1 + m1*x1 - m2*x2) / (m1 - m2)
    y_intersect = m1*(x_intersect - x1) + y1
    
    alt1 = z1 + np.sqrt((x1 - x_intersect)**2 + (y1 - y_intersect)**2)*np.tan(B1)
    alt2 = z2 + np.sqrt((x2 - x_intersect)**2 + (y2 - y_intersect)**2)*np.tan(B2)
    z_intersect = np.mean([alt1, alt2])
    
    # Return the result.
    return x_intersect, y_intersect, z_intersect





def beam_forming(mic_p, windowed_signals, search_radius, azim_curr, elev_curr,res_factor,res_angle,flag,fs, win_l, num_points, a):
    """
    Find the direction using the beamforming idea. 
    Return the calculated direction.
    
    @param mic_p: (np.array, 1X3) The constellation center of four square microphones, in [m]. 
    @param windowed_signals: (float) an array of a sliced signals. 
                            The user chooses in the main program how many samples of the signal to send.
    @param search_radius: (float) The Radius of the sphere of interest. 
    @param azim_curr: (float) The current azimuth angle.
    @param elev_curr: (float) The current elevation angle.
    @param res_factor: (float) a float between 0 and 1 that tells by a factor of how much to create the finer mesh using num_points.
    @param flag: (float) a float that tells whether it is the first time calling the function.
    @param fs: (float) the sampling rate in [Hz].
    @param win_l: (float) the sliced window signal in samples.
    @param num_points: (float) the square root of the total number of point that will be in the dome of microphones.
    @param a: (float) speed of sound. [m/sec]
    
    """

    # Adjusting the microphone center to be the middle of the given constellation.
    mic_c = np.mean(mic_p, axis=0)

    # Define alpha - Angle of mesh tuning in [rad].
    alpha = res_angle*np.pi/180
    
    # Create a grid of points of interest.
    # First we initialize the size of the azimuth and elevation of interest. In case of convergence we will build smaller yet finer mesh.
    if flag == False:
        # Create the Azimuth and Elevation grids.
        Azimuth, Elevation = np.meshgrid(np.linspace(0, 2 * np.pi, num_points), np.linspace(0, np.pi/2, num_points))

    else:
        Azimuth, Elevation = np.meshgrid(np.linspace(azim_curr - alpha, azim_curr + alpha, round(res_factor*num_points)), 
                                         np.linspace(elev_curr - alpha, elev_curr + alpha, round(res_factor*num_points))) 

    # Convert the grid points to be 3D using the azimuth, elevation and the search radius.
    grid_positions = mic_c + np.column_stack((search_radius * np.cos(Azimuth.flatten()) * np.cos(Elevation.flatten()),
                                              search_radius * np.sin(Azimuth.flatten()) * np.cos(Elevation.flatten()),
                                              search_radius * np.sin(Elevation.flatten())))
    
    
    # Initialize output energy array
    E = np.zeros((len(grid_positions), 1))

    # Time delays matrix initialization
    time_delays = np.zeros((mic_p.shape[0], grid_positions.shape[0]))

    # Time delays matrix initialization
    true_corr_arr = []

    # Calculation of time delays for each microphone and each grid position
    for mic_idx in range(len(mic_p)):
        for pos_idx in range(len(grid_positions)):
            distance = np.linalg.norm(grid_positions[pos_idx, :] - mic_p[mic_idx, :])
            time_delays[mic_idx, pos_idx] = distance / a

    for mic_idx1 in range(len(mic_p)):
                for mic_idx2 in range(mic_idx1, len(mic_p)):
                    if mic_idx1 != mic_idx2:
                        correlation = correlate(windowed_signals[mic_idx1, :], windowed_signals[mic_idx2, :], mode='full', method='fft')
                        true_corr_arr.append(correlation)
                                             
    true_corr_arr = np.array(true_corr_arr)


    # Convert time delays to sample delays
    sample_delays = np.round(time_delays * fs)

    for pos_idx in range(len(grid_positions)):
        cross_corr_sum = 0

        j = 0 
        for mic_idx1 in range(len(mic_p)):
            for mic_idx2 in range(mic_idx1, len(mic_p)):
                if mic_idx1 != mic_idx2:
                    delay_diff = int(sample_delays[mic_idx1, pos_idx] - sample_delays[mic_idx2, pos_idx])
                    cross_corr = true_corr_arr[j]
                    cross_corr_sum += cross_corr[win_l + delay_diff]
                    j+=1

        E[pos_idx] = cross_corr_sum

    # Find the grid position with the maximum output energy
    max_idx = np.argmax(np.sum(E, axis=1))
    estimated_position = grid_positions[max_idx, :]

    # Calculate the estimated direction vector
    direction_vector = estimated_position - mic_c

    # Calculate the estimated azimuth
    estimated_azimuth = np.arctan2(direction_vector[1], direction_vector[0])

    # Calculate the estimated elevation
    estimated_elevation = np.arcsin(direction_vector[2] / search_radius)
    
    # Return the result.
    return estimated_azimuth, estimated_elevation


def get_location(num_points,mic_p_l,mic_p_r,Source_location,win_l,sig_overlap,initial_search_radius,res_factor,res_angle,a,fs):
    
    """
    This documentation is yet to be verified!!!!!
    
    Gets the location of the source of sound.
    Parameters:
    num_points : int
    Number of points to use in the search.
    mic_p_l : array_like
    Coordinates of the left microphone.
    mic_p_r : array_like
    Coordinates of the right microphone.
    Source_location : array_like
    Initial guess for the location of the source.
    win_l : int
    Length of the left window.
    sif_overlap : float
    Percentage of overlap between windows.
    initial_search_radius : float
    Initial radius of the search area.
    a : float
    fs : float
    Sampling frequency.
    Returns

    location : array_like
    Location of the source.
    """

    # Beamforming test, we assume a radius large enough to read the signals as a
    # wave, and linear propagation
    


    # Calculate the center of the two constellations.
    mic_c_l = np.mean(mic_p_l, axis=0)
    mic_c_r = np.mean(mic_p_r, axis=0)


    # Simulation of real-time input of a sinusoidal wave from a certain source to the 2 constellations.
    sim_mic_l1 = sample_sim(mic_p_l[0,0], mic_p_l[0,1], mic_p_l[0,2], Source_location[0],
                            Source_location[1], Source_location[2],a,fs)
    sim_mic_l2 = sample_sim(mic_p_l[1,0], mic_p_l[1,1], mic_p_l[1,2], Source_location[0],
                            Source_location[1], Source_location[2],a,fs)
    sim_mic_l3 = sample_sim(mic_p_l[2,0], mic_p_l[2,1], mic_p_l[2,2], Source_location[0],
                            Source_location[1], Source_location[2],a,fs)
    sim_mic_l4 = sample_sim(mic_p_l[3,0], mic_p_l[3,1], mic_p_l[3,2], Source_location[0],
                            Source_location[1], Source_location[2],a,fs)
    sim_mic_r1 = sample_sim(mic_p_r[0,0], mic_p_r[0,1], mic_p_r[0,2], Source_location[0],
                            Source_location[1], Source_location[2],a,fs)
    sim_mic_r2 = sample_sim(mic_p_r[1,0], mic_p_r[1,1], mic_p_r[1,2], Source_location[0],
                            Source_location[1], Source_location[2],a,fs)
    sim_mic_r3 = sample_sim(mic_p_r[2,0], mic_p_r[2,1], mic_p_r[2,2], Source_location[0],
                            Source_location[1], Source_location[2],a,fs)
    sim_mic_r4 = sample_sim(mic_p_r[3,0], mic_p_r[3,1], mic_p_r[3,2], Source_location[0],
                            Source_location[1], Source_location[2],a,fs)


    azim_l_list = []
    elev_l_list = []
    azim_r_list = []
    elev_r_list = []

    search_radius_l = initial_search_radius
    search_radius_r = initial_search_radius
    azim_l_curr = 0
    elev_l_curr = 0
    azim_r_curr = 0
    elev_r_curr = 0
    Resolution_flag = False
    
    overlap=int(np.round(win_l * sig_overlap))

    j = 0
    for i in range(0, len(sim_mic_l1) - win_l + 1, overlap):
        
        # Cut the signals
        s_arr_l = np.zeros((4, win_l))
        s_arr_r = np.zeros((4, win_l))
        
        s_arr_l[0, :] = sim_mic_l1[i:i+win_l]
        s_arr_l[1, :] = sim_mic_l2[i:i+win_l]
        s_arr_l[2, :] = sim_mic_l3[i:i+win_l]
        s_arr_l[3, :] = sim_mic_l4[i:i+win_l]
        s_arr_r[0, :] = sim_mic_r1[i:i+win_l]
        s_arr_r[1, :] = sim_mic_r2[i:i+win_l]
        s_arr_r[2, :] = sim_mic_r3[i:i+win_l]
        s_arr_r[3, :] = sim_mic_r4[i:i+win_l]
        azim_l, elev_l = beam_forming(mic_p_l, s_arr_l, search_radius_l, azim_l_curr,
                                      elev_l_curr,res_factor, Resolution_flag,res_angle,fs, win_l, num_points, a)
        azim_r, elev_r = beam_forming(mic_p_r, s_arr_r, search_radius_r, azim_r_curr,
                                      elev_r_curr,res_factor, Resolution_flag,res_angle,fs, win_l, num_points, a)

        if len(azim_l_list) >= 5:
            azim_l_list.pop(0)
            elev_l_list.pop(0)
            azim_r_list.pop(0)
            elev_r_list.pop(0)

        azim_l_list.append(azim_l)
        elev_l_list.append(elev_l)
        azim_r_list.append(azim_r)
        elev_r_list.append(elev_r)
        
        p_x, p_y, p_z = intersection(mic_c_l[0], mic_c_l[1], mic_c_l[2], azim_l, elev_l,
                                    mic_c_r[0], mic_c_r[1], mic_c_r[2], azim_r, elev_r)
        
        # Check if elements are within 10% error
        if all(abs(azim - azim_l) <= 0.5 * abs(azim_l) for azim in azim_l_list) and all(
            abs(elev - elev_l) <= 0.5 * abs(elev_l) for elev in elev_l_list) and all(
            abs(azim - azim_r) <= 0.5 * abs(azim_r) for azim in azim_r_list) and all(
            abs(elev - elev_r) <= 0.5 * abs(elev_r) for elev in elev_r_list) and j > 5:
            Resolution_flag = True
            search_radius_l = math.sqrt((p_x - mic_c_l[0])**2 + (p_y - mic_c_l[1])**2 + (p_z - mic_c_l[2])**2)
            search_radius_r = math.sqrt((p_x - mic_c_r[0])**2 + (p_y - mic_c_r[1])**2 + (p_z - mic_c_r[2])**2)


        if Resolution_flag:
                azim_l_curr = azim_l
                elev_l_curr = elev_l
                azim_r_curr = azim_r
                elev_r_curr = elev_r
        
        err = math.sqrt((p_x - Source_location[0])**2 + (p_y - Source_location[1])**2 + (p_z - Source_location[2])**2)

        j += 1
    
    
        
    # Get the current location of the source.
    curr_true_vec_loc = Source_location

    # Get the estimated location of the source.
    estimated_loc = np.array([p_x, p_y, p_z])

    # Calculate the vector from the current true location to the right microphone.
    curr_true_vec_r = curr_true_vec_loc - mic_c_r

    # Calculate the vector from the current true location to the left microphone.
    curr_true_vec_l = curr_true_vec_loc - mic_c_l

    # Calculate the vector from the estimated location to the right microphone.
    curr_est_vec_r = estimated_loc - mic_c_r

    # Calculate the vector from the estimated location to the left microphone.
    curr_est_vec_l = estimated_loc - mic_c_l

    # Calculate the dot product of the right constellation vectors.
    dot_prod_r = np.dot(curr_est_vec_r, curr_true_vec_r)

    # Calculate the dot product of the left constellation vectors.
    dot_prod_l = np.dot(curr_est_vec_l, curr_true_vec_l)

    # Calculate the norm of the right constellation vectors.
    norm_r = np.linalg.norm(curr_true_vec_r) * np.linalg.norm(curr_est_vec_r)

    # Calculate the norm of the left constellation vectors.
    norm_l = np.linalg.norm(curr_true_vec_l) * np.linalg.norm(curr_est_vec_l)

    # Calculate the angle between the right vectors in degrees.
    err_deg_r = np.arccos(dot_prod_r / norm_r) * (180 / np.pi)

    # Calculate the angle between the left vectors in degrees.
    err_deg_l = np.arccos(dot_prod_l / norm_l) * (180 / np.pi)

    # Calculate the total error using norm.
    err_length = np.linalg.norm(estimated_loc - curr_true_vec_loc)
    
    # output = f"-----------------------------------------------------------------------------------------------\nThe current Number of points: {num_points*num_points} \nTrue Location: {curr_true_vec_loc} \nEstimated location: {estimated_loc} \nError norm {err_length}[M] \nError, left constellation: {err_deg_l}[deg] \nError, right constellation: {err_deg_r}[deg]"
    # print(output)
    
    # Return the data.
    return [curr_true_vec_loc,estimated_loc,norm_r,norm_l,err_deg_r,err_deg_l,err_length]



def optimize_param(param_arr,param_str,data,points_vec):
    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame()
    print("Now checking the affect of "+param_str)
    
    for i in range(0,len(param_arr)):
        for j in range(0,len(points_vec)):
            data[param_str],data['Source_location']= param_arr[i] , points_vec[j]
            start_time = time.time()   
            result=get_location(**data)
            end_time = time.time()
            elapsed_time = end_time - start_time
            output=f"Elapsed time, testing {param_str+'='+str(data[param_str])} for point {j+1}: {elapsed_time} Seconds"
            print(output)
            
            # Store the results in the DataFrame
            result_dictionary = {
                param_str: data[param_str],
                'Current True location [m]': [np.array(result[0])],
                'Estimated location [m]': [np.array(result[1])],
                'Error norm, right constellation [m] ': result[2],
                'Error norm, left constellation [m]': result[3],
                'Error, right constellation [deg]': result[4],
                'Error, left constellation [deg]': result[5],
                'Error norm [m]': result[6],
                'Runtime [Sec]': elapsed_time
            }
            
            curr_dataframe=pd.DataFrame(result_dictionary)
            results_df = pd.concat([results_df, curr_dataframe], ignore_index=True)

    return results_df
