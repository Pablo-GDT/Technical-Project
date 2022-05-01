import sys
from tkinter import Y
import Square_bursting_oscillations as sb
import spectrum as sp
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from numpy.core.fromnumeric import std
import os
import scipy.io, scipy.signal
import copy
from scipy.io import loadmat
# import colorednoise as cn

sys.path.insert(0, "torbness_vimeapy")
try:
    from torbness_vimeapy import cython_funcs, MoI
except ImportError:
    print('No Import')

def point_like_potential(pos_vec: np.array, I : float, homog_elec_cond: float):
    """Calculates the potential given by a current point source, I, positioned at (u=w=v=0) at a given position. 
    Assumes an infinite homogeneous electrical conductor with conductivity.

    Args:
        pos_vec (np.array): Position at which to evaluate the potential given 3D coordinates [x, y, z]
        I (float): The strength of the current eminating from the point source
        homog_elec_cond (float): The conductivity of the homogeneous electrical conductor e.g brain tissue

    Returns:
        [float]: potential at some point of interest
    """
    euclid_dist = np.linalg.norm(pos_vec)
    pos_potential = I/ (4 * math.pi * homog_elec_cond * euclid_dist)
    return pos_potential

def assert_is_zero(val):
    assert val == 0, "A necessary value is not zero: {}.".format(val)

def assert_is_positive(val):
    assert val > 0

def summation_terms(displacement_vec:np.array, source_vec: np.array, W_ts: float, n: int, h: float, I: np.array, sigma_t: float):
    """Calculates individual terms in equation 8 given by the paper by Barrera et al DOI: 10.1007/s12021-015-9265-6. 
    Assumes negligible conductivity of the glass electrode plate.

    Args:
        displacement_vec (np.array): The displacement vector from potential source to electrode [x, y, z]
        source_vec (np.array): The position vector of the potential source [x, y, z]
        W_ts (float): a combined conductivity measure of neural tissue and glass electrode plate
        n (int): The index of the series
        h (float): The height of brain slice region in the z direction 
        I (np.array): The strength of the current eminating from the point source
        sigma_t (float): conductivity of the neural tissue

    Returns:
        [float]: nth term in the series
    """
    W_ts_n = np.power(W_ts, n)

    pos_z_vec = [displacement_vec[0], displacement_vec[1], 2*n*h - source_vec[2]]
    neg_z_vec = [displacement_vec[0], displacement_vec[1], - source_vec[2] - 2*n*h]
    pos_z_term = point_like_potential(pos_z_vec, I, sigma_t) 
    neg_z_term = point_like_potential(neg_z_vec, I, sigma_t)
    n_term = W_ts_n*(pos_z_term + neg_z_term)
    return n_term

def MEA_point_source(pos_vec: np.array, source_vec: np.array, I: np.array, sigma_t: float = 0.366, sigma_s: float = 1.408, h: float = 200, N_max: int = 20):
    """Calculates the MEA potential recorded at a plate electrode located at z=0 from a point source current positioned at source_vec in equation 8 
    given by DOI: 10.1007/s12021-015-9265-6. 

    Args:
        pos_vec (np.array): The position vector of the glass electrode [x, y, 0]
        source_vec (np.array): The position vector of the potential source [x, y, z]
        I (np.array): The strength of the current eminating from the point source
        sigma_t (float): conductivity of the neural tissue (0.2-0.6 S/m  see page 404 of Ness et al DOI: 10.1007/s12021-015-9265-6.)
        sigma_s (float): conductivity of the saline solution
        h (float): The height of brain slice region in the z direction (e.g 300 micro_u m)
        N_max (int, optional): The number of term to consider in the series. Defaults to 30.

    Returns:
        [np.array]: Returns the potential for every value of current given by I 
    """
    assert_is_zero(pos_vec[2])
    assert_is_positive(source_vec[2])
    
    W_ts = (sigma_t - sigma_s)/(sigma_t + sigma_s)
    displacement_vec = pos_vec - source_vec 
    evaluation_coordinates = [displacement_vec[0], displacement_vec[1], - source_vec[2]]
    standard_potential =  2*point_like_potential( evaluation_coordinates, I, sigma_t)
    series_potential = 2* sum([summation_terms(displacement_vec, source_vec, W_ts, n, h, I, sigma_t) for n in range(1, N_max + 1)])
    total_potential = standard_potential + series_potential
    
    return total_potential

def electrode_measurements(neuron_list: list, MEA_set_up: dict, currents):
    
    potential = sum([ MEA_point_source(MEA_set_up['electrode_position'], neuron['location'], currents[count][:, 1], MEA_set_up['sigma_tissue'], 
                MEA_set_up['sigma_saline'], MEA_set_up['brain_slice_height']) for count, neuron in enumerate(neuron_list)])
  
    potential_signal = np.array(list(zip(currents[0][:, 0], potential)))
    
    return potential_signal 

def add_pink_noise(signal, framerate:float = 0, amp: float = None) -> np.array:
    
    pn = sp.PinkNoise()
    print(len(signal)/ framerate)
    pn = pn.make_wave(duration = len(signal)/ framerate, framerate = framerate)
    print(len(pn.ys),type(pn.ys))
   
    if amp is not None:
        amplified_noise = pn.ys *amp
        signal[: len(amplified_noise), 1] += amplified_noise
        return signal
    
    signal[:, 1] += pn.ys
    return signal


def plot_current(sol_t: np.array, sol_I: np.array, title: str ="Intracellular current signal derived from transmembrane voltage recording") -> None:
        plt.plot(sol_t, sol_I, label = 'current')
        plt.xlabel("time (arbitrary)", fontsize=18)
        plt.ylabel("Current $(mA)$", fontsize=18)
        plt.title(title, fontsize=22)
        plt.show()
        return None

def assert_same_timeframe(neurons_list):
    stepsizes = [neuron['time_range'][2] for neuron in neurons_list]
    assert all(el==stepsizes[0] for el in stepsizes), "Not all neurons have been set to the same timestep. Please ensure the last argument in the 'time_range' keys match "

def integrate_neurons(neurons_list: List[dict]) -> Tuple[np.array]:
    assert_same_timeframe(neurons_list)
    
    ts = []
    voltages = []
    currents = []
  
    for neuron_dict in neurons_list:
        sol = sb.ivp_solver(sb.morris_lecar, time_range = neuron_dict['time_range'], initial_cond = neuron_dict['initial_cond'], params = neuron_dict['param_set'])

        ts.append(sol.t)
        voltages.append(sol.y[0])
        current = sb.convert_ml_voltage_to_current(*sol.y, neuron_dict['param_set'])
        currents.append(current)
   
    return ts, voltages, currents


def plot_2D_signal(signal: np.array) -> None:
    
    plt.plot(signal[0][:, 0], signal[0][:, 1])
    plt.show()
    return None

def plot_signal_and_events(signal:List[np.array], events:List[np.array]):
    for i, (sig, event) in enumerate(zip(signal, events)):
        plt.plot(sig[:, 0], sig[:, 1], label = "neuron_{}_sig".format(i))
        plt.scatter(event[:, 0], event[:, 1], label = "neuron_{}_events".format(i), c = 'r')
    plt.legend(loc = 'best', fontsize = 'x-small')
    plt.show()

def shift_signal(signal: np.array, shift_amount: float) -> np.array:
    print(signal)
    new_signal = signal.copy()
    new_signal[:, 0] += shift_amount
    return new_signal

def time_intersection(signals: List[np.array]) -> Tuple[float, float]:
    min_times = []
    max_times = []

    for i in range(len(signals) - 1):
        current_signal = set(signals[i][:, 0])
        next_signal = set(signals[i+1][:, 0])
        intersection = current_signal & next_signal

        min_time = min(intersection)
        max_time = max(intersection)

        min_times.append(min_time)
        max_times.append(max_time)

    min_time = max(min_times)
    max_time = min(max_times)

    return min_time, max_time

def splice_signal_based_on_intersection(signal: np.array, lower: float, upper: float):
    spliced_x_column = signal[(upper >= signal[:, 0]) & (signal[:, 0] >= lower)]
    return spliced_x_column

def  assert_list_is_not_singular(list:list):
    if len(list) > 1:
        pass
    else:
        raise Exception("The list you passed has one or no elements. Please verify its input: {}".format(list))

def combine_time_and_signal_into_2d_array(times: List[np.array], signals: List[np.array]):
    return [np.array(list(zip(t_event, y_event))) for t_event, y_event in zip(times, signals)]

def shift_and_splice_signals(signals: List[np.array], shifts: list):
    assert_list_is_not_singular(signals)
    assert_list_is_not_singular(shifts)
    shifted_signals = [shift_signal(signal, shift) for signal, shift in zip(signals, shifts)]
    lower, upper = time_intersection(shifted_signals)
    spliced_signals =  [splice_signal_based_on_intersection(sig, lower, upper) for sig in shifted_signals]
    return spliced_signals


def load_matlab_file(filedir: str):
    mat = loadmat(filedir)
    return mat

def make_dataframe(matfile):
    df = pd.DataFrame(matfile['cluster_class'])
    return df
    
def get_cluster_times(df):
    cluster_class = df.iloc[:, 0]
    cluster_times= df.iloc[:, 1]
    return cluster_class, cluster_times

def process_waveclus_results(waveclus_filedir, events_filedir, par_pre: int = 20):
    colors = ['grey', 'b', 'r', 'g', 'tab:brown', 'darkviolet']
    neuron_colours = ['m', 'lawngreen', 'm'] #'darkorange'
    mat = load_matlab_file(waveclus_filedir)
    df = make_dataframe(mat)
    cluster_class, cluster_times = get_cluster_times(df)
    peak_cluster_times = cluster_times + par_pre * 0.1
    print(len(peak_cluster_times))
    unique_clusters = np.unique(cluster_class)
    spikes_no_detected = len(cluster_times)

    spikes_list = np.load(events_filedir, allow_pickle=True)
  
    true_pos = 0  
    for cluster in unique_clusters:
        c_time= [cluster_times[i] for i in range(len(cluster_times)) if cluster_class[i] == cluster] 
        plt.scatter(c_time, np.ones(len(c_time)), c= colors[int(cluster)], s =90, label= "cluster {}".format(int(cluster)))
    if spikes_list[0].ndim == 1:
        print(" number of actual spikes {}. Number of detected spikes {}".format(len(spikes_list), spikes_no_detected))
        plt.scatter(spikes_list[:,0], np.ones(len(spikes_list[:,1])) + 0.01, color=[0.4940, 0.1840, 0.5560], label= "TIDA neuron 1", s =90)
        for sp in cluster_times:
            abs_diff = [abs(i) for i in (sp - spikes_list[:,0])]
            for val in abs_diff:
                if  val < 1:
                    true_pos += 1
                    pass
    else:
        print(" number of actual spikes {}. Number of detected spikes {}".format(sum([len(i) for i in spikes_list]), spikes_no_detected))
        for i, neuron in enumerate(spikes_list):
            plt.scatter(neuron[:,0], np.ones(len(neuron[:,1])) + 0.005*(i+1), color= neuron_colours[i], label= "TIDA neuron {}".format(i + 1), s =90)
            for sp in cluster_times:
                abs_diff = [abs(i) for i in (sp - neuron[:,0])]
                for val in abs_diff:
                    if  val < 1:
                        true_pos += 1
                        pass
    plt.ylim([0.9,1.2])
    plt.xlabel("Time (ms)", fontsize = 20)
    plt.xticks( fontsize = 20)
    plt.legend(fontsize = 20)
    plt.show()

      
   
    print("number of true positives {}".format(true_pos))

def scale_to_seconds(measurement):
    cp = copy.deepcopy(measurement[:, 0])
    cp= cp/1000
    return cp


def plot_electrode_measurement( electode_measurement: np.array, mea_parameters: dict, scale_to_seconds = False, spikes = None, rescale=None , y_val = None, title: str = "Potential measurement recorded at the electrode at position {}", plot_label : str = None, y_label = "potential ($m$V)", x_label= "time (arbitrary) "):
    
    
    colours= ['m', 'lawngreen', 'maroon']
    if rescale is not None:
        plt.plot(np.interp(electode_measurement[:, 0], (electode_measurement[:, 0].min(), electode_measurement[:, 0].max()), (0, 120)),   electode_measurement[:, 1], label = plot_label if plot_label is not None else None)
        if spikes is not None:
            for num, events in enumerate(spikes):
                plt.scatter(np.interp(events[:, 0], (electode_measurement[:, 0].min(), electode_measurement[:, 0].max()), (0, 120)), events[:, 1] if y_val is None else np.ones(len(events[:, 0])) * y_val, label = "TIDA neuron {}".format(num + 1), c = colours[num], s =50)
    else:
        plt.plot( electode_measurement[:, 0],   electode_measurement[:, 1], label = plot_label if plot_label is not None else None)
        if spikes is not None:
            for num, events in enumerate(spikes):
                plt.scatter(events[:, 0], events[:, 1] if y_val is None else np.ones(len(events[:, 0])) * y_val, label = "TIDA neuron {} spikes".format(num + 1), s =50,  c = colours[num])
    plt.title(title.format(mea_parameters['electrode_position']), fontsize = 20)
    plt.xlabel(x_label, fontsize = 20)
    plt.ylabel(y_label, fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.show()

def plot_decay_with_distance_example():
    neurons_list = {'time_range': (0, 10000, 0.01) ,'location': np.array([0,0,100])} 
    mea_parameters = {'sigma_tissue': 0.3, 'sigma_saline': 0.3, 'brain_slice_height': 200}
    mea_parameters_hetro = {'sigma_tissue': 0.3, 'sigma_saline': 2, 'brain_slice_height': 200}
    potentials = []
    potentials_hetro = []
    # Source currents
    t = np.linspace(0, 1, 1)  # ms
    imem = np.array([[1.]])  # nA

    h = 200 # slice thickness [um]

    # Electrode positions
    elec_x = np.linspace(0, 1000, 100)  # um
    elec_y = np.zeros(len(elec_x))  # um
    elec_z = 0.  # um

    um_conversion_factor = 1000
    for x, y in zip(elec_x,elec_y):
       
        elec_vec = [x, y, elec_z]
        potential = MEA_point_source(pos_vec= elec_vec, source_vec= neurons_list['location'], I=imem, sigma_t= mea_parameters['sigma_tissue'], sigma_s= mea_parameters['sigma_saline'], h=mea_parameters['brain_slice_height'], N_max = 20)
        potential_hetro = MEA_point_source(pos_vec= elec_vec, source_vec= neurons_list['location'], I=imem, sigma_t= mea_parameters_hetro['sigma_tissue'], sigma_s= mea_parameters_hetro['sigma_saline'], h=mea_parameters_hetro['brain_slice_height'], N_max = 20)
        potentials.append(potential[0] * um_conversion_factor)# nA to UA
        potentials_hetro.append(potential_hetro[0] * um_conversion_factor) # nA to UA

    
    ness_elec_x, ness_phi_homo, ness_phi_hetro = MoI.plot_decay_with_distance_example()
    
    plt.plot(elec_x, potentials, label = "Our homogenous potentials", lw= 5)
    plt.plot(elec_x, potentials_hetro, label = "Our hetro potentials ($\sigma_{s}$ = 2, $\sigma_{t}$ = 0.3)", lw= 5)
    plt.plot(ness_elec_x, ness_phi_homo[:, 0],  label = "Ness' homgenous potentials", alpha=0.9, lw= 4, linestyle = '--' )
    plt.plot(ness_elec_x, ness_phi_hetro[:, 0],  label = "Ness' hetro potentials ($\sigma_{s}$ = 2, $\sigma_{t}$ = 0.3)", alpha=0.9, lw= 4, linestyle = '--' )
    plt.ylabel("Potential $\Phi(t)$  at electode $(\mu V)$")
    plt.xlabel("Distance $(\mu m)$ ")
    plt.legend()
    plt.show()


def save_electrode_measurement_to_matfile( measurement: np.array, filename: str = 'latest_electrode_measurement'):
    mdict = { 'data' : measurement}
    dir = r"D:\Uni work\Engineering Mathematics Work\Technical Project\Simulation_results"
    file_path = os.path.join(dir, filename) + ".mat"
    scipy.io.savemat(file_path, mdict)

def remove_integration_artifacts(sol: np.ndarray, percent: float= 0.01) -> np.array:
    rows = np.shape(sol.y)[1]
    new_inital_point = round(percent*rows)
    shortened_sig = sol.y[:, new_inital_point:]
    shortened_t = sol.t[new_inital_point:]
    
    return shortened_t, shortened_sig

def interpolate_times_in_ms(ts, spike_times, interpolation_range = (0,120_000)):
    interpolated_times = np.array([np.interp(spike_t , (ts.min(), ts.max()), interpolation_range) for spike_t in spike_times])
    return interpolated_times

def find_spikes(signal:np.array, height_for_peak_detection: list = [0,30]):
    peaks = scipy.signal.find_peaks(signal[:,1], height_for_peak_detection) 
    peak_indexes = peaks[0]
    spike_heights = peaks[1]['peak_heights']
    spike_times = [signal[index, 0] for index in peak_indexes]
    spikes = np.array(list(zip(spike_times, spike_heights)))
    return spikes


def single_bursting_example(save_events_path: str = None, return_voltage: np.ndarray =None):
    neurons_list = [{'param_set': sb.TIDA_defaults(), 'time_range': (0, 8825, 0.1), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'track_event': sb.voltage_passes_threshold , 'location': np.array([0,0,30])}]
    mea_parameters = {'sigma_tissue': 0.366, 'sigma_saline': 1.408, 'brain_slice_height': 200, 'electrode_position': np.array([0, 0, 0])}
    ts, voltages, currents= integrate_neurons(neurons_list)
    
    voltages = combine_time_and_signal_into_2d_array(ts, voltages)
    currents = combine_time_and_signal_into_2d_array(ts, currents)
    # print(currents)
    plt.plot(voltages[0][:,0], voltages[0][:,1])
    plt.title("The voltage signal of the 3D Morris Lecar system for parameter values that display TIDA-like behaviour", fontsize=20)
    plt.xlabel("Time (ms)", fontsize =20)
    plt.ylabel("Voltage ($mV$)", fontsize =20)
    plt.xticks(fontsize =20)
    plt.yticks(fontsize =20)
    plt.show()
    elec_m = electrode_measurements(neurons_list, mea_parameters, currents)
    spikes= find_spikes(elec_m)
    noise_adjusted_potential =  add_pink_noise(elec_m, framerate = 10000, amp = 0.06)
    print(len(noise_adjusted_potential[:, 1]))
    plot_electrode_measurement(  noise_adjusted_potential, mea_parameters, spikes= [spikes], x_label= "Time ($ms$)",y_label= "Voltage ($mV$)")
    save_electrode_measurement_to_matfile( noise_adjusted_potential[:, 1], filename = 'single_spiking_high_noise')
    if save_events_path is not None:
        np.save(save_events_path + '\\events',  spikes)
  
def realign_spikes(spikes, electrode_rec):
    realigned_spikes = copy.deepcopy(spikes)
    for i, neuron in enumerate(spikes):
        realigned_spikes[i][:,0] = np.subtract(neuron[:,0], electrode_rec[0,0])
    return realigned_spikes


def near_synchronous_dual_bursting_example(save_events_path = None):
    neurons_list = [{'param_set': sb.TIDA_defaults(), 'time_range': (0, 8825, 0.1), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01),  'track_event': sb.voltage_passes_threshold ,'location': np.array([0,0,30])}
                     , {'param_set': sb.TIDA_defaults(), 'time_range': (0, 8825, 0.1), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01),  'track_event': sb.voltage_passes_threshold , 'location': np.array([30,30,3])}]
    mea_parameters = {'sigma_tissue': 0.366, 'sigma_saline': 1.408, 'brain_slice_height': 200, 'electrode_position': np.array([0, 0, 0])}
    shifts = [0, 500]
    ts, voltages, currents = integrate_neurons(neurons_list)
    voltages = combine_time_and_signal_into_2d_array(ts, voltages)
    currents = combine_time_and_signal_into_2d_array(ts, currents)
    spliced_currents = shift_and_splice_signals(currents, shifts)
    spikes= [find_spikes(curr, height_for_peak_detection= [-1, 6]) for curr in spliced_currents]
    elec_m = electrode_measurements(neurons_list, mea_parameters, spliced_currents)
    plt.plot(elec_m[:, 0], elec_m[:,1])
    plt.show()
    realigned_spikes = realign_spikes(spikes, elec_m)
   
    noise_adjusted_potential =  add_pink_noise(elec_m, framerate = 10_000, amp = 0.01)
    plot_electrode_measurement( noise_adjusted_potential, mea_parameters,  spikes = spikes, y_val = 0, x_label= "Time ($ms$)",y_label= "Voltage ($mV$)", plot_label = 'Potenial at the electrode')
    print("recording len in ms:",len(noise_adjusted_potential)/10_000)
    save_electrode_measurement_to_matfile(noise_adjusted_potential[:, 1], filename = 'near_sync_different_parameters' )
    if save_events_path is not None:
        np.save(save_events_path + '\\events',  realigned_spikes)


def synchronous_bursting_example(save_events_path):
    neurons_list = [{'param_set': sb.TIDA_defaults(), 'time_range': (0, 8825, 0.1), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01),  'track_event': sb.voltage_passes_threshold ,'location': np.array([0,0,30])}
                     , {'param_set': sb.TIDA_defaults(), 'time_range': (0, 8825, 0.1), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01),  'track_event': sb.voltage_passes_threshold , 'location': np.array([30,30,3])}]
    mea_parameters = {'sigma_tissue': 0.366, 'sigma_saline': 1.408, 'brain_slice_height': 200, 'electrode_position': np.array([0, 0, 0])}
    shifts = [0, 80]
    ts, voltages, currents = integrate_neurons(neurons_list)
    voltages = combine_time_and_signal_into_2d_array(ts, voltages)
    currents = combine_time_and_signal_into_2d_array(ts, currents)
    spliced_currents = shift_and_splice_signals(currents, shifts)
    spikes= [find_spikes(curr, height_for_peak_detection= [-1, 6]) for curr in spliced_currents]
    elec_m = electrode_measurements(neurons_list, mea_parameters, spliced_currents)
    realigned_spikes = realign_spikes(spikes, elec_m)

    noise_adjusted_potential =  add_pink_noise(elec_m, framerate = 10_000, amp = 0.01)
    plot_electrode_measurement( noise_adjusted_potential, mea_parameters,  spikes = spikes, y_val = 0, x_label= "Time ($ms$)",y_label= "Voltage ($mV$)", plot_label = 'Potenial at the electrode')
    print("recording len in ms:",len(noise_adjusted_potential)/10_000)
    save_electrode_measurement_to_matfile(noise_adjusted_potential[:, 1], filename = 'synchronous_bursting' )
    if save_events_path is not None:
        np.save(save_events_path + '\\events',  realigned_spikes)




def bursting_with_additive_noise_example():
    neurons_list = [{'param_set': sb.TIDA_defaults(), 'time_range': (0, 10000, 0.01), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'stretch': 4.2, 'track_event': sb.voltage_passes_threshold ,'location': np.array([20,0,100])}
                    , {'param_set': sb.TIDA_defaults(), 'time_range': (0, 10000, 0.01), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'stretch': 4.4, 'track_event': sb.voltage_passes_threshold , 'location': np.array([0,0,100])}]
    mea_parameters = {'sigma_tissue': 0.366, 'sigma_saline': 1.408, 'brain_slice_height': 200, 'electrode_position': np.array([20, 0, 0])}
    ts, voltages, currents, time_events, y_events = integrate_neurons(neurons_list)

    shifts = [200, 220]
    spliced_voltages, spikes = shift_and_splice_signals(ts, voltages , time_events, y_events, shifts)
    
    plot_signal_and_events( spliced_voltages, spikes)
    noise_adjusted_spliced_voltages =  add_pink_noise(spliced_voltages)
    plot_signal_and_events( noise_adjusted_spliced_voltages, spikes)
    elec_m = electrode_measurements(neurons_list, mea_parameters,  noise_adjusted_spliced_voltages)
    plot_electrode_measurement( spliced_voltages[0][:, 0], elec_m, mea_parameters, spikes= None, plot_label = 'Current converted bursting signals')
    

def test():
    neurons_list = [{'param_set': sb.TIDA_defaults(C_m=1, eplsilon= 0.01), 'time_range': (0, 4000, 0.001), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01),  'track_event': sb.voltage_passes_threshold ,'location': np.array([20,0,100])}
                    ,{'param_set': sb.TIDA_defaults(phi = 0.24, E_Ca= 123,  f=7, epsilon= 0.0002, C_m = 20, mu=0.01,  g_KCa= 0.6), 'time_range': (0, 4000, 0.001), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'track_event': sb.voltage_passes_threshold ,'location': np.array([20,0,100])}
                    ,{'param_set': sb.TIDA_defaults(phi = 0.24, E_Ca= 123,  f=8, epsilon= 0.0002, C_m = 20, mu=0.01, g_KCa= 0.6), 'time_range': (0, 4000, 0.001), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'track_event': sb.voltage_passes_threshold ,'location': np.array([20,0,100])}
                    ,{'param_set': sb.TIDA_defaults(phi = 0.24, E_Ca= 123,  f=9, epsilon= 0.0002, C_m = 20, mu=0.01,  g_KCa= 0.60), 'time_range': (0, 4000, 0.001), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'track_event': sb.voltage_passes_threshold ,'location': np.array([20,0,100])}]
    ts, voltages, currents = integrate_neurons(neurons_list)

    plt.plot(ts[0], voltages[0])
    plt.show()
        # Make a figure 
    fig, (ax1, ax2, ax3, ax4)= plt.subplots(4,1)
    fig.size=(30,8)
    fig.suptitle('Simulations of ML model under $g_{KCa}$ parameter variation', fontsize=15)

    # # Plot the Voltage
    
    # ax1.set_xlabel('time (arbitrary)',fontsize=10)
    ax1.set_ylabel('V ($mV$)', fontsize=15)
    ax1.plot(ts[0], voltages[0], label = "$g_{KCa} = 0.75$")
    ax1.grid()
    
    ax2.plot(ts[1], voltages[1], label = "$g_{KCa} = 0.70$", c = 'r')
    # ax2.set_xlabel('time (arbitrary)', fontsize=10)
    ax2.set_ylabel('V ($mV$)', fontsize=15)
    ax2.grid()

    ax3.plot(ts[2], voltages[2], label = "$g_{KCa} = 0.65$", c = 'g')
    # ax3.set_xlabel('Time (arbitrary)', fontsize=15)
    ax3.set_ylabel('V ($mV$)', fontsize=15)
    ax3.grid()

    ax4.plot(ts[3], voltages[3], label = "$g_{KCa} = 0.60$", c = 'tab:orange')
    ax4.set_xlabel('Time (arbitrary)', fontsize=15)
    ax4.set_ylabel('V ($mV$)', fontsize=15)
    ax4.grid()

    custom_xlim = (500, 4000)
    custom_ylim = (-45, 40)

    plt.setp((ax1, ax2, ax3, ax4), xlim=custom_xlim, ylim=custom_ylim)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    fig.legend(lines, labels)
    plt.show()


def main():
    # plot_decay_with_distance_example()
    # single_bursting_example(r"D:\Uni work\Engineering Mathematics Work\Technical Project\Simulation_results")
    # near_synchronous_dual_bursting_example()
    # synchronous_bursting_example(r"D:\Uni work\Engineering Mathematics Work\Technical Project\Simulation_results\Synchronous_bursting")
    process_waveclus_results(r"D:\Uni work\Engineering Mathematics Work\Technical Project\Simulation_results\Synchronous_bursting\times_synchronous_bursting.mat", r"D:\Uni work\Engineering Mathematics Work\Technical Project\Simulation_results\Synchronous_bursting\events.npy")
    
    # test()


if __name__ == "__main__":

    main()
    










