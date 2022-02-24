
import Square_bursting_oscillations as sb
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from numpy.core.fromnumeric import std
from scipy.ndimage.interpolation import shift

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
    assert val == 0 

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



def add_noise(signal:np.array, mu: float = 0) -> np.array:
    std = np.std(signal)
    noise = np.random.normal(mu, std, len(signal))

    return signal + noise

def plot_current(sol_t: np.array,sol_I: np.array, title: str ="Intracellular current signal derived from transmembrane voltage recording") -> None:
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
    time_events = []
    volt_events = []
    for neuron_dict in neurons_list:
        sol = sb.ivp_solver(sb.morris_lecar, neuron_dict['time_range'], neuron_dict['initial_cond'], neuron_dict['param_set'], neuron_dict['track_event'])
        sol.t, sol.y = remove_integration_artifacts(sol)
        # plt.plot(sol.t, sol.y[0])
        # plt.show()
        sol.t_events, sol.y_events = sb.filter_threshold_passing_events(sol)
        
        ts.append(sol.t)
        voltages.append(sol.y[0])
        time_events.append(sol.t_events)
        volt_events.append(sol.y_events[:,0])
        sol.y[0] = sb.apply_voltage_filter(sol.y[0])
        sol_current = sb.convert_ml_voltage_to_current(*sol.y, neuron_dict['param_set'])
        currents.append(sol_current)
    return ts, voltages, currents, time_events, volt_events

def remove_integration_artifacts(sol: np.ndarray, percent: float= 0.1) -> np.array:
    rows = np.shape(sol.y)[1]
    new_inital_point = round(percent*rows)
    shortened_sig = sol.y[:, new_inital_point:]
    shortened_t = sol.t[new_inital_point:]
    
    return shortened_t, shortened_sig


def mixing_function(sig: List[np.array], onset: List[float]):
    max_onset = max(onset)
    maxlen = np.max([o + len(s) for o, s in zip(onset, sig)])
    result =  np.zeros(maxlen)
    for i in range(len(onset)):
        result[onset[i]:onset[i] + len(sig[i])] += sig[i] 
    return result

def shift_signals(signals: List[np.array], onset: List[float]):
    new_signal_lengths = [o + len(s) for o, s in zip(onset, signals)]
    maxlen = np.max( new_signal_lengths )
    arr = np.zeros((maxlen,))
    shifted_signals = []
    for i in range(len(onset)):
        arr[onset[i]:onset[i] + len(signals[i])] += signals[i] 
        shifted_signals.append(arr)
    return shifted_signals






def MEA_point_source(pos_vec: np.array, source_vec: np.array, I: np.array, sigma_t: float, sigma_s: float, h: float, N_max: int = 30):
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
    
    standard_potential = 2* point_like_potential([displacement_vec[0], displacement_vec[1], - source_vec[2]], I, sigma_t)
    series_potential = 2* sum([summation_terms(displacement_vec, source_vec, W_ts, n, h, I, sigma_t) for n in range(1, N_max + 1)])
    total_potential = standard_potential + series_potential
    
    return total_potential

def electrode_measurements(neuron_list: list, MEA_set_up: dict, currents):
    
    measurement = sum([ MEA_point_source(MEA_set_up['electrode_position'], neuron['location'], currents[count], MEA_set_up['sigma_tissue'], 
                MEA_set_up['sigma_saline'], MEA_set_up['brain_slice_height']) for count, neuron in enumerate(neuron_list)])
    
    return measurement


def plot_signal(signal: np.array) -> None:
    plt.scatter(signal[:, 0], signal[:, 1])
    return None


def shift_signal(signal: np.array, shift_amount: float) -> np.array:

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

def shift_and_splice_signals(ts: np.array, signals: List[np.array],time_events, y_events, shifts: list):
    time_events_matrix =   [np.array(list(zip(t_event, y_event))) for t_event, y_event in zip(time_events, y_events)]
    time_signal_matrix =  [np.array(list(zip(t_arr,v_arr))) for t_arr, v_arr in zip(ts, signals)]
    shifted_events = [shift_signal(event, shift) for event, shift in zip(time_events_matrix, shifts)]
    shifted_signals = [shift_signal(signal, shift) for signal, shift in zip(time_signal_matrix, shifts)]
    lower, upper = time_intersection(shifted_signals)
    spliced_signals =  [splice_signal_based_on_intersection(sig, lower, upper) for sig in shifted_signals]
    spliced_events = [splice_signal_based_on_intersection(event, lower, upper) for event in shifted_events]
    return spliced_signals, spliced_events


def main():

    neurons_list = [{'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 10000, 0.01), 'initial_cond': (-20, 1, 0.001), 'stretch': 4.2, 'track_event': sb.voltage_passes_threshold ,'location': np.array([1,1,1])},
                    {'param_set': sb.morris_lecar_defaults(epsilon = 0.00018), 'time_range': (0, 10000, 0.01), 'initial_cond': (-20, 1, 0.001), 'stretch': 4.22, 'track_event': sb.voltage_passes_threshold ,'location': np.array([1,2,1])}]
    mea_parameters = {'sigma_tissue': 0.3, 'sigma_saline': 1.5, 'brain_slice_height': 300, 'electrode_position': np.array([5, 5, 0])}
    ts, voltages, currents, time_events, y_events = integrate_neurons(neurons_list)
   
    shifts = [1000, 0]
    spliced_signals, spliced_events = shift_and_splice_signals(ts, currents,time_events, y_events, shifts)
    

   
    for i, (sig, event) in enumerate(zip(spliced_signals, spliced_events)):
        plt.plot(sig[:, 0], sig[:, 1], label = "sig_{}".format( i))
        plt.scatter(event[:, 0], event[:, 1], label = "events_{}".format( i), c = 'r')
    plt.legend()
    plt.show()
    exit()
   
    # elec_m = electrode_measurements(neurons_list, mea_parameters, currents)
   
    # measurement = MEA_electrode_recording(np.array([0.5,1,0]), np.array([1,3,6]),  I_total, 0.4, 0.001, 20, 30)
    # measurement = add_noise(measurement)
    # plt.plot(sol.t[1100:4000], I_total[1100:4000], label = 'Current converted bursting signal')
    # plt.plot(sol.t[1100:4000], measurement[1100:4000], label = 'MEA recording')
    # plt.xlabel("time (arbitrary) ",fontsize = 18)
    # plt.ylabel("current (mA)", fontsize = 18)
    # plt.title("Foward modelling an intracellular burst signal using MoI", fontsize = 20)
    # plt.legend()
    # plt.show()

if __name__ == "__main__":

    main()
    










