import sys
import Square_bursting_oscillations as sb
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from numpy.core.fromnumeric import std

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
    standard_potential =  point_like_potential( evaluation_coordinates, I, sigma_t)
    series_potential = 2* sum([summation_terms(displacement_vec, source_vec, W_ts, n, h, I, sigma_t) for n in range(1, N_max + 1)])
    total_potential = standard_potential + series_potential
    
    return total_potential

def electrode_measurements(neuron_list: list, MEA_set_up: dict, currents):
    
    measurement = sum([ MEA_point_source(MEA_set_up['electrode_position'], neuron['location'], currents[count][:, 1], MEA_set_up['sigma_tissue'], 
                MEA_set_up['sigma_saline'], MEA_set_up['brain_slice_height']) for count, neuron in enumerate(neuron_list)])
    
    return measurement

def add_noise(signal:np.array, mu: float = 0) -> np.array:
    std = np.std(signal)
    noise = np.random.normal(mu, std, len(signal))

    return signal + noise

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
    time_events = []
    volt_events = []
    for neuron_dict in neurons_list:
        sol = sb.ivp_solver(sb.morris_lecar, neuron_dict['time_range'], neuron_dict['initial_cond'], neuron_dict['param_set'], neuron_dict['track_event'])
        sol.t, sol.y = remove_integration_artifacts(sol)
        sol.t_events, sol.y_events = sb.filter_threshold_passing_events(sol)
        ts.append(sol.t)
        voltages.append(sol.y[0])
        time_events.append(sol.t_events)
        volt_events.append(sol.y_events[:,0])
        sol.y[0] = sb.apply_voltage_filter(sol.y[0])
        sol_current = sb.convert_ml_voltage_to_current(*sol.y, neuron_dict['param_set'])
        currents.append(sol_current)
    return ts, voltages, currents, time_events, volt_events

def remove_integration_artifacts(sol: np.ndarray, percent: float= 0.3) -> np.array:
    rows = np.shape(sol.y)[1]
    new_inital_point = round(percent*rows)
    shortened_sig = sol.y[:, new_inital_point:]
    shortened_t = sol.t[new_inital_point:]
    
    return shortened_t, shortened_sig

def plot_signal(signal: np.array) -> None:
    plt.scatter(signal[:, 0], signal[:, 1])
    return None

def plot_signal_and_events(signal:List[np.array], events:List[np.array]):
    for i, (sig, event) in enumerate(zip(signal, events)):
        plt.plot(sig[:, 0], sig[:, 1], label = "neuron_{}_sig".format(i))
        plt.scatter(event[:, 0], event[:, 1], label = "neuron_{}_events".format(i), c = 'r')
    plt.legend(loc = 'best', fontsize = 'x-small')
    plt.show()

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

def  assert_list_is_not_singular(list:list):
    if len(list) > 1:
        pass
    else:
        raise Exception("The list you passed has one or no elements. Please verify its input: {}".format(list))

def shift_and_splice_signals(ts: np.array, signals: List[np.array],time_events, y_events, shifts: list):
    assert_list_is_not_singular(signals)
    assert_list_is_not_singular(shifts)
    assert_list_is_not_singular(time_events)
    time_events_matrix =   [np.array(list(zip(t_event, y_event))) for t_event, y_event in zip(time_events, y_events)]
    time_signal_matrix =  [np.array(list(zip(t_arr,v_arr))) for t_arr, v_arr in zip(ts, signals)]
    shifted_events = [shift_signal(event, shift) for event, shift in zip(time_events_matrix, shifts)]
    shifted_signals = [shift_signal(signal, shift) for signal, shift in zip(time_signal_matrix, shifts)]
    lower, upper = time_intersection(shifted_signals)
    spliced_signals =  [splice_signal_based_on_intersection(sig, lower, upper) for sig in shifted_signals]
    spliced_events = [splice_signal_based_on_intersection(event, lower, upper) for event in shifted_events]
    return spliced_signals, spliced_events

def plot_electrode_measurement( time_arr : np.array, electode_measurement: np.array, mea_parameters: dict, spliced_events = None, title: str = "Current measurement recorded at the electrode at position {}", plot_label : str = None):
    if spliced_events is not None:
        for num, events in enumerate(spliced_events):
            plt.scatter(events[:, 0], events[:, 1], label = "neuron_{}".format(num))
    
    plt.plot( time_arr,  electode_measurement, label = plot_label if plot_label is not None else None)
    plt.title(title.format(mea_parameters['electrode_position']))
    plt.xlabel("time (arbitrary) ",fontsize = 18)
    plt.ylabel("current (mA)", fontsize = 18)
    plt.legend()
    plt.show()

def plot_decay_with_distance_example():
    neurons_list = {'time_range': (0, 10000, 0.01) ,'location': np.array([0,0,100])} 
    mea_parameters = {'sigma_tissue': 0.3, 'sigma_saline': 0.3, 'brain_slice_height': 200}
    mea_parameters_hetro = {'sigma_tissue': 0.366, 'sigma_saline': 1.408, 'brain_slice_height': 200}
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
    plt.plot(elec_x, potentials_hetro, label = "Our hetro potentials ($\sigma_{s}$ = 1.408, $\sigma_{t}$ = 0.366)", lw= 5)
    plt.plot(ness_elec_x, ness_phi_homo[:, 0],  label = "Ness' homgenous potentials", alpha=0.9, lw= 4, linestyle = '--' )
    plt.plot(ness_elec_x, ness_phi_hetro[:, 0],  label = "Ness' hetro potentials ($\sigma_{s}$ = 2, $\sigma_{t}$ = 0.3)", alpha=0.9, lw= 4, linestyle = '--' )
    plt.ylabel("Potential $\Phi(t)$  at electode $(\mu V)$")
    plt.xlabel("Distance $(\mu m)$ ")
    plt.legend()
    plt.show()

   



def main():
    plot_decay_with_distance_example()
  

    # neurons_list = [{'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 10000, 0.01), 'initial_cond': (-20, 1, 0.001), 'stretch': 4.2, 'track_event': sb.voltage_passes_threshold ,'location': np.array([1,1,1])}
    #                 , {'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 10000, 0.01), 'initial_cond': (-20, 1, 0.001), 'stretch': 4.22, 'track_event': sb.voltage_passes_threshold , 'location': np.array([1,2,1])}]
    # mea_parameters = {'sigma_tissue': 0.3, 'sigma_saline': 1.5, 'brain_slice_height': 300, 'electrode_position': np.array([5, 5, 0])}
   
    # ts, voltages, currents, time_events, y_events = integrate_neurons(neurons_list)
    
    # # 
    # shifts = [200, 220]
    
    
    # # plot_current(ts[0], currents[0])


    # spliced_currents, spliced_events = shift_and_splice_signals(ts, currents, time_events, y_events, shifts)
    # plot_signal_and_events(spliced_currents, spliced_events)
    # elec_m = electrode_measurements(neurons_list, mea_parameters, spliced_currents)
    # plot_electrode_measurement( spliced_currents[0][:, 0], elec_m, mea_parameters, spliced_events= spliced_events  ,plot_label = 'Current converted bursting signals')
  

if __name__ == "__main__":

    main()
    










