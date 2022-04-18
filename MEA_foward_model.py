import sys
import Square_bursting_oscillations as sb
import math
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import List, Tuple
from numpy.core.fromnumeric import std
import os
import scipy.io, scipy.signal
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
  
    print(np.size(currents[0][:, 0]), np.size(potential) )
    potential_signal = np.array(list(zip(currents[0][:, 0], potential)))
    
    return potential_signal 

def add_pink_noise(signal:  np.array, att=np.log10(0.8)*10, freq_min = 500, freq_max = 1000, rate=44100) -> np.array:
    
    noise = spectrum_noise(lambda x:pink_spectrum(x, freq_min, freq_max, att=att), len(signal[:, 1]), rate=rate) 
    signal[:, 1] += noise
    return signal

def spectrum_noise(spectrum_func, samples=1024, rate=44100):
    """ 
    make noise with a certain spectral density
    Courtesty of Wilbert's website at SocSci https://www.socsci.ru.nl/wilberth/python/noise.html
    """
    freqs = np.fft.rfftfreq(samples, 1.0/rate)            # real-fft frequencies (not the negative ones)
    spectrum = np.zeros_like(freqs, dtype='complex')      # make complex numbers for spectrum
    spectrum[1:] = spectrum_func(freqs[1:])               # get spectrum amplitude for all frequencies except f=0
    phases = np.random.uniform(0, 2*np.pi, len(freqs)-1)  # random phases for all frequencies except f=0
    spectrum[1:] *= np.exp(1j*phases)                     # apply random phases
    noise = np.fft.irfft(spectrum)                        # return the reverse fourier transform
    noise = np.pad(noise, (0, samples - len(noise)), 'constant') # add zero for odd number of input samples
 
    return noise
 
def pink_spectrum(f, f_min=0, f_max=np.inf, att=np.log10(2.0)*10):
    """
    Define a pink (1/f) spectrum
        f     = array of frequencies
        f_min = minimum frequency for band pass
        f_max = maximum frequency for band pass
        att   = attenuation per factor two in frequency in decibel.
                Default is (np.log10(2.0)*10) such that a factor two in frequency increase gives a factor two in power attenuation.
    Courtesty of Wilbert's website at SocSci https://www.socsci.ru.nl/wilberth/python/noise.html
    """
    # numbers in the equation below explained:
    #  0.5: take the square root of the power spectrum so that we get an amplitude (field) spectrum 
    # 10.0: convert attenuation from decibel to bel
    #  2.0: frequency factor for which the attenuation is given (octave)
    s = f**-( 0.5 * (att/10.0) / np.log10(2.0) )  # apply attenuation
    s[np.logical_or(f < f_min, f > f_max)] = 0    # apply band pass
    return s


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
        sol = sb.ivp_solver(sb.morris_lecar, time_range = neuron_dict['time_range'], initial_cond = neuron_dict['initial_cond'], params = neuron_dict['param_set'], track_event = neuron_dict['track_event'])
        sol.t, sol.y = remove_integration_artifacts(sol)
        try:
            sol.t_events, sol.y_events = sb.filter_threshold_passing_events(sol)
            time_events.append(sol.t_events)
            volt_events.append(sol.y_events[:,0])
        except:
            pass
        ts.append(sol.t)
        voltages.append(sol.y[0])

        sol_current = sb.convert_ml_voltage_to_current(*sol.y, neuron_dict['param_set'])
        
        # plt.plot(sol.t, sol_current)
        # plt.xlabel('Sample Number',  fontsize = 25)
        # plt.ylabel('Current ($mA$)', fontsize = 25)
        # plt.title('The outward current without transformation of the voltage signal', fontsize = 25)
        # plt.show()
        currents.append(sol_current)
   
    return ts, voltages, currents, time_events, volt_events


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

def shift_and_splice_signals(ts: List[np.array], signals: List[np.array],time_events, y_events, shifts: list):
    assert_list_is_not_singular(signals)
    assert_list_is_not_singular(shifts)
    assert_list_is_not_singular(time_events)
    time_events_matrix =   combine_time_and_signal_into_2d_array(time_events, y_events)
    time_signal_matrix =  combine_time_and_signal_into_2d_array(ts, signals)
    shifted_events = [shift_signal(event, shift) for event, shift in zip(time_events_matrix, shifts)]
    shifted_signals = [shift_signal(signal, shift) for signal, shift in zip(time_signal_matrix, shifts)]
    lower, upper = time_intersection(shifted_signals)
    spliced_signals =  [splice_signal_based_on_intersection(sig, lower, upper) for sig in shifted_signals]
    spliced_events = [splice_signal_based_on_intersection(event, lower, upper) for event in shifted_events]
    return spliced_signals, spliced_events

def plot_electrode_measurement(  electode_measurement: np.array, mea_parameters: dict, spliced_events = None, title: str = "Potential measurement recorded at the electrode at position {}", plot_label : str = None, y_label = "potential ($\mu$V)"):
    if spliced_events is not None:
        for num, events in enumerate(spliced_events):
            plt.scatter(events[:, 0], events[:, 1], label = "neuron_{}".format(num))
    
    plt.plot( electode_measurement[:, 0],   electode_measurement[:, 1], label = plot_label if plot_label is not None else None)
    plt.title(title.format(mea_parameters['electrode_position']))
    plt.xlabel("time (arbitrary) ",fontsize = 18)
    plt.ylabel(y_label, fontsize = 18)
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


def apply_butterworth_filter(signal: np.array ,filter_order: int, critical_freq: list, output = 'sos'):
    """  Design an Nth-order digital or analog Butterworth filter and return the filter coefficients.

    Args:
        signal (np.array): _description_
        filter_order (int): The order of the filter.
        critical_freq (array_like): The critical frequency or frequencies. For lowpass and highpass filters, Wn is a scalar; for bandpass and bandstop filters, Wn is a length-2 sequence.
                                For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).
        output (str, optional): Type of output: numerator/denominator (‘ba’), pole-zero (‘zpk’), or second-order sections (‘sos’). Default is ‘ba’ for backwards compatibility, but ‘sos’ 
                                should be used for general-purpose filtering. Defaults to 'sos'.

    Returns:
        _type_: Butterworth filtered signal.
    """
    sos = scipy.signal.butter(filter_order, critical_freq, output = output)
    signal[:,1] = scipy.signal.sosfilt(sos, signal[:,1])
    return signal



def save_electrode_measurement_to_matfile( measurement: np.array, filename: str = 'latest_electrode_measurement'):
    mdict = { 'data' : measurement}
    dir = r"D:\Uni work\Engineering Mathematics Work\Technical Project\Simulation_results"
    file_path = os.path.join(dir, filename) + ".mat"
    scipy.io.savemat(file_path, mdict)

def remove_integration_artifacts(sol: np.ndarray, percent: float= 0.1) -> np.array:
    rows = np.shape(sol.y)[1]
    new_inital_point = round(percent*rows)
    shortened_sig = sol.y[:, new_inital_point:]
    shortened_t = sol.t[new_inital_point:]
    
    return shortened_t, shortened_sig

def single_bursting_example():
    neurons_list = [{'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 10300, 0.1), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'stretch': 4.4, 'track_event': sb.voltage_passes_threshold , 'location': np.array([60,0,20])}]
    mea_parameters = {'sigma_tissue': 0.366, 'sigma_saline': 1.408, 'brain_slice_height': 200, 'electrode_position': np.array([0, 0, 0])}
    ts, voltages, currents, time_events, y_events = integrate_neurons(neurons_list)
 
    voltages = combine_time_and_signal_into_2d_array(ts, voltages)
    currents = combine_time_and_signal_into_2d_array(ts, currents)
    events = combine_time_and_signal_into_2d_array(time_events, y_events)
   
    elec_m = electrode_measurements(neurons_list, mea_parameters, currents)
     
    plot_electrode_measurement(elec_m, mea_parameters, spliced_events= None, plot_label = 'Potenial at the electrode')
    noise_adjusted_potential =  add_pink_noise(elec_m)
    plot_electrode_measurement( noise_adjusted_potential, mea_parameters, spliced_events= None, plot_label = 'Potenial at the electrode')
    save_electrode_measurement_to_matfile(noise_adjusted_potential[:, 1], filename = 'single_spiking')


def near_synchronous_dual_bursting_example():
    neurons_list = [{'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 10300, 0.01), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01),  'track_event': sb.voltage_passes_threshold ,'location': np.array([30,0,10])}
                     , {'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 10300, 0.01), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01),  'track_event': sb.voltage_passes_threshold , 'location': np.array([60,0,20])}]
    shifts = [200, 720]
   
    mea_parameters = {'sigma_tissue': 0.366, 'sigma_saline': 1.408, 'brain_slice_height': 200, 'electrode_position': np.array([50, 0, 0])}
    ts, voltages, currents, time_events, y_events = integrate_neurons(neurons_list)

    spliced_currents, spliced_events = shift_and_splice_signals(ts, currents, time_events, y_events, shifts)
    elec_m = electrode_measurements(neurons_list, mea_parameters, spliced_currents)
    noise_adjusted_potential =  add_pink_noise(elec_m)
    print("time", noise_adjusted_potential[-1, 0])
    plot_electrode_measurement( noise_adjusted_potential, mea_parameters, spliced_events = None, plot_label = 'Potenial at the electrode')
    save_electrode_measurement_to_matfile(noise_adjusted_potential[:, 1], filename = 'test2' )

def different_distance_bursting_example():
    neurons_list = [{'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 10000, 0.01), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'stretch': 4.2, 'track_event': sb.voltage_passes_threshold ,'location': np.array([0,0,100])}
                     , {'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 10000, 0.01), 'initial_cond': (-3.06560496e+01,  7.33832272e-03, 8.35251563e-01), 'stretch': 4.4, 'track_event': sb.voltage_passes_threshold , 'location': np.array([1000,0,50])}]
    mea_parameters = {'sigma_tissue': 0.366, 'sigma_saline': 1.408, 'brain_slice_height': 200, 'electrode_position': np.array([200, 0, 0])}
    ts, voltages, currents, time_events, y_events = integrate_neurons(neurons_list)

    shifts = [200, 220]
    spliced_currents, spliced_events = shift_and_splice_signals(ts, currents, time_events, y_events, shifts)
   
    elec_m = electrode_measurements(neurons_list, mea_parameters, spliced_currents)
    print(len(elec_m))
    plot_electrode_measurement(  spliced_currents[0][:,0], elec_m, mea_parameters, spliced_events= None, plot_label = 'Current converted bursting signals')


def bursting_with_additive_noise_example():
    neurons_list = [{'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 10000, 0.1), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'stretch': 4.2, 'track_event': sb.voltage_passes_threshold ,'location': np.array([20,0,100])}
                    , {'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 10000, 0.1), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'stretch': 4.4, 'track_event': sb.voltage_passes_threshold , 'location': np.array([0,0,100])}]
    mea_parameters = {'sigma_tissue': 0.366, 'sigma_saline': 1.408, 'brain_slice_height': 200, 'electrode_position': np.array([20, 0, 0])}
    ts, voltages, currents, time_events, y_events = integrate_neurons(neurons_list)

    shifts = [200, 220]
    spliced_voltages, spliced_events = shift_and_splice_signals(ts, voltages , time_events, y_events, shifts)
    
    plot_signal_and_events( spliced_voltages, spliced_events)
    noise_adjusted_spliced_voltages =  add_pink_noise(spliced_voltages)
    plot_signal_and_events( noise_adjusted_spliced_voltages, spliced_events)
    elec_m = electrode_measurements(neurons_list, mea_parameters,  noise_adjusted_spliced_voltages)
    plot_electrode_measurement( spliced_voltages[0][:, 0], elec_m, mea_parameters, spliced_events= None, plot_label = 'Current converted bursting signals')
    

def test():
    neurons_list = [{'param_set': sb.square_wave_defaults(phi = 0.24, E_Ca= 123,  f=6, epsilon= 0.0002, C_m = 20, mu=0.01,  g_KCa= 0.60), 'time_range': (0, 4000, 0.001), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01),  'track_event': sb.voltage_passes_threshold ,'location': np.array([20,0,100])}
                    ,{'param_set': sb.square_wave_defaults(phi = 0.24, E_Ca= 123,  f=7, epsilon= 0.0002, C_m = 20, mu=0.01,  g_KCa= 0.6), 'time_range': (0, 4000, 0.001), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'track_event': sb.voltage_passes_threshold ,'location': np.array([20,0,100])}
                    ,{'param_set': sb.square_wave_defaults(phi = 0.24, E_Ca= 123,  f=8, epsilon= 0.0002, C_m = 20, mu=0.01, g_KCa= 0.6), 'time_range': (0, 4000, 0.001), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'track_event': sb.voltage_passes_threshold ,'location': np.array([20,0,100])}
                    ,{'param_set': sb.square_wave_defaults(phi = 0.24, E_Ca= 123,  f=9, epsilon= 0.0002, C_m = 20, mu=0.01,  g_KCa= 0.60), 'time_range': (0, 4000, 0.001), 'initial_cond': (-3.06560496e+01,  7.33832272e-03,  8.35251563e-01), 'track_event': sb.voltage_passes_threshold ,'location': np.array([20,0,100])}]
    ts, voltages, currents, time_events, y_events = integrate_neurons(neurons_list)

        # Make a figure 
    fig, (ax1, ax2, ax3, ax4)= plt.subplots(4,1)
    fig.size=(30,8)
    fig.suptitle('Simulations of ML model under $g_{KCa}$ parameter variation', fontsize=15)

    # Plot the Voltage
    
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
    single_bursting_example()
    # near_synchronous_dual_bursting_example()
    # bursting_with_additive_noise_example()
    # different_distance_bursting_example()
    # test()


if __name__ == "__main__":

    main()
    










