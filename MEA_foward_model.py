from numpy.core.fromnumeric import std
import Square_bursting_oscillations as sb
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import List, Tuple

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

def summation_terms(d_vec:np.array, source_vec: np.array, W_ts: float, n: int, h: float, I: np.array, sigma_t: float):
    """Calculates individual terms in equation 8 given by the paper by Barrera et al DOI: 10.1007/s12021-015-9265-6. 
    Assumes negligible conductivity of the glass electrode plate.

    Args:
        d_vec (np.array): The displacement vector from potential source to electrode [x, y, z]
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
    
    pos_z = point_like_potential([d_vec[0], d_vec[1], 2*n*h - source_vec[2]], I, sigma_t) 
    neg_z = point_like_potential([d_vec[0], d_vec[1], - source_vec[2] - 2*n*h], I, sigma_t)
    n_term = W_ts_n*(pos_z + neg_z)
    return n_term

def MEA_electrode_recording(pos_vec: np.array, source_vec: np.array, I: np.array, sigma_t: float, sigma_g: float, h: float, N_max: int = 20):
    """Calculates the MEA potential recorded at a plate electrode located at z=0 from a point source current positioned at source_vec in equation 8 
    given by the paper by Barrera et al DOI: 10.1007/s12021-015-9265-6.

    Args:
        pos_vec (np.array): The position vector of the glass electrode [x, y, 0]
        source_vec (np.array): The position vector of the potential source [x, y, z]
        I (np.array): The strength of the current eminating from the point source
        sigma_t (float): conductivity of the neural tissue
        sigma_g (float): conductivity of the MEA glass electode plate
        h (float): The height of brain slice region in the z direction 
        N_max (int, optional): The number of term to consider in the series. Defaults to 20.

    Returns:
        [np.array]: Returns the potential for every value of current given by I 
    """
    assert_is_zero(pos_vec[2])
    assert_is_positive(source_vec[2])
    
    W_ts = (sigma_t - sigma_g)/(sigma_t + sigma_g)
    d_vec = pos_vec - source_vec 
    
    p1 = 2* point_like_potential([d_vec[0], d_vec[1], - source_vec[2]], I, sigma_t)
    p2 = 2* sum([summation_terms(d_vec, source_vec, W_ts, n, h, I, sigma_t) for n in range(1, N_max + 1)])
    total_potential = p1 + p2
    
    return total_potential

def add_noise(signal:np.array, mu = 0) -> np.array:
    std = np.std(signal)
    noise = np.random.normal(mu, std, len(signal))

    return signal + noise


def plot_current(sol_t: np.array,sol_I: np.array, title="Intracellular current signal derived from transmembrane voltage recording") -> None:
        plt.plot(sol_t, sol_I, label = 'current')
        plt.xlabel("time (arbitrary)", fontsize=18)
        plt.ylabel("Current $(mA)$", fontsize=18)
        plt.title(title, fontsize=22)
        plt.show()
        return None

def integrate_neurons(neurons_list: List[dict]) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    ts = []
    voltages = []
    currents = []
    time_events = []
    volt_events = []
    for neuron_dict in neurons_list:
        sol = sb.ivp_solver(sb.morris_lecar, neuron_dict['time_range'], neuron_dict['initial_cond'], neuron_dict['param_set'], neuron_dict['track_event']) 
        t_events, y_events = sb.get_ml_threshold_passing_events(sol)
        ts.append(sol.t)
        voltages.append(sol.y[0])
        time_events.append(t_events)
        volt_events.append(y_events)
        sol.y[0] = sb.apply_voltage_filter(sol.y[0])
        sol_current = sb.convert_ml_voltage_to_current(*sol.y, neuron_dict['param_set'])
        currents.append(sol_current)
    return ts, voltages, currents, time_events, volt_events

if __name__ == '__main__':
    neurons_list = [{'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 20000), 'initial_cond': (-20, 1, 0.001), 'stretch': 4.2, 'track_event': sb.voltage_passes_threshold ,'location': (1,1,1)},
                    {'param_set': sb.morris_lecar_defaults(), 'time_range': (0, 20000), 'initial_cond': (-20, 0.5, 0.01), 'stretch': 4.22, 'track_event': sb.voltage_passes_threshold ,'location': (1,2,1)}]
    ts, voltages, currents, time_events, y_events = integrate_neurons(neurons_list)


    plt.plot(ts[0], voltages[0], label = 'transmembrane voltage')

    plt.scatter(time_events[0], y_events[0], label = 'events', color= 'r')
    plt.legend()
    plt.show()

    plt.plot(ts[1], voltages[1], label = 'transmembrane voltage')

    plt.scatter(time_events[1], y_events[1], label = 'events', color= 'r')
    plt.legend()
    plt.show()

    # measurement = MEA_electrode_recording(np.array([0.5,1,0]), np.array([1,3,6]),  I_total, 0.4, 0.001, 20, 30)
    # measurement = add_noise(measurement)
    # plt.plot(sol.t[1100:4000], I_total[1100:4000], label = 'Current converted bursting signal')
    # plt.plot(sol.t[1100:4000], measurement[1100:4000], label = 'MEA recording')
    # plt.xlabel("time (arbitrary) ",fontsize = 18)
    # plt.ylabel("current (mA)", fontsize = 18)
    # plt.title("Foward modelling an intracellular burst signal using MoI", fontsize = 20)
    # plt.legend()
    # plt.show()









