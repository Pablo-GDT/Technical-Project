import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Tuple, List

def morris_lecar_defaults() -> dict:
    params = {
            "V_1" : -1.2,
            "V_2" : 18,
            "V_3" : 12,
            "V_4" : 17.4,

            "E_Ca" : 123,
            "E_k" : -84,
            "E_L" : -60,

        # peak conductances of the channel proteins
            "g_K" : 8,
            "g_L" : 2,
            "g_Ca" : 4,
            "g_KCa" : 0.60,

            "C_m" : 20, 

            # I_app : applied current
            "I_app" : 45,

            "phi" : 0.24,
            #  ratio of free to total calcium in the cell
            "epsilon" : 0.0002,
            #  k_Ca  : calcium removal rate
            "k_Ca" : 1,
            #  mu converting current into a concentration flux involving the cell's surface area to the calcium compartment volume
            "mu" : 0.01,
            "f" : 9,
    }
    return params 

def morris_lecar(t: float, u: tuple, params:dict) -> Tuple[float,float,float]:
    (V, n, Ca_conc) = u

    N_inf = 0.5*(1 + np.tanh((V - params["V_3"])/params["V_4"])) # (3)
    tau_n = 1/(np.cosh((V - params["V_3"])/(2*params["V_4"]))) # (4)

    # I_KCa calcium dependent potassium channel (outward current)
    I_KCa = get_I_KCa(Ca_conc, V, params)
    I_Ca =  get_I_Ca(V, params)

    dVdt = (params["I_app"] - params["g_L"]*(V- params["E_L"]) - params["g_K"]*n*(V-params["E_k"]) - I_Ca - I_KCa)/params["C_m"]
    dndt = params["phi"]*(N_inf - n)/ tau_n
    dCa_conc_dt = params["epsilon"]*(-params["mu"]*I_Ca - params["k_Ca"]*Ca_conc)

    return np.array((dVdt, dndt, dCa_conc_dt))

def get_I_KCa(Ca_conc:float, V:float, params:dict) -> float:
    z = np.power(Ca_conc, params["f"]) / (np.power(Ca_conc, params["f"]) + 1)
    I_KCa = params["g_KCa"]*z*(V-params["E_k"])
    return I_KCa

def get_I_Ca(V:float , params:dict) -> float:
    M_inf = 0.5*(1 + np.tanh((V - params["V_1"])/params["V_2"]))
    I_Ca =  params["g_Ca"]*M_inf*(V-params["E_Ca"])
    return I_Ca

def convert_ml_voltage_to_current( V_arr: np.array, n_arr: np.array, Ca_conc_arr: np.array, params: dict) -> np.array:
    I_KCa = get_I_KCa(Ca_conc_arr, V_arr, params)
    I_Ca = get_I_Ca(V_arr, params)
    #      # feed voltage into Morris lecar to get dvdt then convert to .feed values into this equation  g_L*(V- E_L) - g_K*n*(V-E_k) - I_Ca - I_KCa)/C_m
    I_total = (-params["g_L"]*(V_arr- params["E_L"]) - params["g_K"]*n_arr*(V_arr-params["E_k"]) - I_Ca - I_KCa)
    return I_total




def apply_voltage_filter(v_array: np.array, a: float =4.2) -> np.array:
    stretched = np.array([a*v if v > 0 else v for v in v_array])
    translated = stretched - 30
    return translated


def plot_voltage(time_vec: np.array, volt_vec: np.array, title: str = "The Morris Lecar system integrated in time for parameter values that display TIDA-like behaviour"):
    plt.plot(time_vec, volt_vec, label = 'voltage')
    plt.xlabel("time (arbitrarty)", fontsize=18)
    plt.ylabel("Voltage $(mV)$", fontsize=18)
    plt.title(title,fontsize=22)
    plt.show()
    return None

def voltage_passes_threshold(t, system_state, args):
   
    return system_state[0] +20

def get_ml_threshold_passing_events(sol):

    return sol.t_events[0], sol.y_events[0][:,0]

def ivp_solver(system_of_equations: callable,  time_range: tuple, inital_cond: tuple, params: callable = morris_lecar_defaults(), track_event: callable = voltage_passes_threshold) -> Tuple[np.array,np.array]:
    sol = solve_ivp(system_of_equations, time_range, inital_cond, args=(params,), events= track_event)
    return sol

if __name__ == '__main__':
    C = 20
    sol = solve_ivp(morris_lecar,(0, 10000), (-20, 1, 0.001), args= (morris_lecar_defaults(),), events= voltage_passes_threshold)
    print(sol.t_events)
    # print(sol.y_events[0][:,0])
    plt.plot(sol.t, sol.y[0], label= 'voltage')
    plt.plot(sol.t_events[0],sol.y_events[0][:,0], label = 'events')
    plt.legend()
    plt.show()
    # filtered = apply_voltage_filter(sol.y[0])
    # print(filtered)
    
   