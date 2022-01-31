from __future__ import annotations
import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt


class Params:

    def __init__(self, args: dict):
        self.args = args
    
    def __getitem__(self, key):
        return self.args[key]
    
    def __setitem__(self, key, value):
        self.args[key] = value

    def __delitem__(self, key):
        del self.args[key]
        

    @staticmethod
    def default_morris_lecar():
        return Params({
            "V_1" : -1.2, "V_2" : 18, "V_3" : 12, "V_4" : 17.4,

            "E_Ca" : 123, "E_k" : -84, "E_L" : -60,

        # peak conductances of the channel proteins
            "g_K" : 8, "g_L" : 2, "g_Ca" : 4, "g_KCa" : 0.60,

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
    })

class StateSystem:
    def __init__(self, state_space: callable):
        self.state_space = state_space


    def solve_ivp(self, time_range: tuple, initial_conditions: tuple , params: Params) -> TimeSeries:
        sol= sp.solve_ivp(self.state_space, time_range, initial_conditions, args= (params,))
        ts = TimeSeries(sol.y, sol.t)
        return ts

class TimeSeries:
    def __init__(self, x, t):   
        self.x = x
        self.t = t

    def apply_votlage_filter(self, stretch_factor=4.2, translation_factor = 30):
        self.x[0, :] = stretch_factor * self.x[0,:]
        self.x[0, :] =  self.x[0, :] - translation_factor
        return None

    def plot_voltage(self):
        plt.plot(self.t, self.x[0], label = 'voltage')
        title = "The Morris Lecar system integrated in time for parameter values that display TIDA-like behaviour"
        plt.xlabel("time (arbitrary)", fontsize=18)
        plt.ylabel("Voltage $(mV)$", fontsize=18)
        plt.title(title, fontsize=22)
        plt.show()
        return None
        
    def get_I_KCa(self, params: Params) -> np.array:
        Ca_conc =self.x[2, :]
        V = self.x[0, :]
        z = np.power(Ca_conc, params["f"]) / (np.power(Ca_conc, params["f"]) + 1)
        I_KCa = params["g_KCa"]*z*(V-params["E_k"])
        return I_KCa

    def get_I_Ca(self, params) -> np.array:
        V = self.x[0, :]
        M_inf = 0.5*(1 + np.tanh((V - params["V_1"])/params["V_2"]))
        I_Ca =  params["g_Ca"]*M_inf*(V-params["E_Ca"])
        return I_Ca


def morris_lecar(t, u, params):
    (V, n, Ca_conc) = u

    N_inf = 0.5*(1 + np.tanh((V - params["V_3"])/params["V_4"])) # (3)
    tau_n = 1/(np.cosh((V - params["V_3"])/(2*params["V_4"]))) # (4)

    # I_KCa calcium dependent potassium channel (outward current)
    z = np.power(Ca_conc, params["f"]) / (np.power(Ca_conc, params["f"]) + 1)
    I_KCa = params["g_KCa"]*z*(V-params["E_k"])

    M_inf = 0.5*(1 + np.tanh((V - params["V_1"])/params["V_2"]))
    I_Ca =  params["g_Ca"]*M_inf*(V-params["E_Ca"])

    dVdt = (params["I_app"] - params["g_L"]*(V- params["E_L"]) - params["g_K"]*n*(V-params["E_k"]) - I_Ca - I_KCa)/params["C_m"]
    dndt = params["phi"]*(N_inf - n)/ tau_n
    dCa_conc_dt = params["epsilon"]*(-params["mu"]*I_Ca - params["k_Ca"]*Ca_conc)

    return np.array((dVdt, dndt, dCa_conc_dt))




if __name__ == '__main__':
    my_params = Params.default_morris_lecar()
    morris_lecar_system = StateSystem(morris_lecar)
    time_series = morris_lecar_system.solve_ivp((0, 10000), (-20, 1, 0.001), my_params)
    time_series.apply_votlage_filter()
    time_series.plot_voltage()
   

    
  