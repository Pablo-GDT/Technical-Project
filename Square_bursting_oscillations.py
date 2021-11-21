import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def morris_lecar2_defaults():
    params = {
            "V_1" : -1.2,
            "V_2" : 18,
            "V_3" : 12,
            "V_4" : 17.4,

            "E_Ca" : 120,
            "E_k" : -84,
            "E_L" : -60,

        # peak conductances of the channel proteins
            "g_K" : 8,
            "g_L" : 2,
            "g_Ca" : 4,
            "g_KCa" : 0.75,

            "C_m" : 1, 

            # I_app : applied current
            "I_app" : 80,

            "phi" : 4.6,
            #  ratio of free to total calcium in the cell
            "eta" : 0.1,
            #  k_Ca  : calcium removal rate
            "k_Ca" : 1,
            #  mu converting current into a concentration flux involving the cell's surface area to the calcium compartment volume
            "mu" : 0.02,
            "f" : 10,
      
    }
    return params 

def mod_Morris_Lecar2(t, u, p):
    (V, n, Ca_conc) = u

    z = np.power(Ca_conc, p["f"]) / (np.power(Ca_conc, p["f"]) + 1)
    

    M_inf = 0.5*(1 + np.tanh((V - p["V_1"])/p["V_2"])) # (2)
    N_inf = 0.5*(1 + np.tanh((V - p["V_3"])/p["V_4"])) # (3)
    tau_n = 1/(np.cosh((V - p["V_3"])/(2*p["V_4"]))) # (4)

    # I_KCa calcium dependent potassium channel (outward current)
    I_KCa = p["g_KCa"]*z*(V-p["E_k"])
    I_Ca =  p["g_Ca"]*M_inf*(V-p["E_Ca"])

    dVdt = (p["I_app"] - p["g_L"]*(V- p["E_L"]) - p["g_K"]*n*(V-p["E_k"]) - I_Ca - I_KCa)/p["C_m"]
    dndt = p["phi"]*(N_inf - n)/ tau_n
    dCa_conc_dt = p["eta"]*(-p["mu"]*I_Ca - p["k_Ca"]*Ca_conc)

    return np.array((dVdt, dndt, dCa_conc_dt))



def Morris_lecar(t, u, C_m: float = 1):

    # N : recovery variable: the probability that the K+ channel is conducting
    (V, n, Ca_conc) = u
    
    # Here V_1, V_2, V_3, V_4 are parameters chosen to fit voltage clamp data
    V_1 = -1.2 
    V_2 = 18
    V_3 = 12      
    V_4 = 17.4
    
    # equilibrium potential of relevant ion channels
    E_Ca = 120
    E_k = -90
    E_L = -60 

    # peak conductances of the channel proteins
    g_K = 10
    # leaky channel conductance
    g_L = 2
    g_Ca = 3.6
    g_KCa = 0.8

    # membrane capacitance
    # C_m = 1 

    # I_app = applied current
    I_app = 45 

    phi = 5
    #  ratio of free to total calcium in the cell
    eta = 0.004
    #  k_Ca  = calcium removal rate
    k_Ca = 1
    #  mu converting current into a concentration flux involving the cell's surface area to the calcium compartment volume
    mu = 0.01

    
    # z is the gating variable with a near Hill-like dependence on the near-membrane calcium concentration
    p = 6
    z = np.power(Ca_conc, p) / (np.power(Ca_conc, p) + 1)
    

    M_inf = 0.5*(1 + np.tanh((V - V_1)/V_2)) # (2)
    N_inf = 0.5*(1 + np.tanh((V - V_3)/V_4)) # (3)
    tau_n = 1/(np.cosh((V - V_3)/(2*V_4))) # (4)
    
    # I_KCa calcium dependent potassium channel (outward current)
    I_KCa = g_KCa*z*(V-E_k)
    I_Ca =  g_Ca*M_inf*(V-E_Ca)

    dVdt = (I_app - g_L*(V- E_L) - g_K*n*(V-E_k) - I_Ca - I_KCa)/C_m
    dndt = phi*(N_inf - n)/ tau_n
    dCa_conc_dt = eta*(-mu*I_Ca - k_Ca*Ca_conc)
    
    return np.array((dVdt, dndt, dCa_conc_dt))



def Hindmarsh_rose(t, u):
    a = 1
    b = 3
    c = 1
    d = 4
    r = 0.001
    I=2
    x,y,z = u

    dxdt = y - a*np.power(x,3) + b*np.power(x, 2) - z + I
    dydt = c - d*np.power(x,2) - y 
    dzdt = r*(4 * (x + 1.6)-z)


    return np.array([dxdt,dydt,dzdt])

def ivp_solver(system_of_equations, params: list, t0, t_end, inital_cond):
    sol = solve_ivp(system_of_equations,(t0, t_end), inital_cond, args=params)
    return sol.t, sol.y



def convert_ml_voltage_to_current(v_array, capcitance):
     I_total = -capcitance * v_array
     return I_total
    

if __name__ == '__main__':
    pass
    # t0 = 0
    # t_end =1000
    # y_morris = (-25, 0.9, 0.001)
    # time_vec, vars_vec = ivp_solver(Morris_lecar_current, t0, t_end, y_morris)
    # C = 2
    # I_total = -C * vars_vec[0]
    # plot_sol(time_vec[:200],I_total[:200])
    



    # y_hein = (0, 0.9, 0.001)
    # heind_sol = solve_ivp(Hindmarsh_rose, (t0, t_end), y_hein)
    # X, Y, Z = heind_sol.y
    # plt.plot(heind_sol.t, X)
    # plt.show()
    # plt.plot(heind_sol.t, Y)
    # plt.show()
    # plt.plot(heind_sol.t, Z)
    # plt.show()