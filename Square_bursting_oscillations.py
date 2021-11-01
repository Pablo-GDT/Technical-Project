import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def mod_Morris_Lecar(t, u):
    (V, n, Ca_conc) = u
    # Here V_1, V_2, V_3, V_4 are parameters chosen to fit voltage clamp data
    V_1 = -1.2
    V_2 = 18
    V_3 = 12
    V_4 = 17.4

    E_Ca = 120
    E_k = -84
    E_L = -60

    # peak conductances of the channel proteins
    g_K = 8
    g_L = 2
    g_Ca = 4
    g_KCa = 0.75

    C_m = 1 

    # I_app = applied current
    I_app = 80

    phi = 4.6
    #  ratio of free to total calcium in the cell
    eta = 0.1
    #  k_Ca  = calcium removal rate
    k_Ca = 1
    #  mu converting current into a concentration flux involving the cell's surface area to the calcium compartment volume
    mu = 0.02

    
    # z is the gating variable with a near Hill-like dependence on the near-membrane calcium concentration
    p = 10
    z = np.power(Ca_conc,p) / (np.power(Ca_conc,p) + 1)
    



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


if __name__ == '__main__':
    t0 = 0
    t_end =200
    y0 = (20,200,0.1)
    sol = solve_ivp(mod_Morris_Lecar,(t0, t_end),y0)
    V, N, Ca = sol.y
    
    plt.plot(sol.t, V)
    plt.show()
