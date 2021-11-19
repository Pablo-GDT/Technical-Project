

import Square_bursting_oscillations as sb
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def point_like_potential(pos_vec: np.array, current: float, homog_elec_cond: float):
    euclid_dist = np.linalg.norm(pos_vec)
    pos_potential = current/ (4 * math.pi * homog_elec_cond * euclid_dist)
    return pos_potential

 
def assert_is_zero(val):
    assert val == 0 

def assert_is_positive(val):
    assert val > 0

def summation_terms(d_vec:np.array, source_vec: np.array, W_ts:float, n:int, h:float, current:np.array, sigma_t:float):
        W_ts_n = np.power(W_ts, n)
        
        pos_z = point_like_potential([d_vec[0], d_vec[1], 2*n*h - source_vec[2]], current, sigma_t) 
        neg_z = point_like_potential([d_vec[0], d_vec[1], - source_vec[2] - 2*n*h], current, sigma_t)
        n_term = W_ts_n*(pos_z + neg_z)
        return n_term

def MEA_electrode_recording(pos_vec: np.array, source_vec: np.array, current: np.array, sigma_t: float, sigma_g: float, h: float, N_max: int = 20):
    assert_is_zero(source_vec[2])
    assert_is_positive(pos_vec[2])
    
    W_ts = (sigma_t - sigma_g)/(sigma_t + sigma_g)
    d_vec = pos_vec - source_vec 
    
    p1 = 2* point_like_potential([d_vec[0], d_vec[1], - source_vec[2]], current, sigma_t)
    p2 = 2* sum([summation_terms(d_vec, source_vec, W_ts, n, h, current, sigma_t) for n in range(1, N_max + 1)])
    total_potential = p1 + p2
    
    return total_potential



if __name__ == '__main__':
    t0 = 0
    t_end =1000
    y_morris = (-20, 1, 0.001)
    time_vec, vars_vec = sb.ivp_solver(sb.Morris_lecar, t0, t_end, y_morris)
    C = 1
    I_total = -C * vars_vec[0]
    

    measurement = MEA_electrode_recording(np.array([1,1,1]), np.array([3,8,0]),  I_total, 0.5, 0.03, 15, 30)
    print(I_total, measurement)
    plt.plot(time_vec[1100:2000], I_total[1100:2000], label = 'I_total')
    plt.plot(time_vec[1100:2000], measurement[1100:2000], label = 'MEA_recording')
    plt.legend()
    plt.show()









