

import Square_bursting_oscillations as sb
import math
import numpy as np

def point_like_potential(pos_vec: np.array, current: float, homog_elec_cond: float):
    euclid_dist = np.linalg.norm(pos_vec)
    pos_potential = current/ (4 * math.pi * homog_elec_cond * euclid_dist)
    return pos_potential

 
def assert_is_positve(val):
    assert val > 0 

def summation_terms(d_vec:np.array, source_vec: np.array, W_ts:float, n:int, h:float, current:float, sigma_t:float):
        W_ts_n = np.power(W_ts,n)
        assert_is_positve(source_vec[2])
        pos_z = point_like_potential([d_vec[0], d_vec[1], 2*n*h - source_vec[2]], current, sigma_t) 
        neg_z = point_like_potential([d_vec[0], d_vec[1], - source_vec[2] - 2*n*h], current, sigma_t)
        n_term = W_ts_n*(pos_z + neg_z)
        return n_term

def elec_point(pos_vec: np.array, source_vec: np.array, current: float, sigma_t: float, sigma_g: float, h: float, N_max: int = 20):
    assert_is_positve(source_vec[2])
    W_tg = (sigma_t + sigma_g)/(sigma_t + sigma_g)
    W_ts = (sigma_t - sigma_g)/(sigma_t + sigma_g)

    d_vec = pos_vec - source_vec 
    

    p1 = 2* point_like_potential([d_vec[0], d_vec[1], - source_vec[2]], current, sigma_t)
    p2 = 2* sum([summation_terms(d_vec, source_vec, W_ts, n, h, current, sigma_t) for n in range(0, N_max + 1)])
    return p1 + p2



if __name__ == '__main__':
    point = point_like_potential(np.array([1,1,1]), 10, 0.5)
    measurement = elec_point( np.array([1,1,1]), np.array([2,2,-1]) ,  5 , 0.5, 0.0001, 5, 20)
    print(point, measurement)










