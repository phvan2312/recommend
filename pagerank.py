import numpy as np
from scipy.sparse import csc_matrix
import random

def pagerank(G, s=0.85, max_err=0.01, max_step = 10000, is_init=False, init_r = None):
    n_samples = G.shape[0]

    sum_samples = np.sum(G, axis=1, dtype=np.float32)
    norm_G = G[np.where(sum_samples > 0)[0],:] / sum_samples.reshape(-1,1)

    r_0 = np.zeros(n_samples)
    r_1 = np.ones(n_samples) if is_init == False else init_r
    assert r_1.shape[0] == n_samples

    n_step = 1

    while np.sum(np.abs(r_1 - r_0)) > max_err and n_step < max_step:
        r_0 = r_1.copy()
        r_1 = s*np.dot(norm_G.T,r_0) + (1-s)*np.ones(n_samples)
        n_step += 1

    return r_1

if __name__ == '__main__':
    G = np.array([[0, 0, 1, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0, 0],
                  [1, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 0, 1]])

    print pagerank(G,s=0.86,max_err=.0001)
    pass