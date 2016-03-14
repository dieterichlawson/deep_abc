import numpy as np
import math, random

# Simulate theta from Exp(beta)
def gen_theta(beta, B):
	return np.random.exponential(beta, B)

# Calcualte the change in energy due to flipping one point on the grid.
def delta_energy(array, i, j):
    deltaE = 2*array[i][j]*(array[i][(j+1) % d] + array[i][(j-1) % d] + \
                            array[(i+1) % d][j] + array[(i-1) % d][j])
    return deltaE

def suffstat(latt):
    S = 0
    for i in range(0, d):
        for j in range(0, d):
            S += delta_energy(latt, i, j)

    return 0.5*S

# d : dimension of the lattice
def gen_latt(theta, d, gibbs_steps):
	latt = np.random.choice([1,-1],[d,d])
	for t in range(0,gibbs_steps):
	    for i in range(0,d):
	        for j in range(0,d):
	            deltaE = delta_energy(latt,i,j)
	            p = math.exp(-deltaE*theta)
	            if random.random() < p:
	                latt[i][j] = -latt[i][j]
	return latt

# B : batch size
def gen_batch(B, theta, d, gibbs_steps = 1000):
    X = np.ndarray([B, d, d])
    for i in range(0,B):
        X[i] = gen_latt(theta[i], d, gibbs_steps)
    return X       