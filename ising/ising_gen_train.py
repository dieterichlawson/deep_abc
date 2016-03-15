import numpy as np
import math, random
import matplotlib.pyplot as plt
import sys
# Simulate theta from Exp(beta)
def gen_theta(beta, B):
	return np.random.exponential(beta, B)

# Calcualte the change in energy due to flipping one point on the grid.
def delta_energy(array, i, j):
    d = array.shape[0]
    deltaE = 2*array[i][j]*(array[i][(j+1) % d] + array[i][(j-1) % d] + \
                            array[(i+1) % d][j] + array[(i-1) % d][j])
    return deltaE

def suffstat(latt):
    d = latt.shape[0]
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

def ising_abc(n, Sobs, epsilon):
    theta_post = []
    for i in range(0, n):
      theta = gen_theta(beta, 1)
      Xp = gen_latt(theta, lattice_size, gibbs_steps)
      Sp = suffstat(Xp)
      if np.abs(Sp - Sobs) < epsilon:
        theta_post.append(theta[0])

    return theta_post

def ising_abc(n):
    for i in range(0, n):
      theta = gen_theta(beta, 1)
      Xp = gen_latt(theta, lattice_size, gibbs_steps)
      Sp = suffstat(Xp)
      if np.abs(Sp - Sobs) < epsilon:
        theta_post.append(theta[0])

    return theta_post

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: Ising_gen_backup.py <num points> <lattice size> <param filename> <data filename>")
        exit(-1)

    beta = 0.4406
    gibbs_steps = 10000
    n = int(sys.argv[1])
    lattice_size = int(sys.argv[2])
    param_filename = sys.argv[3]
    data_filename = sys.argv[4]

    theta = gen_theta(beta, n)
    X = gen_batch(n, theta, lattice_size, gibbs_steps)
    out = theta
    Xout = np.asarray([x.reshape(1, np.prod(x.shape))[0] for x in X])

    # output format: S, theta  
    np.savetxt(param_filename, out, fmt = "%d %.4f")
    # a vector for one lattice
    np.savetxt(data_filename, Xout, fmt = "%d")



