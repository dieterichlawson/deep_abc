import numpy as np
from approx_summary_statistic import *
ising_lattice_strs = open("data/ising-X-10000","r").readlines()
ising_data = np.array([[int(x) for x in line.split()] for line in ising_lattice_strs]).reshape(10000,10,10)
ising_ass = IsingApproximateSummaryStatistic("ising/ising_snapshots/ising_iter_1600.caffemodel",prototxt='ising/ising_predict.prototxt')
print "Loaded Ising summary statistic"
ising_theta_hat = ising_ass.predict(ising_data)
print "Predicted theta hats"
ising_theta_strs = open("data/ising-10000","r").readlines()
ising_theta = np.array([float(x.split()[1]) for x in ising_theta_strs])
print "Loaded true thetas"
np.save("theta_vs_thetahat",np.vstack((ising_theta,ising_theta_hat)))
