import caffe
import numpy as np
import json
import math
import random
class IsingDataLayer(caffe.Layer):

  # Simulate theta from Exp(beta)
  def gen_theta(self, beta, B):
    return np.random.exponential(beta, B)

  # Calcualte the change in energy due to flipping one point on the grid
  def delta_energy(self,array, i, j):
      d = self.lattice_size
      deltaE = 2*array[i][j]*(array[i][(j+1) % d] + array[i][(j-1) % d] + \
                              array[(i+1) % d][j] + array[(i-1) % d][j])
      return deltaE

  def suffstat(self,latt):
      S = 0
      for i in range(0, d):
          for j in range(0, d):
              S += self.delta_energy(latt, i, j)

      return 0.5*S

  # d : dimension of the lattice
  def gen_latt(self,theta, d, gibbs_steps):
    latt = np.random.choice([1,-1],[d,d])
    for t in range(0,gibbs_steps):
        for i in range(0,d):
            for j in range(0,d):
                deltaE = self.delta_energy(latt,i,j)
                p = math.exp(-deltaE*theta)
                if random.random() < p:
                    latt[i][j] = -latt[i][j]
    return latt

  # B : batch size
  def gen_batch(self, B, theta, d, gibbs_steps = 1000):
      X = np.ndarray([B, d, d])
      for i in range(0,B):
          X[i] = self.gen_latt(theta[i], d, gibbs_steps)
      return X

  def setup(self, bottom, top):
    if len(bottom) != 0:
      raise Exception("Ising data layer does not accept inputs")
    if len(top) != 2:
      raise Exception("Ising data layer produces two tops, data and labels.")
    args = {
        'batch_size':64,
        'beta':0.4406,
        'gibbs_steps':1000,
        'lattice_size':10
    }
    args.update(json.loads(self.param_str))
    self.batch_size = args['batch_size']
    self.beta = args['beta']
    self.gibbs_steps= args['gibbs_steps']
    self.lattice_size= args['lattice_size']
        
  def reshape(self, bottom, top):
    top[0].reshape(self.batch_size, 1 , self.lattice_size, self.lattice_size)
    top[1].reshape(self.batch_size)

  def forward(self, bottom, top):
    top[1].data[...] = self.gen_theta(self.beta, self.batch_size)
    top[0].data[...] = self.gen_batch(self.batch_size, top[1].data,\
                            self.lattice_size, self.gibbs_steps)\
                            .reshape(self.batch_size, 1 , self.lattice_size, self.lattice_size) 
   
  def backward(self, top, propagate_down, bottom):
    pass

