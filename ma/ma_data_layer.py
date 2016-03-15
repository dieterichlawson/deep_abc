import caffe
import numpy as np
import json

class MADataLayer(caffe.Layer):
  def gen_ma(self, theta, n):
    q = len(theta)
    Z = np.random.normal(0,1,n+q)
    theta_rev = list(theta[::-1])
    theta_rev.append(1)
    X = [np.dot(Z[i:i+q+1], theta_rev) for i in range(0,n)]
    return X

  def gen_ma_theta(self):
    theta1 = np.random.uniform(-2, 2, 1);
    theta2 = np.random.uniform(max(-theta1-1, theta1-1), 1)
    return [theta1, theta2]

  def gen_ma_thetas(self, B):
    thetas = np.ndarray([B, 2])
    for i in range(0,B):
      thetas[i] = self.gen_ma_theta()
    return thetas

  def gen_batch(self, B, thetas, n):
    batch = np.ndarray([B, n])
    for i in range(0,B):
      batch[i] = self.gen_ma(thetas[i], n)
    return batch

  def setup(self, bottom, top):
    if len(bottom) != 0:
      raise Exception("MA data layer does not accept inputs")
    if len(top) != 2:
      raise Exception("MA data layer produces two tops, data and labels.")
    args = {
        'batch_size':64,
        'seq_length':1000
    }
    args.update(json.loads(self.param_str))
    self.batch_size = args['batch_size']
    self.seq_length = args['seq_length']

  def reshape(self, bottom, top):
    top[0].reshape(self.batch_size, 1, 1, self.seq_length) 
    top[1].reshape(self.batch_size, 2)

  def forward(self, bottom, top):
    top[1].data[...] = self.gen_ma_thetas(self.batch_size)
    top[0].data[...] = self.gen_batch(self.batch_size, top[1].data, self.seq_length)\
                           .reshape(self.batch_size, 1, 1, self.seq_length)

  def backward(self, top, propagate_down, bottom):
    pass
