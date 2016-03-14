import caffe
import numpy as np

class IsingApproximateSummaryStatistic:
  
  def __init__(self, weights, prototxt='ising_predict.prototxt',input_layer='data'):
    self.net = caffe.Net(prototxt, weights, caffe.TEST)
    self.input_dims = self.net.blobs[input_layer].data.shape
    self.lattice_size = self.input_dims[2:]
    self.batch_size = self.input_dims[0]

  def predict(self,isings):
    assert isings.shape[1:] == self.lattice_size
    assert len(isings.shape) == 3
    N,H,W = isings.shape
    isings = isings.reshape(N,1,H,W)
    out =self.net.forward_all(**{'data': isings}) 
    return out['fc6'].reshape(N)

class MAApproximateSummaryStatistic:
  
  def __init__(self, weights, prototxt='ma_predict.prototxt', input_layer='data'):
    self.net = caffe.Net(prototxt, weights, caffe.TEST)
    self.input_dims = self.net.blobs[input_layer].data.shape
    self.seq_length = self.input_dims[3]
    self.batch_size = self.input_dims[0]

  def predict(self,seq):
    assert seq.shape[1] == self.seq_length
    assert len(seq.shape) == 2 
    N,S = seq.shape
    seq = seq.reshape(N,1,1,S)
    out = self.net.forward_all(**{'data': seq})
    return out['fc6'].reshape(N,2)

