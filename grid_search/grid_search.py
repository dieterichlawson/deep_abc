from net_writer import *
import os



def prepare_directory(dirname,conv_sizes,fc_sizes,conv_batchnorm,fc_batchnorm,lr):
  os.makedir(dirname)
  proto = os.path.join(dirname,"train.prototxt")
  write_prototxt(proto,conv_sizes,fc_sizes,conv_batchnorm, fc_batchnorm)
  
  
def write_prototxt(outfile,conv_sizes,fc_sizes,conv_batchnorm,fc_batchnorm):
  net = NetWriter()
  net.header()
  for num_filters,kernel_size in conv_sizes:
    net.conv(num_filters,kernel_size)
    if conv_batchnorm:
      net.batchnorm(False)
    net.relu()
  for num_hidden in fc_sizes:
    net.fc(num_hidden)
    if fc_batchnorm:
      net.batchnorm(False)
    net.relu()
  net.loss()
  with open(outfile,'w') as f:
    f.writeln(net.proto)


def write_solver(outfile,lr,gamma,stepsize):
  text= """net: "train.prototxt"
test_iter: 0
test_interval:256
type: "Adam"
base_lr: %f
lr_policy: "step"
gamma: %f
stepsize: %d
display: 4
max_iter: 100000
momentum: 0.9
snapshot: 100
snapshot_prefix: "%s""" % (lr, gamma, stepsize,directory)
  with open(outfile,'w') as f:
    f.write(text)
