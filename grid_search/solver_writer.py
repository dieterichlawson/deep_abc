class SolverWriter():

  def __init__(directory,gamma,stepsize,lr):
    self.proto = """net: "ising_train.prototxt"
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

