import caffe
import numpy as np
import argparse

caffe.set_mode_gpu()
caffe.set_device(1)

parser = argparse.ArgumentParser(description='Train a net')
parser.add_argument('solver', help='Solver prototxt.')
parser.add_argument('--weights', help='Weights to fine-tune from.')
parser.add_argument('--snapshot', help='Solver snapshot to resume from.')
args = parser.parse_args()

solver = caffe.SGDSolver(args.solver)

i = 0
while True:
  solver.step(1)
  i+=1
