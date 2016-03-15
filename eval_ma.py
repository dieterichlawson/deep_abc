import numpy as np
from approx_summary_statistic import *
seq_strs = open("data/MA_X_1e5","r").readlines()
seqs = np.array([[float(x) for x in line.split()] for line in seq_strs]).reshape(100000,100)
ma_ass = MAApproximateSummaryStatistic("ma/ma_snapshots/ma_100_iter_12300.caffemodel",prototxt='ma/ma_predict.prototxt')
print "Loaded MA summary statistic"
ma_S_hat = ma_ass.predict(seqs)
print "Predicted S hats"
ma_S_str = open("data/MA_1e5","r").readlines()
ma_S = np.array([[float(y) for y in x.split()] for x in ma_S_str])
print "Loaded true S"
np.save("ma_100_summ_stat",np.hstack((ma_S,ma_S_hat)))
