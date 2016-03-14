from approx_summary_statistic import *
from IPython import embed
import numpy as np
ising_ass = IsingApproximateSummaryStatistic("ising_snapshots/ising_iter_300.caffemodel")
print "Loaded Ising statistic"
ising_out = ising_ass.predict(np.random.rand(500,10,10))
ma_ass = MAApproximateSummaryStatistic("ma_snapshots/ma_iter_100.caffemodel")
print "Loaded MA statistic"
ma_out = ma_ass.predict(np.random.rand(500,1000))
embed()
