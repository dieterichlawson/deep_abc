setwd('~/Google Drive/deep_abc/')
library(IsingSampler)
N = 10      # Number of nodes
n = 10000   # Number of samples
nSample = 1
Resp <- c(0L,1L)

gen.graph <- function(N) {
  a = matrix(0, N^2, N^2)
  for (i in 1:N) {
    for (j in 1:N) {
      if (i > 1) {
        a[(i-1)*N+j,(i-2)*N+j] = 1
      }
      if (i < N) {
        a[(i-1)*N+j,i*N+j] = 1
      }
      if (j > 1) {
        a[(i-1)*N+j, (i-1)*N+j-1] = 1
      }
      if (j < N) {
        a[(i-1)*N+j, (i-1)*N+j+1] = 1
      }
    }
  }
  return(a)
}
Graph = gen.graph(N)
Thresh <- -(rnorm(N)^2)

beta = 0.4406
theta = rexp(n, 1/beta)

# Simulate with metropolis:
start = proc.time()
X = matrix(0, n, N^2)
for(i in 1:n) {
  X[i,] <- IsingSampler(nSample, Graph, Thresh, theta[i], 1000, responses = Resp, method = "MH")
}
proc.time() - start
write.table(X, file= 'ising_r_x', sep = " ", row.names = FALSE, col.names = FALSE)
write.table(theta, file= 'ising_r_theta', sep = " ", row.names = FALSE, col.names = FALSE)
