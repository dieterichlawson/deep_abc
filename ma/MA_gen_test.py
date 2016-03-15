''' 
Simulates random samples from MA(2)
'''
import numpy as np
import sys 

''' 
Simulates random samples from MA(2)
'''
import numpy as np

'''
Given theta, simulates a sequence of data from a MA(2) model
'''
def MA_gen_seq(theta, n):
    q = len(theta)
    Z = np.random.normal(0,1,n+q)
    theta_rev = list(theta[::-1])
    theta_rev.append(1)
    X = [np.dot(Z[i:i+q+1], theta_rev) for i in range(0,n)]
    return X

'''
Simulates a two dimentional vector uniformaly from a triangle
'''
def MA_gen_theta():
    theta1 = np.random.uniform(-2, 2, 1);
    theta2 = np.random.uniform(max(-theta1-1, theta1-1), 1)
    return [theta1, theta2]

'''
Simulates a batch of thetas
'''
def MA_gen_thetas(B):
    thetas = np.ndarray([B, 2]) 
    thetas[0] = [0.6, 0.2] 
    for i in range(1,B):
        thetas[i] = MA_gen_theta()
    return thetas

'''
Simulates a batch of data of length n
'''
def MA_gen_batch(B, thetas, seq_length):
    batch = np.ndarray([B, seq_length])
    for i in range(0,B):
        batch[i] = MA_gen_seq(thetas[i], seq_length)
    return batch

'''
Computes the first and second autocovariance of a sequence
'''
def MA_autocov(X):
    XXlag1 = [X[i]*X[i+1] for i in range(0, len(X)-1)]
    XXlag2 = [X[i]*X[i+2] for i in range(0, len(X)-2)]
    ac1 = np.mean(XXlag1)
    ac2 = np.mean(XXlag2)
    return (ac1, ac2)

'''
Computes the log likehood of the sequence X
'''
def MA_loglikelihood(X, theta, sigma = 1):
    theta1, theta2 = theta
    sigmasq = sigma**2
    v = (1 + theta1**2 + theta2**2)*(sigmasq)
    c1 = theta1*(1 + theta2)*sigmasq
    c2 = theta2*sigmasq
    seq_length = len(X)
    SIGMA = np.diag([v]*seq_length, 0)
    SIGMA += np.diag([c1]*(seq_length-1), 1)
    SIGMA += np.diag([c1]*(seq_length-1), -1)
    SIGMA += np.diag([c2]*(seq_length-2), 2)
    SIGMA += np.diag([c2]*(seq_length-2), -2)
    
    ll = multivariate_normal.pdf(X, [0]*seq_length, SIGMA)
    return ll


'''
Estimates the posterior moments via vanilla MC 
'''
def MA_post_moments(mc_steps, seq_length, sigma = 1):
    thetas = MA_gen_thetas(int(mc_steps))
    X = [MA_gen_seq(theta, seq_length, sigma) for theta in thetas]
    den = [MA_loglikelihood(X[i], thetas[i], sigma) for i in range(0, mc_steps)]
    den = den - np.mean(den)
    is_w = den
    is_w /= sum(is_w)
    m1, m2 = sum(thetas*is_w[:,np.newaxis])
    s1 = np.sqrt(sum((thetas[:,0] - m1)**2 * is_w))
    s2 = np.sqrt(sum((thetas[:,1] - m2)**2 * is_w))
    rho12 = sum((thetas[:,0] - m1)*(thetas[:,1] - m2)*is_w) / (s1*s2)
    return (m1, m2, s1, s2, rho12)

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: MA_gen_backup.py <num points> <seq length> <MC steps> <param filename> <data filename>")
        exit(-1)
    
    n = int(sys.argv[1])
    seq_length = int(sys.argv[2])
    mc_steps = int(sys.argv[3])
    param_filename = sys.argv[4]
    data_filename = sys.argv[5]
    thetas = MA_gen_thetas(n)
    X = MA_gen_batch(n, thetas, seq_length)
    AC = [MA_autocov(x) for x in X]
    param_out = np.column_stack((AC, thetas))

    # output format: ac1, ac2, t1, t2
    np.savetxt(param_filename, param_out, fmt = "%.4f")
    # each row is one instance 
    np.savetxt(data_filename, X, fmt = "%.4f")
