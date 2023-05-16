import numpy as np
from matplotlib import pyplot as plt

# Constants
X_0 = np.array([[5], 
                [-1]])
A = np.array([[0.9974, 0.0539], 
              [-0.1078, 1.1591]])
B = np.array([[0.0013], 
              [0.0539]])
Q = np.array([[0.25, 0], 
              [0, 0.5]])
R = 0.15
H = np.zeros([2,2])
N = 121

# Algorithm 
def DP():
    P = np.empty(N, dtype=object)
    F = np.empty(N-1, dtype=object)
    P[N-1] = H

    # Backward pass
    for k in reversed(range(0,N-1)): # 119 -> 0 all inclusive
        F[k] = -np.dot(np.linalg.inv(R + np.dot(B.transpose(), np.dot(P[k+1], B))),
                       np.dot(B.transpose(), np.dot(P[k+1], A)))
        P[k] = np.dot((A + np.dot(B, F[k])).transpose(), np.dot(P[k+1], (A + np.dot(B, F[k])))) + np.dot(F[k].transpose(), np.dot(R,F[k])) + Q
    
    # Forward pass
    X = np.empty(N, dtype=object)
    U = np.empty(N-1, dtype=object)
    J = np.empty(N, dtype=object)
    X[0] = X_0
    J[0] = 1/2 * np.dot(X[0].transpose(), np.dot(P[0], X[0]))
    for k in range(0,N-1): # 0 -> 119
        U[k] = np.dot(F[k], X[k])
        X[k+1] = np.dot(A, X[k]) + np.dot(B, U[k])
        J[k+1] = 1/2 * np.dot(X[k+1].transpose(), np.dot(P[k+1], X[k+1]))
        
    return X, U, F, P, J

# Plotting

X, U, F, P, J = DP()
# X[0] is a column vector 2x1
# U[0] is a scalar
# F[0] is row vector 1x2
# P[0] is a 2x2 matrix
# J[0] is a scalar

Ns = np.arange(0,N) # [0, 1 ... 120]
X0s = np.array([subarray[0][0] for subarray in X])
X1s = np.array([subarray[1][0] for subarray in X])
F0s = np.array([subarray[0][0] for subarray in F])
F1s = np.array([subarray[0][1] for subarray in F])

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
fig.subplots_adjust(wspace=0.2)  # Set horizontal space
fig.subplots_adjust(hspace=0.5)  # Set vertical space

axes[0][0].plot(Ns, X0s, lw=4, color='red')
axes[0][0].set_xlabel("Number of steps")
axes[0][0].set_ylabel(r"State ($x_0$)")

axes[0][1].plot(Ns, X1s, lw=4, color='orange')
axes[0][1].set_xlabel("Number of steps")
axes[0][1].set_ylabel(r"State ($x_1$)")

axes[1][0].plot(Ns[0:N-1], U, lw=4, color='blue')
axes[1][0].set_xlabel("Number of steps")
axes[1][0].set_ylabel("Control Input (U)")

axes[1][1].plot(Ns, J, lw=4, color='green')
axes[1][1].set_xlabel("Number of steps")
axes[1][1].set_ylabel("Cost (J)")

axes[2][0].plot(Ns[0:N-1], F0s, lw=4, color='violet')
axes[2][0].set_xlabel("Number of steps")
axes[2][0].set_ylabel(r"Control gain ($f_0$)")

axes[2][1].plot(Ns[0:N-1], F1s, lw=4, color='chartreuse')
axes[2][1].set_xlabel("Number of steps")
axes[2][1].set_ylabel(r"Control gain ($f_1$)")

plt.show()