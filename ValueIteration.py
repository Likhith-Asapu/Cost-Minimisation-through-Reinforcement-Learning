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
def VI():
    num_iter = 100
    P_0 = np.zeros([2,2])
    F_0 = np.array([[ 0.75, -0.65]])
    for _ in range(num_iter):
        P_1 = np.dot((A + np.dot(B, F_0)).transpose(), np.dot(P_0, (A + np.dot(B, F_0)))) + np.dot(F_0.transpose(), np.dot(R,F_0)) + Q
        F_1 = -np.dot(np.linalg.inv(R + np.dot(B.transpose(), np.dot(P_1, B))),
               np.dot(B.transpose(), np.dot(P_1, A)))
        P_0 = P_1
        F_0 = F_1
    P = P_1
    F = F_1
    
    X = np.empty(N, dtype=object)
    U = np.empty(N-1, dtype=object)
    J = np.empty(N, dtype=object)
    X[0] = X_0
    J[0] = 1/2 * np.dot(X[0].transpose(), np.dot(P, X[0]))
    for k in range(0,N-1): # 0 -> 119
        U[k] = np.dot(F, X[k])
        X[k+1] = np.dot(A, X[k]) + np.dot(B, U[k])
        J[k+1] = 1/2 * np.dot(X[k+1].transpose(), np.dot(P, X[k+1]))
        
    return X, U, F, P, J

# Plotting
X, U, F, P, J = VI()

Ns = np.arange(0,N) # [0, 1 ... 120]
X0s = np.array([subarray[0][0] for subarray in X])
X1s = np.array([subarray[1][0] for subarray in X])

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
fig.subplots_adjust(wspace=0.2)  # Set horizontal space
fig.subplots_adjust(hspace=0.2)  # Set vertical space

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

plt.show()