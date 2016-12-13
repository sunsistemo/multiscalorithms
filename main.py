from random import normalvariate

import numpy as np
import matplotlib.pyplot as plt


V = lambda q: 1/4 * (q**2 - 1)**2    # double-well potential
dV = lambda q: (q**2 - 1) * q

gamma = 1
N = 100

t_end = 1
h = 0.0001
num_steps = int(t_end / h)

q = np.zeros(num_steps + 1)
p = np.zeros(num_steps + 1)
q[0] = 0
p[0] = 0

v = np.zeros([num_steps + 1, N])
u = np.zeros([num_steps + 1, N])
for j in range(N):
    u[0, j] = 100 * normalvariate(0, 1)

# Forward Euler
for i in range(1, num_steps):
    for j in range(N):
        u[i, j] = u[i-1, j] + h * v[i-1, j]
        v[i, j] = v[i-1, j] + h * -j**2 * (u[i-1, j] - q[i-1])

    q[i] = q[i-1] + h * p[i-1]
    p[i] = p[i-1] + h * (-dV(q[i-1]) + gamma**2 * sum([u[i-1, j] - q[i-1] for j in range(N)]))

plt.plot(np.arange(num_steps + h), q)
plt.show()
