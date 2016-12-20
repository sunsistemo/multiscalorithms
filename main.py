from random import normalvariate

import numpy as np
import matplotlib.pyplot as plt

from calc import forward_euler, runge_kutta_4


V = lambda q: 1/4 * (q**2 - 1)**2    # double-well potential
dV = lambda q: (q**2 - 1) * q
gamma = 1
N = 100
vect_length = 2 + N + N

def init(num_steps, N):
    x = np.zeros([num_steps + 1, 2 + N + N])  # steps, variables
    x[0, 0] = 0.
    x[0, 1] = 0.1
    for i in range(2, N+2):
        x[0, i] = 100 * normalvariate(0, 1)
        x[0, i+N] = 0
    return x

def f(x):
    f.vect[0] = x[1]
    f.vect[1] = -dV(x[0]) + gamma**2 * sum([x[2+i] - x[0] for i in range(2, N+2)])
    for i in range(2, N+2):
        f.vect[i] = x[i+N]
    for j, i in enumerate(range(N+2, 2*N + 2)):
        f.vect[i] = -j**2 * (x[i-N] - x[0])
    return f.vect
f.vect = np.empty(vect_length)  # create vect once to vectorise

h = 0.001
t_end = 10
num_steps = int(t_end / h)
times = h * np.array(range(num_steps + 1))

results = {}
for method in [forward_euler, runge_kutta_4]:
    x = init(num_steps, N)
    x = method(x, f, h, num_steps)
    results[method.__name__] = x

# Filter out divergent forward Euler values
feuler = results["forward_euler"]
diverged = False
for i in range(len(feuler)):
    if diverged:
        feuler[i] = np.zeros(vect_length)
    if feuler[i][0] > 20:
        diverged = True

plt.plot(times, results["forward_euler"][:, 0], 'b', label="Euler")
plt.plot(times, results["runge_kutta_4"][:, 0], 'r', label="RK4")
plt.title("Multiscalorithms")
plt.xlabel("Time")
plt.ylabel("q")
plt.legend()
plt.show()
