from random import normalvariate, seed

import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix

from calc import forward_euler, runge_kutta_4, backward_euler
from sym import jacobian


V = lambda q: 1/4 * (q**2 - 1)**2    # double-well potential
dV = lambda q: (q**2 - 1) * q
gamma = 1
N = 10
vect_length = 2 + N + N
SEED = 42

def init(num_steps, N, water=None):
    seed(water)
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


def backward():
    h = 1
    t_end = 10
    num_steps = int(t_end / h)
    times = h * np.array(range(num_steps + 1))
    tolerance = 0.01

    x = init(num_steps, N)
    j = lambda v: jacobian(gamma, h, N)(*v)
    x = backward_euler(x, f, j, h, num_steps, tolerance)

    plt.plot(times, x[:, 0], 'b', label="Backward Euler")
    plt.show()
    return x


def main():
    h = 0.001
    t_end = 1
    num_steps = int(t_end / h)
    times = h * np.array(range(num_steps + 1))

    # Explicit integrators
    results = {}
    for method in [forward_euler, runge_kutta_4]:
        x = init(num_steps, N, SEED)
        x = method(x, f, h, num_steps)
        results[method.__name__] = x

    # Backward Euler
    h2 = 0.01
    tolerance = 1E-10
    num_steps2 = int(t_end / h2)
    times2 = h2 * np.array(range(num_steps2 + 1))
    x2 = init(num_steps2, N, SEED)
    j = lambda v: jacobian(gamma, h2, N)(*v)
    results["backward_euler"] = backward_euler(x2, f, j, h, num_steps2, tolerance, tqdm=True)

    # Filter out divergent forward Euler values
    feuler = results["forward_euler"]
    diverged = False
    for i in range(len(feuler)):
        if diverged:
            feuler[i] = np.zeros(vect_length)
        if feuler[i][0] > 20:
            diverged = True

    plt.plot(times, results["forward_euler"][:, 0], 'b', label="Feuler")
    plt.plot(times, results["runge_kutta_4"][:, 0], 'r', label="RK4")

    plt.plot(times2, results["backward_euler"][:, 0], 'g', label="Beuler")

    plt.title("Multiscalorithms")
    plt.xlabel("Time")
    plt.ylabel("q")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
