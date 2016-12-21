import argparse
from random import normalvariate, seed
from time import process_time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sympy import Matrix

from calc import forward_euler, runge_kutta_4, backward_euler
from sym import generate_jacobian


V = lambda q: 1/4 * (q**2 - 1)**2    # double-well potential
dV = lambda q: (q**2 - 1) * q
gamma = 1
N = 100
vect_length = 2 + N + N
SEED = 42

def init(num_steps, N, water=None):
    """Set the system's initial conditions."""
    seed(water)
    x = np.zeros([num_steps + 1, 2 + N + N])  # steps, variables
    x[0, 0] = 0.
    x[0, 1] = 0.1
    for i in range(2, N+2):
        x[0, i] = 100 * normalvariate(0, 1)
        x[0, i+N] = 0
    return x

def f(x):
    """Computes the value of the system at x.
    x Is a vector with as components: q, p, u_1 to u_N, v_1 to v_n
    """
    f.vect[0] = x[1]
    f.vect[1] = -dV(x[0]) + gamma**2 * sum([x[2+i] - x[0] for i in range(2, N+2)])
    for i in range(2, N+2):
        f.vect[i] = x[i+N]
    for j, i in enumerate(range(N+2, 2*N + 2)):
        f.vect[i] = -j**2 * (x[i-N] - x[0])
    return f.vect
f.vect = np.empty(vect_length)

def main():
    parser = argparse.ArgumentParser(description="Integrate the model by Ford, Kac and Zwanzig",
                                     epilog="Homepage: https://github.com/sunsistemo/multiscalorithms")
    args = parser.parse_args()
    h = 0.01
    t_end = 10
    num_steps = int(t_end / h)
    times = h * np.array(range(num_steps + 1))
    cpu_times = {}

    # Explicit integrators
    results = {}
    for method in [forward_euler, runge_kutta_4]:
        x = init(num_steps, N, SEED)
        start = process_time()
        x = method(x, f, h, num_steps)
        end = process_time()
        cpu_times[method.__name__] = end - start
        results[method.__name__] = x

    # Backward Euler
    h2 = 0.1
    tolerance = 1E-5
    num_steps2 = int(t_end / h2)
    times2 = h2 * np.array(range(num_steps2 + 1))
    x2 = init(num_steps2, N, SEED)
    j = generate_jacobian(gamma, h2, N)
    start = process_time()
    x = backward_euler(x2, f, j, h2, num_steps2, tolerance)
    end = process_time()
    cpu_times["backward_euler"] = end - start
    results["backward_euler"] = x

    # Scipy Integrate
    x3 = init(1, N, SEED)[0]
    j3 = generate_jacobian(gamma, h2, N)
    print("ODEINT")
    x3 = odeint(lambda y, t: f(y), x3, times2, Dfun=lambda y, t: j3(y))
    results["scipy_integrate"] = x3

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

    plt.plot(times2, results["scipy_integrate"][:, 0], 'm', label="scipy")
    plt.plot(times2, results["backward_euler"][:, 0], 'g', label="Beuler")

    plt.title("Multiscalorithms")
    plt.xlabel("Time")
    plt.ylabel("q")
    plt.legend()
    plt.show()

    return cpu_times

if __name__ == "__main__":
    main()
