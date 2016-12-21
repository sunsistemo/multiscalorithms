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

def integrate_explicit(method, h, num_steps, N):
    x = init(num_steps, N, SEED)
    start = process_time()
    x = method(x, f, h, num_steps)
    end = process_time()
    return x, end - start

def integrate_implicit(h, num_steps, N, tolerance):
    x = init(num_steps, N, SEED)
    j = generate_jacobian(gamma, h, N)
    start = process_time()
    x = backward_euler(x, f, j, h, num_steps, tolerance)
    end = process_time()
    return x, end - start

def integrate_scipy(h, times, N):
    x = init(1, N, SEED)[0]
    j = generate_jacobian(gamma, h, N)
    start = process_time()
    x = odeint(lambda y, t: f(y), x, times, Dfun=lambda y, t: j(y))
    end = process_time()
    return x, end - start

def suppress_diverge(x, vect_length):
    """Replaces diverging values with zero's."""
    diverged = False
    for i in range(len(x)):
        if x[i][0] > 20:
            diverged = True
        if diverged:
            x[i] = np.zeros(vect_length)
    return x


def main():
    global N
    parser = argparse.ArgumentParser(description="Numerically Integrate the model by Ford, Kac and Zwanzig",
                                     epilog="Homepage: https://github.com/sunsistemo/multiscalorithms")
    parser.add_argument("-T", "--time", help="integration time period", type=int, default=10)
    parser.add_argument("-dt", "--time-step", help="integration time step", type=float, default=0.01)
    parser.add_argument("-N", type=int, default=100)
    parser.add_argument("-m", "--method", help="integration method", type=str, default="rk4")
    parser.add_argument("-tol", "--tolerance", help="Newton's method convergence tolerance", type=float, default=1E-5)
    args = parser.parse_args()

    t_end = args.time
    h = args.time_step
    num_steps = int(t_end / h)
    times = h * np.array(range(num_steps + 1))

    N = args.N
    vect_length = 2 + N + N
    f.vect = np.empty(vect_length)

    tolerance = args.tolerance
    methods = {"fe": forward_euler, "rk4": runge_kutta_4, "be": backward_euler, "scipy": odeint}
    method = methods.get(args.method)
    if method is None:
        raise ValueError("Available methods are: {}".format(", ".join(methods.keys())))

    if method.__name__ in ["forward_euler", "runge_kutta_4"]:
        x, t = integrate_explicit(method, h, num_steps, N)
    elif method.__name__ == "backward_euler":
        x, t = integrate_implicit(h, num_steps, N, tolerance)
    elif method.__name__ == "odeint":
        x, t = integrate_scipy(h, times, N)
    return x, t

def plot(times, x):
    plt.plot(times, x[:, 0], 'm')
    plt.title("Multiscalorithms")
    plt.xlabel("Time")
    plt.ylabel("q")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
