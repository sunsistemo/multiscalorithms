import argparse
import gc
from random import normalvariate, seed
from time import process_time, perf_counter, time

import numpy as np
import matplotlib
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
    start = perf_counter()
    cpu_start = process_time()
    x = method(x, f, h, num_steps)
    cpu_end = process_time()
    end = perf_counter()
    return x, cpu_end - cpu_start, end - start

def integrate_implicit(h, num_steps, N, tolerance):
    x = init(num_steps, N, SEED)
    j = generate_jacobian(gamma, h, N)
    start = perf_counter()
    cpu_start = process_time()
    x = backward_euler(x, f, j, h, num_steps, tolerance)
    cpu_end = process_time()
    end = perf_counter()
    return x, cpu_end - cpu_start, end - start

def integrate_scipy(h, times, N):
    x = init(1, N, SEED)[0]
    j = generate_jacobian(gamma, h, N)
    start = perf_counter()
    cpu_start = process_time()
    x = odeint(lambda y, t: f(y), x, times, Dfun=lambda y, t: j(y))
    cpu_end = process_time()
    end = perf_counter()
    return x, cpu_end - cpu_start, end - start

def suppress_diverge(x, vect_length):
    """Replaces diverging values with zero's."""
    diverged = False
    for i in range(len(x)):
        if abs(x[i][0]) > 20:
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
    parser.add_argument("-tol", "--tolerance", help="Newton's method convergence tolerance", type=float, default=1E-2)
    parser.add_argument("--compare-explicit-implicit", help=compare_explicit_implicit.__doc__, action="store_true")
    parser.add_argument("--plot-methods", help=plot_methods.__doc__, action="store_true")
    parser.add_argument("--plot-potential", help=plot_potential.__doc__, action="store_true")
    args = parser.parse_args()

    t_end = args.time
    h = args.time_step
    num_steps = int(t_end / h)
    times = h * np.array(range(num_steps + 1))

    N = args.N
    vect_length = 2 + N + N
    f.vect = np.empty(vect_length)

    if args.compare_explicit_implicit:
        return compare_explicit_implicit()
    if args.plot_methods:
        return plot_methods()
    if args.plot_potential:
        return plot_potential()

    tolerance = args.tolerance
    methods = {"fe": forward_euler, "rk4": runge_kutta_4, "be": backward_euler, "scipy": odeint}
    method = methods.get(args.method)
    if method is None:
        raise ValueError("Available methods are: {}".format(", ".join(methods.keys())))

    if method.__name__ in ["forward_euler", "runge_kutta_4"]:
        x, cpu_t, t = integrate_explicit(method, h, num_steps, N)
    elif method.__name__ == "backward_euler":
        x, cpu_t, t = integrate_implicit(h, num_steps, N, tolerance)
    elif method.__name__ == "odeint":
        x, cpu_t, t = integrate_scipy(h, times, N)
    return x, cpu_t, t

def compare_explicit_implicit():
    """Compare the CPU time needed to integrate the system with the Runge-Kutta 4
    method vs. the Backward Euler method.

    All other script flags are ignored except N.
    """
    t_end = 1000
    # First we'll do explicit
    h = 0.01
    num_steps = int(t_end / h)
    times = h * np.array(range(num_steps + 1))
    gc.disable()                # don't measure garbage-collection
    x1, cpu_t1, t1 = integrate_explicit(runge_kutta_4, h, num_steps, N)
    del(x1)                     # de-allocate this massive array
    gc.collect()

    # And now implicit
    h2 = 0.1
    tolerance = h**2            # because local truncation error is O(h^2) for Backward Euler
    num_steps2 = int(t_end / h2)
    times2 = h2 * np.array(range(num_steps2 + 1))
    x2, cpu_t2, t2 = integrate_implicit(h2, num_steps2, N, tolerance)
    del(x2)
    gc.enable()
    print("Methods: RK4, Backward Euler")
    print("CPU times: ", cpu_t1, cpu_t2)
    print("Wallclock times: ", t1, t2)
    with open("explicit_implicit_N={}_{}.txt".format(N, int(time())), "w") as f:
        f.writelines(["Method\t CPU time\t Wallclock time\n",
                      "RK4:\t {:<25}\t\t {}\n".format(cpu_t1, t1),
                      "BE: \t {:<25}\t\t {}\n".format(cpu_t2, t2)])

def plot_methods():
    h = 0.01
    t_end = 10
    num_steps = int(t_end / h)
    times = h * np.array(range(num_steps + 1))
    x1, cpu_t1, t1 = integrate_explicit(forward_euler, h, num_steps, N)
    x2, cpu_t2, t2 = integrate_explicit(runge_kutta_4, h, num_steps, N)

    tolerance = h**2
    x3, cpu_t3, t3 = integrate_implicit(h, num_steps, N, tolerance)

    # suppress Forward Euler divergence
    x1 = suppress_diverge(x1, 2 + N + N)

    plt.figure(figsize=(9, 6))
    plt.plot(times, x1[:, 0], 'b', label="Forward Euler")
    plt.plot(times, x2[:, 0], 'g', label="Runge-Kutta 4")
    plt.plot(times, x3[:, 0], 'r', label="Backward Euler")
    plt.title("Multiscalorithms")
    plt.xlabel("Time")
    plt.ylabel("Position (q)")
    plt.legend()
    # plt.show()
    print("Methods: Forward Euler, RK4, Backward Euler")
    print("CPU times: ", cpu_t1, cpu_t2, cpu_t3)
    print("Wallclock times: ", t1, t2, t3)
    plt.show()
    # plt.savefig("methods_position_comparison", dpi=400)

def plot_potential():
    matplotlib.rc("text", usetex=True)
    l = 7
    x = np.arange(-l, l, 0.01)
    y = [V(i) for i in x]
    plt.plot(x, y)
    plt.title(r"The potential $V(q) = \frac{1}{4} (q^2 - 1)^2$")
    plt.xlabel("$q$")
    plt.ylabel("$V$")
    plt.savefig("potential", dpi=400)

if __name__ == "__main__":
    main()
