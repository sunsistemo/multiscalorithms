import numpy as np
from numpy import inf
from numpy.linalg import solve, norm
from tqdm import trange

# range = trange

def forward_euler(x, f, h, num_steps):
    for step in range(num_steps):
        x[step + 1] = x[step] + h * f(x[step])
    return x

def runge_kutta_4(x, f, h, num_steps):
    for step in range(num_steps):
        k1 = f(x[step])
        k2 = f(x[step] + h / 2. * k1)
        k3 = f(x[step] + h / 2. * k2)
        k4 = f(x[step] + h*k3)
        x[step + 1] = x[step] + h / 6. * (k1 + 2*k2 + 2*k3 + k4)
    return x

def newton(v, system, jacobian, tolerance):
    s = np.full(len(v), inf)    # initial difference

    while norm(s, ord=inf) > tolerance:
        f = system(v)
        j = jacobian(v)

        s = solve(j, -f)        # solving J(v) s = -f(v) for s
        v = v + s
    return v

def backward_euler(x, f, j, h, num_steps, tolerance):
    for step in range(num_steps):
        system = lambda v: v - h * f(v) - x[step]
        # Initial guess for Newton's method is 1 step Forward Euler
        x0 = x[step] + h * f(x[step])
        x[step + 1] = newton(x0, system, j, tolerance)
    return x
