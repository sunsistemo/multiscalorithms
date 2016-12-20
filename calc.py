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
