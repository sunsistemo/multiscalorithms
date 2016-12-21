from numpy import array
from sympy import latex, Matrix, symbols, Sum, Indexed


def jacobian(gamma, h, N):
    V = lambda q: 1/4 * (q**2 - 1)**2
    dV = lambda q: (q**2 - 1) * q
    vect_length = 2 + N + N

    q, p = symbols("q p")
    u = symbols(["u" + str(i) for i in range(1, N+1)])
    v = symbols(["v" + str(i) for i in range(1, N+1)])
    j = symbols("j")

    y = Matrix([q, p, *u, *v])

    f = [p,
         -dV(q) + gamma**2 * Sum(Indexed("u", j) - q, (j, 1, N))]
    g = [-j**2 * (u[j-1] - q) for j in range(1, N+1)]
    f = Matrix(f + v + g)

    g = y - h*f                 # Backward Euler equation
    jacob = g.jacobian(y)
    return jacob


def generate_jacobian(gamma, h, N):

    jacob = jacobian(gamma, h, N)
    j10 = str(jacob[1, 0])
    jacob[1, 0] = 0
    jacob = jacob.tolist()
    jacob = [[float(x) for x in row] for row in jacob]

    def j(x):
        q = x[0]
        j.m[1, 0] = eval(j10)
        return j.m
    j.m = array(jacob)

    return j
