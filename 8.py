#!/usr/local/bin/python3.10
import numpy as np
import matplotlib.pyplot as plt #type:ignore
import pandas as pd
from scipy.special import jacobi, gamma
import scipy.integrate as integrate
from scipy.linalg import solve
from numpy.core.arrayprint import array2string
import seaborn as sns
from math import exp

def print_my_var():
    a = r"f(x) = \frac{4+x}{5+2x}u'' + \frac{-x}{5+2x}u' + \frac{1+2x}{5+2x}u = e^x,\ u(-1)= \frac{1}{e}, u(1)=e"
    plt.figure('Диффур второго порядка')
    ax = plt.axes([0,0,0.3,1.0]) #left,bottom,width,height
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.text(0.4,0.4,'$%s$' %a,color="black", fontsize=20)
    plt.show()

def get_info_from_my_var():
    # a = lambda x : (4 + x)/(5 + 2*x)
    # b = lambda x : (-x)/(5 + 2*x)
    # c = lambda x : (1 + 2*x)/(5 + 2*x)
    # d = lambda x : exp(x)
    # left, right = -1, 1
    # leftval, rightval = exp(left), exp(right)
    a = lambda _ : 1
    b = lambda _ : 1
    c = lambda _ : 1
    d = lambda x : 3*exp(x)
    left, right = -1, 1
    leftval, rightval = exp(left), exp(right)

    return a, b, c, d, left, right, leftval, rightval

def jac_times_poly(x, y):
    return jacobi(y, 1, 1)(x) * (x**4 - 2 * x**2 + 1)

def jac_times_dpoly(x, y):
    return jacobi(y, 1, 1)(x) * (4 * x**3 - 4 * x)

def jac_times_ddpoly(x, y):
    return jacobi(y, 1, 1)(x) * (12 * x**2 - 4)

def djac_times_poly(x, y):
    return gamma(y + 4)/(2*gamma(y + 3)) * jacobi(y-1, 1+1, 1+1)(x) * (x**4 - 2 * x**2 + 1)

def ddjac_times_poly(x, y):
    return gamma(y + 5)/(4*gamma(y + 3)) * jacobi(y-2, 1+2, 1+2)(x) * (x**4 - 2 * x**2 + 1)

def djac_times_dpoly(x, y):
    return gamma(y + 4)/(2*gamma(y + 3)) * jacobi(y-1, 1+1, 1+1)(x) * (4 * x**3 - 4 * x)

def get_phi(n):
    P, dP, ddP = [None] * n, [None] * n, [None] * n

    P[0] = lambda x : -0.5 * x**2 - 0.5 * x + 1          # (-1, 1), (0, 1), (1, 0)
    P[1] = lambda x : 3 * x**3 - 0.5 * x**2 - 2.5 * x + 1 # (-1, 0), (0, 1), (0.5, 0), (1, 1)

    for i in range(2, n):
        P[i] = lambda x, deg = i - 2: jac_times_poly(x, deg)

    dP[0] = lambda x : -1*x - 1/2
    dP[1] = lambda x : 9 * x**2 - x - 2.5
    if n > 2:
        dP[2] = lambda x : 4 * x**3 - 4*x

    for i in range(3, n):
        dP[i] = lambda x, deg = i - 2:  jac_times_dpoly(x, deg) + djac_times_poly(x, deg)

    ddP[0] = lambda _ : -1
    ddP[1] = lambda x : 18 * x - 1
    if n > 2:
        ddP[2] = lambda x : 12 * x**2 - 4
        ddP[3] = lambda x : 40 * x**3 - 24 * x

    for i in range(4, n):
        ddP[i] = lambda x, deg = i - 2 : ddjac_times_poly(x, deg) + jac_times_ddpoly(x, deg) + 2 * djac_times_dpoly(x, deg)

    return P, dP, ddP

def get_ans(x, coeffs, F):
    ans = 0
    for (coeff, f) in zip(coeffs, F):
        ans += coeff[0] * f(x)
    return ans

def solve_diffequation(a, b, c, d, left, right, leftval, rightval):
    nodes = 1000
    grid = np.linspace(left, right, nodes)
    data = {'X' : [], 'Функции' : [], 'Y' : []}
    maxN = 5

    phi, dphi, ddphi = get_phi(maxN)

    for i in range(maxN):
        data['X'].extend(grid)
        data['Функции'].extend("phi" + str(i) for _ in range(nodes))
        data['Y'].extend(map(phi[i], grid)) #type: ignore

    dataframe = pd.DataFrame(data)
    sns.lineplot(data=dataframe, x='X', y='Y', hue='Функции')
    plt.show()

    data = {'X' : [], 'N' : [], 'Y' : []}

    for N in range(2, maxN, 1):
        rhs = np.zeros((N + 1, 1))
        matrix = np.identity(N + 1)
        rhs[0], rhs[1] = leftval, rightval
        print("Solving for N = ", N)

        for i in range(2, N + 1):
            for j in range(2, N + 1):
                func = lambda x : phi[i](x) * (a(x)*ddphi[j](x) + b(x)*dphi[j](x) + c(x)*phi[j](x)) #type: ignore
                matrix[i, j] = integrate.quad(func, left, right)[0]
            rhs[i] = integrate.quad(lambda x : phi[i](x) * d(x), left, right)[0] #type: ignore

        print(array2string(matrix, precision=3, max_line_width=300), "\n")
        # print("matrix cond is ", np.linalg.cond(matrix))
        coeffs = solve(matrix, rhs)

        print("coeffs = \n", coeffs)

        data['X'].extend(grid)
        data['N'].extend(str(N) for _ in range(nodes))
        data['Y'].extend(map(lambda x : get_ans(x, coeffs, phi), grid))

    data['X'].extend(grid)
    data['N'].extend("e^x" for _ in range(nodes))
    data['Y'].extend(map(exp, grid))
    # data['Y'].extend(map(sin, grid))
    return data

def main():

    print("Лабораторная работа №8\n"
          "Проекционный метод для краевой задачи. Метод Галёркина\n")
    print("Собственный вариант. u = exp(x)")
    print("Уравнение выглядит как a(x)u'' + b(x)u' + c(x)u = d(x)\n")

    # print_my_var()

    data = solve_diffequation(*get_info_from_my_var())

    dataframe = pd.DataFrame(data)
    print(dataframe)
    sns.lineplot(data=dataframe, x='X', y='Y', hue='N')
    plt.show()
    return

if __name__ == "__main__":
    main()
