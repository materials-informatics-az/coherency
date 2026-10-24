"""
Fig. 5(b,c): parabolic Gibbs energies of the Cu-rich (alpha) and Ag-rich (beta) solid solutions
at 1000 K and the coherent alpha+beta curve for effective misfits of 1 and 5 %.
Panel (a) is the Cu-Ag phase diagram from Thermo-Calc and is not reproduced here.
Energies in J/mol for this figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, solve, lambdify

# parabolas matched to the Thermo-Calc minima of the two solid solutions at 1000 K (J/mol);
# x is the Ag fraction as in Fig. 5(a): alpha is the Cu-rich phase (left), beta the Ag-rich phase (right)
x_a0, x_b0 = 0.2, 0.8
b_a, b_b = -46896.8, -56138.2
a = 94160

# coherency energy, Eq. (1), with the properties of Cu
E, nu, V = 110e9, 0.34, 7.1e-6          # Pa, -, m^3/mol


def W_of(eps):
    return V*E*eps**2/(1 - nu)          # J/mol


x_a, x_b, X, W = symbols('x_a x_b X W')
g_a = a*(x_a - x_a0)**2 + b_a
g_b = a*(x_b - x_b0)**2 + b_b
z_a = (X - x_b)/(x_a - x_b)
sol = solve([diff(g_a, x_a) - diff(g_b, x_b),
             diff(g_b, x_b) - (g_a - g_b + W*(1 - 2*z_a))/(x_a - x_b)], [x_a, x_b], dict=True)[0]
f_xa, f_xb = lambdify((X, W), sol[x_a]), lambdify((X, W), sol[x_b])
f_ga, f_gb = lambdify(x_a, g_a), lambdify(x_b, g_b)


def G_ab(Xs, Wv):
    """Energy of the coherent alpha+beta mixture (nan outside the two-phase range)."""
    xa, xb = f_xa(Xs, Wv), f_xb(Xs, Wv)
    za = (Xs - xb)/(xa - xb)
    G = za*f_ga(xa) + (1 - za)*f_gb(xb) + Wv*za*(1 - za)
    return np.where((za > 0) & (za < 1), G, np.nan)


x = np.linspace(0, 1, 1001)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
for ax, eps, lab in zip(axs, [0.01, 0.05], 'bc'):
    Wv = W_of(eps)
    print('eps = %.2f: W = %.1f J/mol' % (eps, Wv))
    ax.plot(x, f_ga(x), color='#555555')
    ax.plot(x, f_gb(x), color='#555555')
    ax.plot(x, G_ab(x, Wv), color='red', lw=2, label=r'$\alpha+\beta$')
    ax.plot(x, G_ab(x, 0.0), color='k', ls='--', label=r'$\alpha+\beta$ ($\epsilon=0$)')
    ax.set_title(r'(%s) $\epsilon=%.2f$, $\mathcal{W}=%.2f$ kJ/mol' % (lab, eps, Wv/1000))
    ax.set_xlim(0, 1)
    ax.set_ylim(-60000, 0)                   # as in the published figure
    ax.set_xlabel('$x$')
axs[0].set_ylabel('$g$ (J/mol)')
axs[0].legend(loc='upper center')
fig.tight_layout()
fig.savefig('Fig-5bc.png', dpi=300)
print('saved Fig-5bc.png')
