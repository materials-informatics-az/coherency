"""
Numbers quoted in Sections 3.2, 4.1, 4.2, 5.1 and Appendices A3-A4 for the parameters of Table 1.
Analytical values are compared with the reduced solver.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from solver_reduced import solve_point, three_phase_field

EUTECTIC = dict(x0=(0.2, 0.8, 0.5), a=(50, 50, 50), b=(21, 21, 40), phi=0.0167)
PERITECTIC = dict(x0=(0.5, 0.8, 0.3), a=(50, 50, 50), b=(21, 21, 40), phi=0.0167)
a, phi = 50, 0.0167


def solidus(p, W, X, T_hi=1250.0, T_lo=900.0, tol=1e-3):
    """Lowest temperature at which liquid is present in equilibrium at composition X."""
    while T_hi - T_lo > tol:
        T = 0.5*(T_hi + T_lo)
        if 'L' in solve_point(p, W, T, X)[0]:
            T_hi = T
        else:
            T_lo = T
    return T_hi


T_inv = (40 - 21)/phi
print('T_inv = (b_L - b)/phi = %.1f K' % T_inv)
print('W/b = %.3f and %.3f for W = 1 and 3 kJ/mol' % (1/21, 3/21))

print('\nEutectic three-phase field (Eq. A14, A18):')
for W in (1, 3):
    T_min, T_max, dX = three_phase_field(EUTECTIC, W)
    print('  W=%g: dT = %.1f K (analytic %.1f), field between %.1f and %.1f K; dX = %.3f (analytic %.3f)'
          % (W, T_max - T_min, W/(2*phi), T_min, T_max, dX, 0.6 - W/(2*a*0.3)))

print('\nSolidus depression at W = 3 kJ/mol (Appendix A3):')
for X in (0.21, 0.25, 0.5, 0.75):
    print('  X=%.2f: %.1f K' % (X, solidus(EUTECTIC, 0, X) - solidus(EUTECTIC, 3, X)))

print('\nClosure of the coherent alpha+beta field, W_c = a (dx0)^2 (Appendix A4), checked at 300 K:')
for name, p, dx0 in (('eutectic', EUTECTIC, 0.6), ('peritectic', PERITECTIC, 0.3)):
    Wc = a*dx0**2
    n = [sum(solve_point(p, f*Wc, 300.0, X)[0] == 'ab' for X in np.linspace(0, 1, 1001)) for f in (0.99, 1.01)]
    print('  %-10s W_c = %.1f kJ/mol; compositions with alpha+beta at 0.99 W_c: %d, at 1.01 W_c: %d'
          % (name, Wc, *n))
print('  eutectic three-phase minimum closes at 4a|d_a d_b| = %.1f kJ/mol' % (4*a*0.3*0.3))
