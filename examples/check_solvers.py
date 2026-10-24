"""
Verification of the two implementations (Section 3.3):
 1. sympy and reduced solvers select the same state at every grid point of Figs. 2 and 4;
 2. the selected energies agree with direct minimization of Eq. (A6) over the phase fractions,
    which does not use Eq. (5), at random points (symmetric and asymmetric parameter sets);
 3. sympy (solved point by point) and reduced solvers agree for an asymmetric set;
 4. peritectic three-phase stationary states are saddle points (det H < 0, Appendix A5).
Takes several minutes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import solver_sympy as ss
import solver_reduced as sr

EUTECTIC = dict(x0=(0.2, 0.8, 0.5), a=(50, 50, 50), b=(21, 21, 40), phi=0.0167)
PERITECTIC = dict(x0=(0.5, 0.8, 0.3), a=(50, 50, 50), b=(21, 21, 40), phi=0.0167)
ASYM = dict(x0=(0.2, 0.8, 0.5), a=(55, 50, 45), b=(22, 20, 40), phi=0.0167)      # Table 2, set (iv)
ASYM_P = dict(x0=(0.5, 0.8, 0.3), a=(55, 50, 45), b=(22, 20, 40), phi=0.0167)
Xs = np.linspace(0, 1, 201)
Ts = np.linspace(850, 1350, 201)
rng = np.random.default_rng(0)

print('1. Phase maps of Figs. 2 and 4: sympy vs reduced solver')
for name, p in (('eutectic', EUTECTIC), ('peritectic', PERITECTIC)):
    for W in (0, 1, 3):
        m1 = ss.Model(p, W).phase_map(Xs, Ts)
        m2 = sr.phase_map(p, W, Xs, Ts)
        print('   %-10s W=%g: %d of %d grid points differ' % (name, W, np.sum(m1 != m2), m1.size), flush=True)

print('2. Reduced solver vs direct minimization over the phase fractions (random points)')
for name, p in (('eutectic', EUTECTIC), ('peritectic', PERITECTIC), ('eutectic (iv)', ASYM), ('peritectic (iv)', ASYM_P)):
    for W in (1, 3):
        dG = []
        for X, T in zip(rng.uniform(0.02, 0.98, 20), rng.uniform(950, 1300, 20)):
            dG.append(sr.solve_point(p, W, T, X)[1] - sr.minimize_direct(p, W, T, X))
        print('   %-15s W=%g: max |dG| = %.1e kJ/mol' % (name, W, np.max(np.abs(dG))), flush=True)

print('3. Asymmetric set (iv): sympy (point by point) vs reduced solver')
dG = []
for X, T, W in zip(rng.uniform(0.05, 0.95, 6), rng.uniform(1050, 1250, 6), (1, 3)*3):
    G1 = ss.Model(ASYM, W, X0=X, T0=T).energies(T, np.array([X]))
    lab1 = min(G1, key=lambda k: G1[k][0])
    lab2, G2 = sr.solve_point(ASYM, W, T, X)
    dG.append(G1[lab1][0] - G2)
    print('   X=%.3f T=%.1f W=%g: %s / %s' % (X, T, W, lab1, lab2), flush=True)
print('   max |dG| = %.1e kJ/mol' % np.max(np.abs(dG)))

print('4. Hessian determinant of the peritectic three-phase stationary states')
for W in (1, 3):
    dets = []
    for T in np.linspace(1100, 1180, 9):
        for X in np.linspace(0.3, 0.8, 26):
            G, s = sr.three_phase(PERITECTIC, W, T, X)
            if s is not None:
                dets.append(sr.hessian_det(PERITECTIC, W, s['x'], s['z']))
    print('   W=%g: %d states, max det H = %.3g (negative: saddle points)' % (W, len(dets), max(dets)))
