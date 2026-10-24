"""
Table 2: temperature extent dT and maximum composition width dX of the stable three-phase field
for the asymmetric parameter sets, and the check that no peritectic set has a stable
three-phase state (Section 4.3). Uses the reduced solver. Takes several minutes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from solver_reduced import three_phase_window, three_phase_field

PHI = 0.0167
Ws = [1, 3]      # kJ/mol

# name, (a_alpha, a_beta, a_L), (b_alpha, b_beta, b_L), x0_L eutectic, x0_L peritectic
SETS = [('baseline', (50, 50, 50), (21, 21, 40), 0.5, 0.3),
        ('(i)', (45, 50, 55), (21, 21, 40), 0.5, 0.3),
        ('(ii)', (50, 50, 50), (20, 22, 40), 0.5, 0.3),
        ('(iii)', (45, 50, 55), (20, 22, 40), 0.5, 0.3),
        ('(iv)', (55, 50, 45), (22, 20, 40), 0.5, 0.3),
        ('(v)', (50, 50, 50), (21, 21, 40), 0.4, 0.2)]


print('Eutectic system: stable three-phase field')
print('%-9s %8s %8s %8s %8s' % ('set', 'dT(W=1)', 'dT(W=3)', 'dX(W=1)', 'dX(W=3)'))
for name, a, b, xL_e, xL_p in SETS:
    p = dict(x0=(0.2, 0.8, xL_e), a=a, b=b, phi=PHI)
    res = [three_phase_field(p, W) for W in Ws]
    print('%-9s %8.1f %8.1f %8.3f %8.3f' % (name, *[r[1] - r[0] for r in res], *[r[2] for r in res]), flush=True)

print('\nPeritectic system: stable three-phase state found?')
for name, a, b, xL_e, xL_p in SETS:
    p = dict(x0=(0.5, 0.8, xL_p), a=a, b=b, phi=PHI)
    for W in Ws:
        found = any(three_phase_window(p, W, T, n=101) is not None for T in np.arange(850, 1351, 2.0))
        print('%-9s W=%g: %s' % (name, W, found), flush=True)

print('\nEq. (A14) for equal curvatures: baseline dT = %.1f, %.1f K; set (v) dT = %.1f, %.1f K'
      % (1/(2*PHI), 3/(2*PHI), 1/PHI*0.4/0.6, 3/PHI*0.4/0.6))
