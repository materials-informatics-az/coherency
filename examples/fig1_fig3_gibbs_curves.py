"""
Figs. 1 and 3: Gibbs free energy curves of the eutectic and peritectic systems
at three temperatures (rows) and three coherency energies W (columns).
Uses the sympy solver.

Colors as in the published figures: purple g_alpha, green g_beta, blue g_L; light purple alpha+L and
teal beta+L common tangents; orange alpha+beta; magenta alpha+beta+L stationary states.
Solid segments are the lowest-energy states at a given X, dashed segments are not. At W = 0 and
T = T_inv, the degenerate three-phase tie line (spanning all three x0 for Table 1) is drawn in magenta.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
from solver_sympy import Model

EUTECTIC = dict(x0=(0.2, 0.8, 0.5), a=(50, 50, 50), b=(21, 21, 40), phi=0.0167)    # Table 1
PERITECTIC = dict(x0=(0.5, 0.8, 0.3), a=(50, 50, 50), b=(21, 21, 40), phi=0.0167)  # Table 1
Ws = [0, 1, 3]            # kJ/mol, i.e., 0, 0.05b and 0.15b with b = 21 kJ/mol
Wlabels = ['0', '0.05b', '0.15b']
dT = 6.0                  # K, rows are T_inv + dT, T_inv, T_inv - dT

colors = {'a': '#926abe', 'b': '#439458', 'L': '#4494c8', 'aL': '#a696ff', 'bL': '#38c19e',
          'ab': '#ff9933', 'abL': '#e9208d'}
x = np.linspace(0, 1, 1001)


def plot_system(p, name, fname):
    T_inv = (p['b'][2] - p['b'][0])/p['phi']       # invariant temperature, 1137.7 K
    rows = [(T_inv + dT, r'$T>T_%s$' % name), (T_inv, r'$T=T_%s$' % name), (T_inv - dT, r'$T<T_%s$' % name)]
    fig, axs = plt.subplots(3, 3, figsize=(10, 9), sharex=True, sharey=True)
    for j, W in enumerate(Ws):
        m = Model(p, W)
        for i, (T, rlabel) in enumerate(rows):
            ax = axs[i, j]
            G = m.energies(T, x)
            Gmin = np.min(np.vstack(list(G.values())), axis=0)
            for k in ('a', 'b', 'L'):
                ax.plot(x, G[k], color=colors[k])
            for k in ('aL', 'bL', 'ab', 'abL'):
                low = np.where(G[k] <= Gmin + 1e-9, G[k], np.nan)
                high = np.where(G[k] > Gmin + 1e-9, G[k], np.nan)
                ax.plot(x, high, color=colors[k], ls='--')
                ax.plot(x, low, color=colors[k], lw=2)
            if W == 0 and T == T_inv:              # degenerate three-phase tie line at T_inv
                on = (x >= min(p['x0'])) & (x <= max(p['x0']))
                ax.plot(x[on], Gmin[on], color=colors['abL'], lw=2)
            ax.set_xlim(0, 1)
            ax.set_ylim(20.85, 22)
            if i == 0:
                ax.set_title('(%s) $\\mathcal{W}=%s$' % ('abc'[j], Wlabels[j]))
            if j == 0:
                ax.set_ylabel(rlabel + '\n$g$ (kJ/mol)')
            if i == 2:
                ax.set_xlabel('$x$')
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    print('saved', fname)


plot_system(EUTECTIC, 'E', 'Fig-1.png')
plot_system(PERITECTIC, 'P', 'Fig-3.png')
