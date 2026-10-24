"""
Figs. 2 and 4: phase equilibria maps of the eutectic and peritectic systems for W = 0, 0.05b, 0.15b.
Uses the sympy solver; check_solvers.py confirms that the reduced solver gives the same label
at every grid point.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from solver_sympy import Model, PHASES

EUTECTIC = dict(x0=(0.2, 0.8, 0.5), a=(50, 50, 50), b=(21, 21, 40), phi=0.0167)    # Table 1
PERITECTIC = dict(x0=(0.5, 0.8, 0.3), a=(50, 50, 50), b=(21, 21, 40), phi=0.0167)  # Table 1
Ws = [0, 1, 3]                       # kJ/mol, i.e., 0, 0.05b and 0.15b with b = 21 kJ/mol
Wlabels = ['0', '0.05b', '0.15b']
Xs = np.linspace(0, 1, 201)
Ts = np.linspace(850, 1350, 201)

# alpha, beta, L, alpha+L, beta+L, alpha+beta, alpha+beta+L (colors of the published figures)
cmap = ListedColormap(['#926abe', '#439458', '#4494c8', '#b8affc', '#91e5ca', '#c1a58f', '#e9208d'])
names = [r'$\alpha$', r'$\beta$', 'L', r'$\alpha$+L', r'$\beta$+L', r'$\alpha+\beta$', r'L+$\alpha+\beta$']


def plot_system(p, fname):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for j, W in enumerate(Ws):
        lab = Model(p, W).phase_map(Xs, Ts)
        np.savetxt(fname.replace('.png', '_W%g.txt' % W), lab, fmt='%d',
                   header='rows: T = 850..1350 K (201), columns: X = 0..1 (201), labels: ' + ' '.join(PHASES))
        ax = axs[j]
        ax.imshow(lab, origin='lower', cmap=cmap, vmin=0, vmax=6, aspect='auto',
                  extent=[Xs[0], Xs[-1], Ts[0], Ts[-1]], interpolation='nearest')
        for k in range(7):                           # label each region at its centroid
            i, jj = np.nonzero(lab == k)
            if len(i) > 150:
                ax.text(Xs[jj].mean(), Ts[i].mean(), names[k], ha='center', va='center')
        ax.set_title('(%s) $\\mathcal{W}=%s$' % ('abc'[j], Wlabels[j]))
        ax.set_xlabel('$X$')
    axs[0].set_ylabel('$T$ (K)')
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    print('saved', fname)


plot_system(EUTECTIC, 'Fig-2.png')
plot_system(PERITECTIC, 'Fig-4.png')
