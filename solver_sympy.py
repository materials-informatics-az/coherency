"""
Symbolic (sympy) solver for the coherent alpha + beta + L model (Section 2, Eq. 5).

The equilibrium conditions of Eq. (5) are solved once symbolically, with the alloy composition X
and temperature T kept as symbols, and then evaluated numerically. All roots are checked and the
lowest-energy valid root is kept.

Parameters are a dict with tuples ordered (alpha, beta, L):
    p = dict(x0=(0.2, 0.8, 0.5), a=(50, 50, 50), b=(21, 21, 40), phi=0.0167)
Energies in kJ/mol, T in K, W in kJ/mol.
"""
import numpy as np
from sympy import symbols, diff, solve, lambdify, nsimplify

PHASES = ['a', 'b', 'L', 'aL', 'bL', 'ab', 'abL']   # labels used by both solvers

x_a, x_b, x_L, z_a, z_b, X, T = symbols('x_a x_b x_L z_a z_b X T')


def gibbs(p):
    """Symbolic Gibbs energies of alpha, beta and L (Eqs. 6-7)."""
    q = {k: [nsimplify(v, rational=True) for v in p[k]] for k in ('x0', 'a', 'b')}
    phi = nsimplify(p["phi"], rational=True)
    g_a = q['a'][0]*(x_a - q['x0'][0])**2 + q['b'][0]
    g_b = q['a'][1]*(x_b - q['x0'][1])**2 + q['b'][1]
    g_L = q['a'][2]*(x_L - q['x0'][2])**2 + q['b'][2] - phi*T
    return g_a, g_b, g_L


class Model:
    """Solves the stationary conditions once for given parameters and W.

    By default X and T stay symbolic, which is fast for equal curvatures (Table 1). For unequal
    curvatures the conditions become cubic and the symbolic solution is impractically slow;
    pass a single point (X0, T0) to solve with numbers substituted instead.
    """

    def __init__(self, p, W, X0=None, T0=None):
        W = nsimplify(W, rational=True)
        g_a, g_b, g_L = gibbs(p)
        self.g = [lambdify((x_a, T), g_a), lambdify((x_b, T), g_b), lambdify((x_L, T), g_L)]
        self.W = float(W)
        pt = {}
        if X0 is not None:
            pt[X] = nsimplify(X0, rational=True)
        if T0 is not None:
            pt[T] = nsimplify(T0, rational=True)
        g_a, g_b, g_L = g_a.subs(pt), g_b.subs(pt), g_L.subs(pt)

        # alpha + L and beta + L: common tangents (no coherency energy)
        s_aL = solve([diff(g_a, x_a) - diff(g_L, x_L),
                      diff(g_a, x_a) - (g_a - g_L)/(x_a - x_L)], [x_a, x_L], dict=True)
        s_bL = solve([diff(g_b, x_b) - diff(g_L, x_L),
                      diff(g_b, x_b) - (g_b - g_L)/(x_b - x_L)], [x_b, x_L], dict=True)
        self.aL = [lambdify((X, T), [s[x_a].evalf(), s[x_L].evalf()]) for s in s_aL]
        self.bL = [lambdify((X, T), [s[x_b].evalf(), s[x_L].evalf()]) for s in s_bL]

        # alpha + beta with coherency energy W z_a z_b, z_a = (X - x_b)/(x_a - x_b)
        za = ((X - x_b)/(x_a - x_b)).subs(pt)
        s_ab = solve([diff(g_a, x_a) - diff(g_b, x_b),
                      diff(g_b, x_b) - (g_a - g_b + W*(1 - 2*za))/(x_a - x_b)], [x_a, x_b], dict=True)
        self.ab = [lambdify((X, T), [s[x_a].evalf(), s[x_b].evalf()]) for s in s_ab]

        # alpha + beta + L: Eq. (5) together with the mass balance
        self.abL = []
        if W != 0:
            eqs = [diff(g_a, x_a) - diff(g_b, x_b),
                   diff(g_b, x_b) - diff(g_L, x_L),
                   diff(g_L, x_L) - (g_a - g_L + W*z_b)/(x_a - x_L),
                   diff(g_L, x_L) - (g_b - g_L + W*z_a)/(x_b - x_L),
                   (X - ((1 - z_a - z_b)*x_L + z_a*x_a + z_b*x_b)).subs(pt)]
            s_abL = solve(eqs, [x_a, x_b, x_L, z_a, z_b], dict=True)
            self.abL = [lambdify((X, T), [s[v].evalf() for v in (x_a, x_b, x_L, z_a, z_b)]) for s in s_abL]

    def energies(self, Tv, Xs):
        """Energies of the seven candidate states on the composition array Xs at temperature Tv
        (dict of arrays, inf where the state does not exist)."""
        Xs = np.asarray(Xs, dtype=float)
        g_a, g_b, g_L = self.g
        G = {'a': g_a(Xs, Tv), 'b': g_b(Xs, Tv), 'L': g_L(Xs, Tv)}
        for k in ('aL', 'bL', 'ab', 'abL'):
            G[k] = np.full(len(Xs), np.inf)
        with np.errstate(all='ignore'):
            for key, sols, g_s in (('aL', self.aL, g_a), ('bL', self.bL, g_b), ('ab', self.ab, g_b)):
                for f in sols:
                    (xs, xo), ok = _evaluate(f, Xs, Tv)
                    z = (Xs - xo)/(xs - xo)           # fraction of the first phase
                    if key == 'ab':
                        Gk = z*g_a(xs, Tv) + (1 - z)*g_b(xo, Tv) + self.W*z*(1 - z)
                    else:
                        Gk = z*g_s(xs, Tv) + (1 - z)*g_L(xo, Tv)
                    ok &= (z > 0) & (z < 1)
                    G[key] = np.where(ok & (Gk < G[key]), Gk, G[key])
            for f in self.abL:
                (xa, xb, xl, za, zb), ok = _evaluate(f, Xs, Tv, ncomp=3)
                zl = 1 - za - zb
                Gk = za*g_a(xa, Tv) + zb*g_b(xb, Tv) + zl*g_L(xl, Tv) + self.W*za*zb
                ok &= (za > 0) & (zb > 0) & (zl > 0)
                G['abL'] = np.where(ok & (Gk < G['abL']), Gk, G['abL'])
        return G

    def labels(self, Tv, Xs):
        """Index (into PHASES) of the lowest-energy state at each composition."""
        G = self.energies(Tv, Xs)
        return np.argmin(np.vstack([G[k] for k in PHASES]), axis=0)

    def phase_map(self, Xs, Ts):
        return np.array([self.labels(Tv, Xs) for Tv in Ts])


def _evaluate(f, Xs, Tv, ncomp=2):
    """Evaluate a lambdified root (complex arithmetic); keep real roots with 0 <= x <= 1."""
    vals = [np.broadcast_to(np.asarray(v, dtype=complex), Xs.shape) for v in f(Xs + 0j, Tv + 0j)]
    ok = np.all([np.abs(v.imag) < 1e-9 for v in vals], axis=0)
    vals = [v.real for v in vals]
    for x in vals[:ncomp]:
        ok &= (x >= -1e-12) & (x <= 1 + 1e-12)
    return vals, ok
