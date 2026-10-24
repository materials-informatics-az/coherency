"""
Reduced (polynomial) solver for the coherent alpha + beta + L model (Appendix A1-A2).

All phase compositions follow from one scalar, the diffusion potential mu:
    x_i = x0_i + mu/(2 a_i),    g_i = mu^2/(4 a_i) + b_i   (b_L -> b_L - phi*T)
and the stationary conditions of every phase combination reduce to a polynomial
in mu (degree <= 3), solved here with numpy.

Parameters are a dict with tuples ordered (alpha, beta, L):
    p = dict(x0=(0.2, 0.8, 0.5), a=(50, 50, 50), b=(21, 21, 40), phi=0.0167)
Energies in kJ/mol, T in K, W in kJ/mol.
"""
import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.optimize import minimize, minimize_scalar

PHASES = ['a', 'b', 'L', 'aL', 'bL', 'ab', 'abL']   # labels used by both solvers


def ref_energies(p, T):
    """Minima of g_alpha, g_beta, g_L at temperature T."""
    return [p['b'][0], p['b'][1], p['b'][2] - p['phi']*T]


def g_phase(p, T, i, x):
    """Molar Gibbs energy of phase i (0=alpha, 1=beta, 2=L) at composition x."""
    return p['a'][i]*(x - p['x0'][i])**2 + ref_energies(p, T)[i]


_cache = {}


def polys(p, T):
    """x_i(mu), g_i(mu) and omega_i(mu) = g_i - mu*x_i as polynomials in mu (cached per T)."""
    key = (tuple(p['x0']), tuple(p['a']), tuple(p['b']), p['phi'], T)
    if key not in _cache:
        if len(_cache) > 100000:
            _cache.clear()
        x = [Poly([p['x0'][i], 1/(2*p['a'][i])]) for i in range(3)]
        g = [Poly([ref_energies(p, T)[i], 0, 1/(4*p['a'][i])]) for i in range(3)]
        om = [g[i] - Poly([0, 1])*x[i] for i in range(3)]
        _cache[key] = (x, g, om)
    return _cache[key]


def real_roots(poly):
    poly = poly.trim(1e-14)
    if poly.degree() < 1:
        return []
    r = poly.roots()
    return r[np.abs(r.imag) < 1e-9].real


def ok_comp(*xs):
    return all(-1e-12 <= xi <= 1 + 1e-12 for xi in xs)


def two_phase(p, W, T, X, i, j):
    """Lowest stationary state of the i+j mixture at (X, T); W applies only to alpha+beta."""
    x, g, om = polys(p, T)
    w = W if (i, j) == (0, 1) else 0.0
    dx = x[i] - x[j]
    # d/dz_i [z_i g_i + (1-z_i) g_j + w z_i (1-z_i)] = 0 with z_i = (X - x_j)/(x_i - x_j)
    poly = (om[i] - om[j])*dx + w*(dx - 2*(X - x[j]))
    best = (np.inf, None)
    for mu in real_roots(poly):
        xi, xj = x[i](mu), x[j](mu)
        if abs(xi - xj) < 1e-12 or not ok_comp(xi, xj):
            continue
        zi = (X - xj)/(xi - xj)
        if 0 < zi < 1:
            G = zi*g[i](mu) + (1 - zi)*g[j](mu) + w*zi*(1 - zi)
            if G < best[0]:
                best = (G, dict(mu=mu, x=(xi, xj), z=zi))
    return best


def three_phase(p, W, T, X):
    """Lowest stationary alpha+beta+L state at (X, T) (Eq. A11 with the mass balance)."""
    if W <= 0:
        return (np.inf, None)
    x, g, om = polys(p, T)
    za, zb = (om[2] - om[1])/W, (om[2] - om[0])/W
    poly = za*(x[0] - x[2]) + zb*(x[1] - x[2]) + x[2] - X      # mass balance
    best = (np.inf, None)
    for mu in real_roots(poly):
        z = np.array([za(mu), zb(mu), 1 - za(mu) - zb(mu)])
        xs = [x[k](mu) for k in range(3)]
        if np.all(z > 0) and ok_comp(*xs):
            G = sum(z[k]*g[k](mu) for k in range(3)) + W*z[0]*z[1]
            if G < best[0]:
                best = (G, dict(mu=mu, x=xs, z=z))
    return best


def energies(p, W, T, X):
    """Energies of the seven candidate states at (X, T); inf if the state does not exist."""
    G = {PHASES[k]: g_phase(p, T, k, X) for k in range(3)}
    G['aL'] = two_phase(p, W, T, X, 0, 2)[0]
    G['bL'] = two_phase(p, W, T, X, 1, 2)[0]
    G['ab'] = two_phase(p, W, T, X, 0, 1)[0]
    G['abL'] = three_phase(p, W, T, X)[0]
    return G


def solve_point(p, W, T, X):
    """Equilibrium label and energy at (X, T)."""
    G = energies(p, W, T, X)
    lab = min(G, key=G.get)
    return lab, G[lab]


def phase_map(p, W, Xs, Ts):
    """Array of labels (indices into PHASES) on the (T, X) grid."""
    return np.array([[PHASES.index(solve_point(p, W, T, X)[0]) for X in Xs] for T in Ts])


def hessian_det(p, W, x, z):
    """det of the Hessian of G(z_alpha, z_beta) at a three-phase state (Appendix A5)."""
    S = sum(z[k]/(2*p['a'][k]) for k in range(3))
    return -W*(2*(x[0] - x[2])*(x[1] - x[2])/S + W)


# ---------- direct minimization over the phase fractions (Eq. A6), no Eq. (5) ----------

def G_direct(p, W, T, X, za, zb):
    z = np.array([za, zb, 1 - za - zb])
    S = sum(z[k]/(2*p['a'][k]) for k in range(3))
    M = X - sum(z[k]*p['x0'][k] for k in range(3))
    return M**2/(2*S) + np.dot(z, ref_energies(p, T)) + W*za*zb


def minimize_direct(p, W, T, X, n=101):
    """Grid search over the fraction triangle followed by a bounded local polish."""
    best = (np.inf, None)
    for za in np.linspace(0, 1, n):
        for zb in np.linspace(0, 1 - za, max(int(round((1 - za)*(n - 1))) + 1, 1)):
            G = G_direct(p, W, T, X, za, zb)
            if G < best[0]:
                best = (G, (za, zb))
    r = minimize(lambda z: G_direct(p, W, T, X, z[0], z[1]), best[1], method='SLSQP',
                 bounds=[(0, 1), (0, 1)], constraints=[{'type': 'ineq', 'fun': lambda z: 1 - z[0] - z[1]}],
                 options={'ftol': 1e-14})
    return min(best[0], r.fun)


# ---------- stable part of the three-phase field (Table 2, Appendix A3-A4) ----------

def three_phase_window(p, W, T, n=401):
    """(Xmin, Xmax) of stable three-phase states at temperature T, or None.

    The stationary three-phase branch is parametrized by mu (Eq. A11): z_alpha(mu), z_beta(mu),
    X(mu). A state is stable if its energy is below all competing states at the same X.
    """
    x, g, om = polys(p, T)
    za, zb = (om[2] - om[1])/W, (om[2] - om[0])/W
    zL = 1 - za - zb
    # mu intervals with all fractions positive: bounded by the roots of za, zb, zL
    cuts = np.sort(np.concatenate([real_roots(q) for q in (za, zb, zL)]))
    Xs = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        m = 0.5*(lo + hi)
        if min(za(m), zb(m), zL(m)) <= 0:
            continue
        mus = np.linspace(lo, hi, n)
        X_of = [za(mu)*x[0](mu) + zb(mu)*x[1](mu) + zL(mu)*x[2](mu) for mu in mus]
        stable = np.zeros(n, bool)
        for k in range(1, n - 1):
            mu, X = mus[k], X_of[k]
            if not (0 <= X <= 1 and ok_comp(*[x[q](mu) for q in range(3)])):
                continue
            G3 = za(mu)*g[0](mu) + zb(mu)*g[1](mu) + zL(mu)*g[2](mu) + W*za(mu)*zb(mu)
            Gs = energies(p, W, T, X)
            Gs.pop('abL')
            stable[k] = G3 < min(Gs.values()) - 1e-10
        # the ends of the interval (one fraction -> 0) tie with a two-phase state; they belong
        # to the stable window if their neighbours do
        stable[0], stable[-1] = stable[1], stable[-2]
        Xs += [X_of[k] for k in range(n) if stable[k]]
    return (min(Xs), max(Xs)) if Xs else None


def three_phase_field(p, W, T_range=(850, 1350)):
    """(T_min, T_max, maximum width in X) of the stable three-phase field, or None."""
    def width(T):
        w = three_phase_window(p, W, T)
        return 0.0 if w is None else w[1] - w[0]

    def edge(T_in, T_out, tol=1e-3):           # bisection for the end of the field
        while abs(T_in - T_out) > tol:
            T = 0.5*(T_in + T_out)
            if three_phase_window(p, W, T, n=41) is None:
                T_out = T
            else:
                T_in = T
        return T_in

    Ts = [T for T in np.arange(*T_range, 1.0) if three_phase_window(p, W, T, n=41) is not None]
    if not Ts:
        return None
    T_min, T_max = edge(min(Ts), min(Ts) - 1), edge(max(Ts), max(Ts) + 1)
    grid = np.linspace(T_min, T_max, 51)
    ws = [width(T) for T in grid]
    k = int(np.argmax(ws))
    r = minimize_scalar(lambda T: -width(T), bounds=(grid[max(k - 1, 0)], grid[min(k + 1, 50)]),
                        method='bounded', options={'xatol': 1e-3})
    return T_min, T_max, max(max(ws), -r.fun)     # the maximum may sit at a corner (e.g. T_min)
