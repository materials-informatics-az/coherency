"""
Table 3 and Section 5.3: coherency energy estimates for real systems (Appendix A6)
and the order-of-magnitude temperature extents dT = W/(2 dS_f).
"""
import numpy as np

NA = 6.02214076e23
b = 21.0                                     # kJ/mol, reference energy of the model solids


def W_iso(E, nu, V, eps):
    """Eq. (1): W = V E eps^2/(1 - nu) in kJ/mol (E in Pa, V in m^3/mol)."""
    return V*E*eps**2/(1 - nu)/1000


# metallic systems: polycrystalline E, nu (Callister) and molar volumes of the pure metals
metals = {'Cu': (110e9, 0.34, 7.1e-6), 'Ni': (207e9, 0.31, 6.59e-6), 'W': (407e9, 0.28, 9.53e-6)}
print('System           eps      W (kJ/mol)  W/b')
for name, metal, eps in [('Cu-Ag', 'Cu', 0.01), ('Cu-Ag', 'Cu', 0.05), ('Ni gamma/gamma\'', 'Ni', 0.005),
                         ('Refractory (W)', 'W', 0.01), ('Refractory (W)', 'W', 0.05)]:
    W = W_iso(*metals[metal], eps)
    print('%-16s %-8s %-11.3f %.3f' % (name, eps, W, W/b))

# LiFePO4: W = B V_m dx^2/2 with B = 0.19 GPa (Cogswell and Bazant), Eq. (A22)
Vm = 10.33*6.01*4.69e-30*NA/4                # m^3/mol, orthorhombic cell with 4 formula units
for dx in (0.8, 1.0):
    W = 0.5*0.19e9*Vm*dx**2/1000
    print('%-16s dx=%-5s %-11.2f %.2f' % ('LiFePO4', dx, W, W/b))
print('  V_m(LiFePO4) = %.3g m^3/mol, Omega = 11.1 kJ/mol -> W/Omega = %.2f-%.2f'
      % (Vm, 0.5*0.19e9*Vm*0.64/1000/11.1, 0.5*0.19e9*Vm/1000/11.1))


# Mg2Si-Mg2Sn: E, nu and lattice parameters compiled by Yi et al. (2018)
def vrh(C11, C12, C44):
    """Voigt-Reuss-Hill isotropic E and nu of a cubic crystal (C in GPa)."""
    K = (C11 + 2*C12)/3
    G = ((C11 - C12 + 3*C44)/5 + 5*(C11 - C12)*C44/(4*C44 + 3*(C11 - C12)))/2
    return 9*K*G/(3*K + G), (3*K - 2*G)/(2*(3*K + G))


Y_tab = np.mean([57/(1 - 0.2058), 118/(1 - 0.1464)])                   # E/(1 - nu) in GPa, tabulated E, nu
Y_vrh = np.mean([E/(1 - nu) for E, nu in (vrh(83.71, 39.79, 21.69), vrh(114.07, 19.56, 33.32))])
for a_Si, a_Sn in [(6.350, 6.761), (6.431, 6.854), (6.338, 6.765), (6.351, 6.759)]:   # Angstrom
    eps = 2*(a_Sn - a_Si)/(a_Sn + a_Si)                                  # Eq. (A21)
    V = (a_Si**3 + a_Sn**3)/2*1e-30*NA/4                                 # per mole of formula units
    print('%-16s %-8.4f %.1f-%.1f   (V = %.3g m^3/mol)' % ('Mg2Si-Mg2Sn', eps, V*Y_tab*1e6*eps**2,
                                                          V*Y_vrh*1e6*eps**2, V))
print('  Omega = 3 x 5.95 = %.1f kJ/mol per formula unit' % (3*5.95))

# PbTe-PbS: difference of the leading coefficients of the incoherent and coherent fits (Doak 2012)
meV = 0.0964853                              # kJ/mol per meV/cation
W = (289.5 - 74.62)*meV
print('%-16s %-8s %-11.1f %.2f   (incoherent coefficient %.1f kJ/mol)' % ('PbTe-PbS', 0.084, W, W/b, 289.5*meV))

# order-of-magnitude temperature extent dT = W/(2 dS_f), Section 5.3
dS_CuAg = (9.7684 + 9.1477)/2                # J/(mol K), mean entropy of fusion of Cu and Ag (SGTE)
dS_Ni = 17480/1728.3                         # J/(mol K), Ni
print('\nCu-Ag: dS_f = %.2f J/(mol K), dT = %.1f K (eps = 0.01), %.0f K (eps = 0.05)'
      % (dS_CuAg, 1000*W_iso(*metals['Cu'], 0.01)/(2*dS_CuAg), 1000*W_iso(*metals['Cu'], 0.05)/(2*dS_CuAg)))
print('Ni gamma/gamma\': dT = %.1f K (eps = 0.005)' % (1000*W_iso(*metals['Ni'], 0.005)/(2*dS_Ni)))
print('Tungsten reaches W = 3 kJ/mol at eps = %.3f; W(tungsten)/W(Cu) = %.1f'
      % (np.sqrt(3/W_iso(*metals['W'], 1.0)), W_iso(*metals['W'], 1)/W_iso(*metals['Cu'], 1)))
