# Eutectic and peritectic equilibria in coherent binary alloys

Python code for the paper

> S. Hassan, J. Wang, T. Meissner, P. A. Deymier, M. I. Latypov,
> *Eutectic and peritectic equilibria in coherent binary alloys*,
> Acta Materialia (2026) 122841. https://doi.org/10.1016/j.actamat.2026.122841

The code computes phase equilibria of a binary system with two coherent solid phases (α, β) and an incoherent liquid (L), with the coherency energy W z_α z_β added to the chemical Gibbs energies.

## Installation

Requires Python ≥ 3.9 with numpy, scipy, sympy and matplotlib:
```
pip install -r requirements.txt
```

## Structure

```
solver_sympy.py      symbolic solver (Section 3.3)
solver_reduced.py    reduced (polynomial) solver (Appendices A1–A2)
examples/            scripts that reproduce the figures, tables and numbers of the paper
```

| File | Method | Used for |
|---|---|---|
| `solver_sympy.py` | The stationary conditions of Eq. (5) and the two-phase conditions are solved symbolically with `sympy`, then evaluated on (X, T) grids | Figs. 1–4 |
| `solver_reduced.py` | All compositions follow from the diffusion potential μ, and each phase combination reduces to a polynomial in μ of degree ≤ 3. Also has direct minimization over the phase fractions (Eq. A6), the Hessian test (Appendix A5), and the extents of the stable three-phase field | Table 2, verification, numbers in the text |

Both solvers take the parameters as a dict with tuples ordered (α, β, L), e.g. the eutectic of Table 1:

```python
import solver_reduced as sr

p = dict(x0=(0.2, 0.8, 0.5), a=(50, 50, 50), b=(21, 21, 40), phi=0.0167)
label, G = sr.solve_point(p, W=3, T=1100, X=0.5)      # 'abL'
```
Energies are in kJ/mol (J/mol for Cu–Ag), T in K.

The symbolic solution with X and T kept as symbols is fast for equal curvatures (Table 1). For unequal curvatures the conditions are cubic and the fully symbolic solution is impractically slow, so
`solver_sympy.Model(p, W, X0=..., T0=...)` solves at a single point instead (used in `check_solvers.py`).

## Examples

Run from the `examples/` folder, e.g. `cd examples && python fig2_fig4_phase_maps.py`. Output files
are written to the current folder.

| Script | Reproduces | Run time |
|---|---|---|
| `fig1_fig3_gibbs_curves.py` | Figs. 1 and 3 (Gibbs energy curves) | seconds |
| `fig2_fig4_phase_maps.py` | Figs. 2 and 4 (phase equilibria maps); also saves the label grids as text | seconds |
| `fig5_cu_ag.py` | Fig. 5(b,c) (Cu–Ag parabolas, W at 1 and 5 % misfit). Panel (a) is from Thermo-Calc | seconds |
| `table2_sensitivity.py` | Table 2 (ΔT, ΔX of the asymmetric sets) and the absence of a stable three-phase state for all peritectic sets | ~5 min |
| `table3_estimates.py` | Table 3, Appendix A6, and the ΔT estimates of Section 5.3 | < 1 s |
| `check_numbers.py` | T_inv = 1137.7 K, ΔT = 29.9 / 89.8 K, ΔX, solidus depression (89.8 K, 18.0 K at X = 0.21), closure of the α+β field at W_c = 18 and 4.5 kJ/mol | < 1 min |
| `check_solvers.py` | Section 3.3 verification: identical labels of both solvers on the grids of Figs. 2 and 4; agreement with direct minimization; sympy vs reduced solver for an asymmetric set; det H < 0 for peritectic three-phase states | ~2 min |

Notes:
* W = 1 and 3 kJ/mol are the values labeled as 0.05b and 0.15b (b = 21 kJ/mol).
* The Gibbs curve rows are at T_inv ± 6 K (set `dT` in `fig1_fig3_gibbs_curves.py`). At W = 0 and T = T_inv the three-phase state is degenerate (any fractions along the tie line); it is drawn as the tie line.

## Citation

Please cite this work as:

```bibtex
@article{hassan2026eutectic,
  title={Eutectic and peritectic equilibria in coherent binary alloys},
  author={Hassan, Samiah and Wang, Jiayang and Meissner, Teddy and Deymier, Pierre A and Latypov, Marat I},
  journal={Acta Materialia},
  pages={122841},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.actamat.2026.122841}
}
```

Citation metadata is also in [`CITATION.cff`](CITATION.cff).

## License

See [`LICENSE`](LICENSE) (non-commercial educational and research use).
