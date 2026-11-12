# S3 binding table

Association constants in L/mol; `I` is the ionic strength (mol/L) at which the target is
defined and the model evaluated. Free energy is -ln(K_total * 1 mol/L) in kT.

| pair | I | K_total target | K_total model | K_inner target | K_inner model | monodentate | multidentate | outer | inner share | dG (kT) | K_total at I = 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ca_acetate | 0.00 | 15.14 | 15.12 | 3.60 | 3.58 | 2.63 | 0.94 | 11.54 | 0.24 | -2.72 | 4.69 |
| Mg_acetate | 0.00 | 18.20 | 18.23 | 3.90 | 3.94 | 3.94 | 0.00 | 14.29 | 0.22 | -2.90 | 5.57 |
| Ca_sulfate | 0.00 | 229.09 | 229.27 | 66.44 | 67.89 | 64.12 | 3.77 | 161.38 | 0.30 | -5.43 | 23.09 |
| Mg_sulfate | 0.00 | 181.97 | 181.51 | 18.20 | 18.07 | 18.05 | 0.02 | 163.45 | 0.10 | -5.20 | 18.31 |
| SO4_guanidinium | 0.00 | 15.00 | 15.22 | - | 9.27 | 0.00 | 0.00 | 5.95 | 0.61 | -2.72 | 4.86 |
| SO4_ammonium | 0.00 | 8.00 | 8.00 | - | 2.82 | 0.00 | 0.00 | 5.18 | 0.35 | -2.08 | 2.51 |
| SO4_imidazolium | 0.00 | 10.00 | 10.02 | - | 4.37 | 0.00 | 0.00 | 5.65 | 0.44 | -2.30 | 3.22 |

For the sulfate-donor pairs 'inner' means a direct hydrogen bond and is reported in the
K_inner model column; there is no separate target for it.

Counterfactual: Mg2+ with the contact energy of Ca2+ and its own desolvation cost has an
inner-sphere share with acetate of 2.6e-06 (K_inner = 3.7e-05 L/mol).

## Measured constants the model was not fitted to

| check | I | measured log K (range) | model log K |
|---|---|---|---|
| Ca_acetate.log_K_total_I0p1 | 0.1 | 0.55 (0.48-0.62) | model 0.67 |
| Mg_acetate.log_K_total_I0p1 | 0.1 | 0.51 (0.46-0.56) | model 0.75 |
| Ca_sulfate.log_K_total_I0p1 | 0.1 | 1.53 (1.48-1.58) | model 1.36 |
| Mg_sulfate.log_K_total_I0p1 | 0.1 | 1.48 (1.46-1.50) | model 1.26 |
| Ca_succinate.log_K_total | 0.0 | 2.00 (1.90-2.10) | gauche 2.62; extended 2.27 |
| Mg_succinate.log_K_total | 0.0 | 2.18 (2.10-2.30) | gauche 2.83; extended 2.36 |

## Contact geometry from Monte Carlo (one ion, one ligand, I = 0.1 mol/L)

Distances are from the ion centre (S for sulfate) to the partner atom (O or N).
'Inner' is inner-sphere for the cations and a direct hydrogen bond for sulfate; the bite
angle is atom-ion-atom when two partner atoms are held.

| ion | ligand | mean inner distance (nm) | s.d. (nm) | mean outer distance (nm) | rms off-axis angle (deg) | mean bite angle (deg) | unbound | outer | inner, one atom | inner, two atoms |
|---|---|---|---|---|---|---|---|---|---|---|
| Ca | acetate | 0.243 | 0.012 | 0.463 | 19.1 | 52.5 | 0.849 | 0.118 | 0.0290 | 0.0042 |
| Mg | acetate | 0.207 | 0.005 | 0.426 | 7.0 | - | 0.842 | 0.126 | 0.0322 | 0.0000 |
| SO4 | methylguanidinium | 0.379 | 0.041 | 0.494 | - | 34.4 | 0.862 | 0.062 | 0.0323 | 0.0441 |
