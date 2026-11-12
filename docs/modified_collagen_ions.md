# Explicit divalent ions for modified collagen (`modcoll/ions`)

Mg2+, Ca2+ and SO4 2- as rigid multisite particles around fixed protein ligand
atoms, sampled by grand-canonical Monte Carlo, with a countable definition of a
bridge. This is milestone M2 and suite item S3 of the modified-collagen work,
plus the sampler the pair-level runs (M3/S4) will call.

Units everywhere: nm, kT at 298.15 K, elementary charges, mol/L.

```
modcoll/ions/
  params.toml     every number, with units, source and status
  params.py       loader, Debye length, numeric tables
  geometry.py     polyhedra, quaternions, ligand groups, the LigandSet interface
  energy.py       the energy function (numba kernels)
  sampler.py      Metropolis Monte Carlo, grand canonical
  calibrate.py    association constants by quadrature; fit of the contact terms
  bridges.py      contacts, bridges, lifetimes
  validate_s3.py  S3 tables and figures -> results/modcoll/ions/
```

## The model, honestly

**There is no water in it.** Everything water does is folded into four devices:
a dielectric constant, a Debye length for the monovalent salt, a two-valued
state on each coordination site of a cation (water there, or not), and short
range contact energies that were fitted until single-pair association constants
came out right. None of the contact energies is a physical bond energy. What
the model can claim is narrower and is listed under *What is calibrated*; what
it cannot represent is listed under *Known limits*.

### Bodies

| Body | Geometry | Sites | Charge |
|---|---|---|---|
| Mg2+ | rigid octahedron, Mg-O 0.207 nm | 6 coordination sites, each W or L | +2 on the centre |
| Ca2+ | pentagonal bipyramid, Ca-O 0.238 nm (octahedron or square antiprism selectable) | 7 (6, 8), each W or L | +2 on the centre |
| SO4 2- | rigid tetrahedron, S-O 0.1495 nm | 4 oxygens, no state | -1/2 on each oxygen |

A body is a position and a unit quaternion. A cation site is in state **W** (it
holds a water) or **L** (the water has left; a ligand atom may sit there).

The published multisite ("dummy atom") models supply the geometry: six dummies
on an octahedron for Mg2+ (Oelschlaeger et al. 2007; Saxena & Sept 2013; Duarte
et al. 2014), seven on a pentagonal bipyramid for Ca2+ (Saxena & Sept 2013).
Their charge partitions are not used by default. They range from 0 to -3.69 e
on the centre between models and exist to fit explicit-water hydration free
energies (Li & Merz 2017); in continuum water an octahedral or bipyramidal
arrangement has no dipole or quadrupole, and moving the charge from the centre
to points 0.09 nm out changes the energy of a ligand oxygen at contact by less
than 0.3 kT between an on-site and an off-site direction (tested). The layout is
one parameter (`centre_charge`) if someone wants it.

The protein side is a flat list of atoms, `LigandSet`: position, type, charge,
owner-molecule id, group id, and a plane normal for planar donors. That list is
the whole interface to the site model. Types and charges:

| Group | Atoms | Charge |
|---|---|---|
| carboxylate (Asp, Glu, succinyl-Lys, C-terminus) | two O, 0.222 nm apart, and the carbon | -1/2 per O |
| carboxylic acid (protonated) | two O and the carbon | 0; not a ligand |
| guanidinium (Arg) | three coplanar N, 0.230 nm apart, and the carbon | +1/3 per N |
| ammonium (unmodified Lys) | one N | +1 |
| imidazolium (His) | two N, 0.22 nm apart | +1/2 each |

The carbon of a carboxylate or guanidinium is an uncharged, excluded-volume
atom. It keeps an ion from sitting between the oxygens closer than a real one
can, and the contact energies were calibrated with it in place.

`place_on_surface(s, phi, rho, kinds, mol)` puts one group per site on a
cylinder using the axial and azimuthal coordinates of
`tools/azimuthal_profile.py` (`s` = z in nm, `phi` = atan2(y, x)), so a site
table built with the existing sequence-to-surface logic can be passed in
unchanged.

### Energy

Every term is a one-body term or a pair term between two rigid bodies, smooth
with compact support. With B(x) = (1 - x^2)^2 for |x| < 1 and zero outside, and
WCA(r; r_min) the Weeks-Chandler-Andersen repulsion that starts at r_min:

**Electrostatics.** Between every pair of charged points on different bodies,

    u = l_B q1 q2 exp(-kappa r) / r,     l_B = 0.71 nm,
    kappa^2 = 8 pi l_B N_A I,            1/kappa = 0.305 / sqrt(I) nm.

I is the ionic strength of the *implicit monovalent* ions only (Na+, Cl-,
acetate, H2PO4-). The explicit divalent ions screen one another explicitly and
must not be counted in I again; the Cl- or Na+ that comes with a divalent salt
is monovalent and does belong in I.

**Cation c, foreign atom j** (a protein atom, or an atom of a sulfate), at
distance r from the centre, theta_k from the axis of site k:

    WCA(r; R_c + R_j)                                 ion core; R_c + R_O = d
    per W site:  WCA(|x_j - p_k|; R_w + R_j)          the water is in the way
    per L site:  -eps_in B((r - d)/w_r) B_ang(theta_k)  inner-sphere contact
    -(n_W / n) eps_out B((r - d_out)/w_out)           outer-sphere contact
    one-body:    +dG_des per L site                   the water was given up

B_ang is 1 inside an angle theta_flat, falls smoothly to 0 at theta_max. The
flat part is the shell's angular slack: zero for Mg2+, fitted for Ca2+.

So a ligand oxygen can reach the metal only through a site in state L, where it
gains eps_in after dG_des has been paid; otherwise the site's water holds it
out at 0.34-0.49 nm, where it sees the outer-sphere shell (scaled by the
fraction of the shell that is still water) and the screened Coulomb attraction.

What one site may hold is set by the width of its well, and the loader enforces
it. A Mg2+ well is narrower than half the O...O distance of a carboxylate: one
oxygen per site, so Mg2+ is monodentate by construction, as in every survey.
A Ca2+ well is wider than that but narrower than an oxygen radius: one site can
take both oxygens of the *same* carboxylate (each 27.8 deg off the site axis),
never oxygens of two groups. That is the bidentate mode, and it costs one
water, not two, which is how crystallographers count it (Zheng et al. 2014:
a bidentate carboxylate "could be considered as occupying one coordination
site"). A carboxylate can also span two neighbouring Ca2+ sites, at the cost of
two waters; with the measured single-contact affinity that route alone gives
at most 12% bidentate, against 20-35% observed.

**Sulfate oxygen a, donor nitrogen j:**

    -eps_hb B((|x_j - o_a| - d_hb)/w_hb) P_j,   d_hb = 0.29 nm

P_j confines the bond to within 35 deg of the donor's plane for guanidinium and
imidazolium (their N-H bonds lie in the plane) and is 1 for ammonium. Two
oxygens 0.244 nm apart fit the two NH2 nitrogens 0.230 nm apart: the arginine
fork, worth 2 eps_hb. A sulfate stacked on the guanidinium face gets nothing.

**Ion with ion.** A sulfate's atoms are foreign atoms to a cation, with the same
terms as above, so a sulfate oxygen can enter a cation site (CaSO4 and MgSO4
ion pairs). Cation-cation and sulfate-sulfate: Coulomb and cores only.

**Excluded bodies.** z-parallel cylinders stand for the molecular cores; an ion
centre is repelled by WCA in its distance from the cylinder surface.

### Statistical mechanics of one ion and one or two ligands

Take one cation at position r with orientation Omega and site states
s = (s_1 .. s_n), next to fixed ligand atoms. The energy is linear in the site
indicators,

    U(r, Omega, s) = U_W(r, Omega) + sum_{k in L} D_k(r, Omega),

where U_W has every site hydrated and D_k = dG_des - [water repulsion at k]
- eps_in [inner weight at k] + (eps_out / n) [outer shell] is the cost of
opening site k alone. (That linearity is tested on random configurations.) The
sum over the 2^n states therefore factorizes:

    sum_s exp(-U) = exp(-U_W) prod_k (1 + exp(-D_k)).

A free ion has D_k = dG_des for every k: z_free = (1 + exp(-dG_des))^n, and each
site is open with probability 1 / (1 + exp(dG_des)), about 1e-7 for Mg2+.

With one ligand atom in reach of one site, only that site's factor differs from
the free ion's. The states are: unbound; outer sphere (site W, atom in the
second shell); inner sphere (site L, atom in the well). The association
constants are the configuration integrals of the two bound classes relative to
the free ion,

    K_outer = N_A/z_free int d3r < exp(-U_W) prod_k (1 + [k holds no atom] exp(-D_k)) >_Omega [contact]
    K_inner = N_A/z_free int d3r < exp(-U_W) [prod_k (1 + exp(-D_k)) - prod_k (1 + [k holds no atom] exp(-D_k))] >_Omega

"k holds an atom" and "contact" are the geometric definitions of the next
section. In the inner-sphere well, to leading order,

    K_inner ~ n N_A v_in exp(-(dG_des - eps_in) - u_C(d)),

with v_in the thermal volume of the well in position and orientation: **only
the difference eps_in - dG_des is fixed by a binding constant.** The split
between the two is bookkeeping. dG_des is set from the water-exchange barrier
(16 kT for Mg2+, about 6 kT for Ca2+) and matters for kinetics and for nothing
at equilibrium.

With two ligand atoms (the two oxygens of a carboxylate, or one oxygen from
each of two molecules) two site factors change, and expanding the product gives
the four inner/outer combinations. Keeping the terms in which the open sites
hold exactly m atoms between them gives the m-dentate weight; m = 2 on one
carboxylate is bidentate binding, m = 1 + 1 on two molecules is an inner-inner
bridge. Nothing new is fitted for these: they follow from the single-contact
terms and the rigid geometry. That is the point of carrying the geometry, and
also where a rigid polyhedron costs something real (see *Known limits*).

`calibrate.py` evaluates these integrals without a Markov chain: the state sum
exactly as above (in logarithms, because an atom sitting in a hydrated site
makes U_W astronomically large and D_k correspondingly negative), the integral
over position and orientation by importance sampling from a mixture of uniform
and "site placed on a ligand atom" configurations with the exact mixture
density. The integrator is checked against an independent two-dimensional
quadrature for a one-site ion, and the Markov chain is checked against the
integrator.

### Contacts and bridges

A **contact** joins an ion and a partner atom and is purely geometric:

| | inner | outer |
|---|---|---|
| cation, carboxylate O | a site in state L has the atom inside its well | not inner, atom within d_out + w_out of the centre (0.50 nm Mg, 0.54 nm Ca) |
| sulfate, donor N | an oxygen within the hydrogen-bond well of the donor ("direct") | not direct, N within 0.60 nm of S |

A **bridge** is an ion in contact with atoms of two different molecules. An ion
on a single carboxylate is not a bridge, nor is one held by two groups of the
same molecule. A bridge is recorded once per pair of molecules with: the class
on each side (so inner-inner, inner-outer and outer-outer are counted
separately), the closest atom of that class on each side and its distance, the
number of inner contacts per side (denticity), the number of distinct protein
groups the ion holds in all, the angle atom-ion-atom, the span between the two
atoms, and the ion's position along the molecule axis (z). `BridgeTracker`
follows each (ion, molecule pair) through a run and returns lifetimes.

**Lifetimes are in Monte Carlo sweeps and are not times.** The dock/undock
moves below jump an ion straight between outer- and inner-sphere states. They
exist precisely to remove the water-exchange barrier from the sampling, so a
magnesium bridge and a calcium bridge of equal free energy have equal lifetimes
here, while in reality the magnesium one would be three to four orders of
magnitude slower to form and break. Lifetimes compare the stability of bridges
within one run; nothing more.

### Sampler

Grand-canonical Metropolis Monte Carlo of the ions around fixed atoms, in an
orthorhombic box with each axis periodic or walled. Stationary distribution:

    P(N, {r, q, s}) ~ prod_species zeta^N / N!  exp(-U)  d^N r d^N q / (8 pi^2)^N

**Activity convention.** `bath[species]` is the concentration c of the *free*
ion in a reservoir that contains only the implicit monovalent electrolyte, and
in which the free ion's energy is the zero of U. Its number density there is
c N_A = zeta z_free, so zeta = c N_A / z_free. This is an ideal-dilute activity
with the monovalent Debye background as reference state. It is not the
analytical (total) concentration once ions are bound or paired: at 20 mM each
of Ca2+ and sulfate about a quarter of each is CaSO4(aq). No activity
coefficient for divalent-divalent interactions in the bath is applied, so the
free-ion density inside a box of interacting ions comes out a few per cent
below c N_A at 20-50 mM.

Moves, each satisfying detailed balance on its own:

- **translate**, **rotate**: symmetric proposals, each drawn at the full step or
  a fifth of it. An ion in a contact well a few hundredths of a nanometre wide
  and one in the bulk need different step sizes; with a single size the
  bidentate calcium state was entered and left so rarely that its population
  was wrong by a factor of two in runs of five million moves.
- **flip**: one site W <-> L, symmetric; half the time together with a
  translate-and-rotate displacement, so that an ion holding one ligand can
  swing a second site onto another ligand as that site opens.
- **dock / undock**: the pair that connects outer and inner sphere. A hydrated
  site cannot admit a ligand (its water is in the way) and an open empty site
  costs dG_des, so with local moves alone the inner sphere of Mg2+ would be
  reached once in exp(16) attempts.
  *Dock* (site k in state W): choose one of the C dockable atoms within R_nb of
  the centre; draw a Haar-uniform orientation and a point uniform in a ball of
  radius rho around the atom; put site k there; set it to L.
  *Undock* (site k in state L with M >= 1 dockable atoms within rho of it):
  choose one of them; draw a Haar-uniform orientation and a centre uniform in
  the ball of radius R_nb about that atom; set the site to W.
  A docked configuration y can be proposed from x through any atom that is both
  within R_nb of the old centre and within rho of the new site position; call
  their number m. Then q(x->y) = (1/n)(m/C)(1/8pi^2)(1/v_rho) and
  q(y->x) = (1/n)(m/M)(1/8pi^2)(1/V_nb), m cancels, and

      A(dock)   = min(1, exp(-dU) C v_rho / (M V_nb)),
      A(undock) = min(1, exp(-dU) M V_nb / (C v_rho)),

  with C counted around the undocked centre and M around the docked site.
- **insert / delete**: a new ion uniform in the box, Haar orientation, all sites
  W, accepted with min(1, zeta V / (N+1) exp(-dU)); deletion of a uniformly
  chosen ion of the species, refused unless all its sites are W, accepted with
  min(1, N / (zeta V) exp(-dU)). Restricting both directions to the
  fully hydrated state keeps the pair reversible; the other states are reached
  by flips, and z_free in zeta accounts for them.

All randomness comes from the numpy `Generator` given to the sampler; a run is
reproduced by its seed, and splitting a run into chunks does not change it.

The screened Coulomb term is cut and shifted at 4/kappa (clipped to 1-5 nm and
to what fits in half the shortest periodic edge). `interacting=False` removes
every ion-ion term, which gives the exact infinite-dilution limit used in the
isotherm test.

## What is calibrated, what is provisional

`params.toml` carries every number with units, source and one of three
statuses. The split, in words:

**Taken from the literature** (geometry and targets): bond lengths (Mg-O 0.207,
Ca-O 0.238, S-O 0.1495 nm), second-shell distances, coordination geometry, the
hydrogen-bond distance, the water-exchange barrier of Mg2+, and the calibration
targets for the four cation pairs: total association constants from the NIST
critical compilation, the Raman contact-pair constants of Mendes de Oliveira et
al. (2020) for acetate, the contact share of MgSO4, and the bidentate share of
Ca-carboxylate contacts in protein structures.

**Fitted here** (12 numbers): eps_in and eps_out for each cation with
carboxylate and with sulfate oxygen; eps_hb for sulfate with each of the three
donors; and the angular slack of Ca2+. All are fitted at I = 0 to the model's
own two-body integral. `python -m modcoll.ions.calibrate` prints the table;
`--write` refits.

| Pair (I = 0) | Target | Model | Fitted |
|---|---|---|---|
| Ca2+ + acetate | K_total 15.1 /M (log K 1.18) | 15.1 | eps_out 1.73 kT |
| | K_inner 3.6 /M | 3.60 | eps_in 8.20 kT (dG_des 6.0) |
| | bidentate share of inner 0.27 | 0.27 | angular slack 23.4 deg |
| Mg2+ + acetate | K_total 18.2 /M (log K 1.26) | 18.2 | eps_out 2.01 kT |
| | K_inner 3.9 /M | 3.90 | eps_in 20.7 kT (dG_des 16.0) |
| Ca2+ + SO4 2- | K_total 229 /M (log K 2.36) | 229 | eps_out 1.24 kT |
| | inner share 0.29 (provisional) | 0.29 | eps_in 6.13 kT |
| Mg2+ + SO4 2- | K_total 182 /M (log K 2.26) | 182 | eps_out 1.21 kT |
| | inner share 0.10 | 0.10 | eps_in 16.9 kT |
| SO4 2- + guanidinium | K_total 15 /M (estimate) | 15.0 | eps_hb 2.96 kT |
| SO4 2- + ammonium | K_total 8 /M (estimate) | 8.0 | eps_hb 0.34 kT |
| SO4 2- + imidazolium | K_total 10 /M (estimate) | 10.0 | eps_hb 3.43 kT |

Every pair has as many fitted numbers as targets, so every target is met to
within the sampling error of the integral (about 1% on K_total, 2-3% on
K_inner, 5% on the bidentate share). That is a statement about the fit having
been done, not evidence for the model.

**What the fit says about Ca2+ against Mg2+.** Inner-sphere pairs are 24% of
Ca-acetate pairs and 21% of Mg-acetate pairs: both ions are mostly outer-sphere
on a single carboxylate, and they do not differ in affinity. This is what the
measurements say (equal total constants, equal Raman contact constants) and
the model reproduces it because it was told to. It is **not** the picture of
calcium binding directly and magnesium through its water shell. In the model
the tight hexahydrate of Mg2+ (dG_des 16 kT against 6) is offset by a contact
energy larger by nearly the same amount (eps_in 20.7 against 8.2; net 4.7
against 2.2 kT). If Mg2+ gained only as much on contact as Ca2+ does, its
inner-sphere share with acetate would be about 3e-6: the hydration-shell
argument, taken alone, predicts an ion that is never inner-sphere, and the
Raman data say otherwise. The magnesium target is disputed (another
spectroscopic reading gives about 6% contact pairs instead of 21%); the low end
is in `params.toml` as the range of that target and belongs in the sensitivity
suite. What separates the two ions here is:

- geometry: Mg2+ is rigid, strictly six-coordinate and monodentate; Ca2+ has
  seven loosely directed sites and holds a carboxylate bidentate 27% of the
  time;
- kinetics, which the Monte Carlo does not have;
- reach: the outer-sphere shell is at 0.42 nm for Mg2+ and 0.46 nm for Ca2+.

**The split of eps_in and dG_des** is not quite pure bookkeeping for Ca2+. A
bidentate carboxylate on one site gains eps_in twice and pays dG_des once, so
at fixed single-contact affinity a larger dG_des means more bidentate binding.
The angular slack was fitted with dG_des = 6 kT and must be refitted if that
number is changed (4-11 kT is the plausible range).

**Checks the model was not fitted to.**

| Check | Measured | Model |
|---|---|---|
| Ca2+ + acetate, log K at I = 0.1 | 0.55 (0.48-0.62) | 0.67 |
| Mg2+ + acetate, log K at I = 0.1 | 0.51 (0.46-0.56) | 0.74 |
| CaSO4(aq), log K at I = 0.1 | 1.53 (1.48-1.58) | 1.36 |
| MgSO4(aq), log K at I = 0.1 | 1.48 (1.46-1.50) | 1.26 |
| Ca2+ + succinate(2-), log K at I = 0 | 2.00 (1.9-2.1) | 2.27 extended, 2.62 gauche |
| Mg2+ + succinate(2-), log K at I = 0 | 2.18 (2.1-2.3) | 2.36 extended, 2.84 gauche |
| ion-strength slope at I -> 0 | Debye-Hueckel limiting law | reproduced to 2% |

Two honest readings. First, screening: going from I = 0 to 0.1 M the model
weakens the 2:1 pairs by 0.5 log units where the measurements give 0.6-0.75,
and the 2:2 pairs by 1.0 where they give 0.8. Plain Debye-Hueckel screening is
within a factor 1.3-1.7 in K at the ionic strength the simulations will run at,
not better. Second, two carboxylates on one cation: the model over-binds
succinate by a factor of about 2 (extended conformer) to 4 (gauche). The
conformers are rigid and the real ion is a mixture that also loses internal
entropy on chelation, so some excess is expected, but the direction matters:
**a cation held by two carboxylates is what a bridge is, and this model is more
likely to overstate bridging than to understate it.**

**Provisional** (no measurement behind them; first candidates for sensitivity
runs): the three sulfate-donor targets (no association constant of sulfate with
guanidinium, alkylammonium or imidazolium has been measured; these are the
weakest numbers in the file), the inner-sphere share of CaSO4, dG_des of Ca2+
(its water exchange has never been measured directly), all well widths and the
angular half-widths, the water and atom contact radii, the outer-shell
half-width, the outer contact distance of sulfate, the charge layout of
sulfate, and the cylinder wall radii.



## S3 results (`results/modcoll/ions/`)

Produced by `python -m modcoll.ions.validate_s3` (seed 20261112); the numbers
behind every figure are in the two JSON files next to them.

**Single pairs.** `s3_binding_table.md` and `s3_binding_constants.png` give the
table above and the checks. Geometry from Monte Carlo of one ion and one ligand
at I = 0.1 M (`s3_distances.png`):

| | Model | Reference |
|---|---|---|
| Ca-O, inner sphere | 0.243 +- 0.012 nm | 0.236-0.246 nm |
| Mg-O, inner sphere | 0.207 +- 0.005 nm | 0.207 nm |
| Ca to outer-sphere O | 0.463 nm | 0.458 nm (second shell) |
| Mg to outer-sphere O | 0.426 nm | 0.41-0.43 nm |
| rms angle off the site axis, Ca / Mg | 19 deg / 7 deg | working tolerances 15-20 / 5-10 deg |
| bidentate bite O-Ca-O | 52.5 deg | 52-54 deg in crystals |
| S...N in a direct sulfate-guanidinium pair | 0.379 +- 0.041 nm | S to C-zeta 0.40 nm computed for the fork |

The distances are close to what was put in; the angular spreads and the bite
angle were not put in. Direct sulfate-guanidinium pairs hold two nitrogens
(the fork) more often than one, 58% of the time.

`s3_inner_outer.png`: inner-sphere shares from the quadrature at I = 0 with the
Monte Carlo shares at I = 0.1 M marked (22% and 20% inner for Ca2+ and Mg2+).
The bidentate share in that particular Monte Carlo run rests on a few hundred
samples and is not to be read off the table; the test suite compares it with
the quadrature in a longer run.

**Two facing rows** (`s3_bridges_vs_gap.png`, `s3_bridge_kinds.png`). Eight
groups per row, 0.86 nm apart, on the facing sides of two excluded cylinders;
periodic along the rows; I = 0.1 M; bath 1, 5, 20, 50 mM. Carboxylate rows are
protonated at random for pH 4.2 and pKa 4.3 (44% charged), four independent
patterns per point; guanidinium rows are fully charged. The gap is the distance
between the two layers of partner atoms. Counts are bridges per facing pair of
groups.

- *Cations bridge mostly through water.* At 50 mM there are about 0.2 bridges
  per facing pair from 0.35 to 0.65 nm and 0.11-0.13 out to 0.9 nm, falling to
  zero by 1.0 nm (Ca2+) and 0.95 nm (Mg2+). Nearly all are outer-outer. At 5 mM
  the count is 0.05-0.10. For scale, only 19% of facing pairs have both groups
  charged at this pH, so at 50 mM there are about as many bridges as doubly
  charged facing pairs; a quarter to a third of the bridging ions touch three
  or more groups.
- *Inner-outer bridges* appear in a window: up to 0.08 per pair for Ca2+ at
  0.60-0.65 nm, up to 0.04 for Mg2+ at 0.55 nm.
- *Inner-inner bridges* were seen for Ca2+ only, at 0.35 and 0.45 nm, as the
  doubly bidentate state: 0.03 per pair on average with individual runs between
  0 and 0.12. This is the unconverged state described under *Known limits*;
  the number is not a result.
- *Ca2+ and Mg2+ are alike* within the pattern-to-pattern scatter (the bands in
  the figure, about +-25%). The model does not reproduce "calcium bridges
  directly and short, magnesium through its water shell and long": with the
  measured single-contact affinities both ions bridge through their water
  shells, Mg2+ slightly more and over a slightly shorter range. Calcium's
  distinct behaviour is confined to the short-gap doubly chelated state, which
  this work cannot quantify.
- *Sulfate between guanidinium rows* bridges more: 0.3-0.75 per facing pair at
  50 mM, with peaks at 0.45-0.55 nm and at 0.75 nm. At 0.75 nm over half of the
  bridges are held by two hydrogen bonds on each side and nearly all involve
  exactly two guanidinium groups; at 0.45-0.55 nm most involve three or four
  groups, neighbours along the row included. Turning one row's guanidinium
  planes by 90 deg changes little at 0.75 nm but at 0.40-0.50 nm removes most
  of the direct-direct bridges (0.01-0.11 per pair against 0.37-0.64) while
  the total falls by less than half: bridging is selective in the gap and, at
  short gaps, in the mutual orientation of the two arginines. All of this scales with the estimated sulfate-donor
  constants; if those are off by a factor of three so is the bridging.
- *Lifetimes* of bridges at 50 mM are 130-550 sweeps for the cations and
  75-930 for sulfate (longest at the 0.75 nm peak). See the caveat on what a
  sweep is.

The rows are a demonstration of the machinery, not a model of two collagen
molecules: the spacing, the facing registry and the cylinder radius are
arbitrary, and both rows carry the same sign of charge with no opposing
charges nearby.

## Known limits

- **No polarization and no charge transfer.** Charges are fixed. The contact
  energies absorb whatever polarization does on average in a single ion pair
  and cannot follow it into other environments. Published fixed-charge models
  of these ions over-bind acetate by 10-30 kJ/mol (4-12 kT; Mendes de Oliveira
  et al. 2020) unless calibrated against solution data; this model is
  calibrated, on single pairs only.
- **Implicit water.** No hydrogen-bond network, no dielectric saturation near
  the ions, no hydration forces between surfaces, no water-mediated correlation
  beyond the one outer-sphere shell. The continuum Coulomb term is used down to
  contact, where it is not valid, and the contact terms are whatever makes the
  association constants right in spite of that.
- **The split of eps_in and dG_des is not determined** by any equilibrium
  quantity. dG_des is tied to water-exchange kinetics; the Monte Carlo has no
  kinetics. The measured difference between Mg2+ and Ca2+ in how fast a site
  switches (microseconds against about a nanosecond) is not represented by
  anything a Monte Carlo sweep count can be compared with.
- **Rigid polyhedra.** A real coordination shell rearranges; here the second
  ligand of an ion must find a second site of a rigid frame, and pays the full
  orientational entropy of doing so. The angular slack given to Ca2+ is the
  only relief. Coordination number is a parameter (6, 7, 8 for Ca2+), not an
  outcome, and the longer Ca-O bonds of bidentate carboxylate (0.25-0.28 nm)
  are represented only by the width of the radial well: a bidentate oxygen
  sits at 0.238 nm here.
- **Doubly chelated calcium is not converged.** Where two carboxylates of
  different molecules face each other with their oxygen layers closer than
  about 0.46 nm, a Ca2+ can be held bidentate by both: four oxygens through two
  sites. In this model that state lies of order 10-15 kT below the outer-sphere
  states and is very narrow. Runs of eight million moves on one facing pair at
  0.42 nm, differing only in seed, spent 3%, 10%, 34% and 38% of the time in
  it; the quadrature there gives a multidentate-to-monodentate ratio of 1.9
  with four million samples and 3.6 with forty million. At 0.48 nm, where the
  state is nearly out of reach, sampler and quadrature agree (0.70 against
  0.68). Inner-inner calcium counts at gaps below
  0.46 nm are therefore unconverged in everything reported here, and the depth
  of the state is an extrapolation: it assumes four contacts are worth four
  times one, which the succinate check suggests is too generous. Mg2+ cannot
  form it (no bidentate binding) and its singly bonded inner-inner bridges are
  rarer still and are essentially never visited: with a 16 kT site-opening cost
  only the dock/undock pair opens Mg2+ sites, and it places one site at a time.
  The pair-level work will need a move that docks a second site while the
  first stays put, or tempering in the contact energy.
- **Quenched protonation.** Each carboxylate is protonated or not, drawn once.
  A cation does not displace a proton and a bound cation does not shift a
  neighbour's pKa. Near pH 4.2, where the carboxyl pKa lies, real binding is a
  competition with protons; the model under-counts cations on sites it froze
  as protonated and over-counts none.
- **Monovalent ions are a Debye length.** No Na+ competition for carboxylates
  (log K about -0.1), no acetate or chloride pairing with the cations, no
  H2PO4-. Debye-Hueckel screening is used at ionic strengths (0.1 M and above)
  where it is no longer quantitative; see the I = 0.1 checks.
- **The bath is ideal.** See the activity convention. Gypsum does not
  precipitate in a simulation: equimolar Ca2+ and sulfate above about 20 mM is
  supersaturated in reality.
- **Protein atoms are fixed.** No side-chain flexibility; a carboxylate cannot
  turn towards an ion. The excluded body is a cylinder.
- **Sulfate with arginine, lysine and histidine rests on estimates.** No
  association constant has been measured for any of the three. These are the
  weakest numbers in the parameter file.

## Running it

```
python3 -m venv --without-pip .venv        # or any environment with pip
.venv/bin/python -m pip install -r modcoll/ions/requirements.txt
.venv/bin/python -m pytest tests/modcoll -q          # about fifteen minutes
.venv/bin/python -m modcoll.ions.calibrate           # model against targets
.venv/bin/python -m modcoll.ions.calibrate --write   # refit after editing params.toml
.venv/bin/python -m modcoll.ions.validate_s3         # S3 tables and figures
```

numba is optional: without it the same kernels run interpreted, about a hundred
times slower, and the statistical tests become impractical.

Minimal use:

```python
import numpy as np
from modcoll.ions import (IonSampler, BridgeAnalyzer, LigandSet, carboxylate, load_params)

params = load_params()                      # ca_coordination=6 or 8 for the variants
box = [6.0, 6.0, 6.0]
ligands = LigandSet.concat([
    carboxylate([2.7, 3.0, 3.0], [+1, 0, 0], mol=0),
    carboxylate([3.4, 3.0, 3.0], [-1, 0, 0], mol=1),
])
sampler = IonSampler(params, ligands, box, periodic=[True] * 3, ionic_strength=0.1,
                     rng=np.random.default_rng(1), bath={"Ca": 0.02})
analyzer = BridgeAnalyzer(params, ligands, box, [True] * 3)
sampler.sweep(5000)
for bridge in analyzer.bridges(sampler.snapshot()):
    print(bridge.species, bridge.kind, bridge.span, bridge.z)
```

## References

Association constants and ion pairing
- Martell, Smith & Motekaitis. NIST Standard Reference Database 46: Critically Selected Stability Constants of Metal Complexes, version 8.0. doi:10.18434/M32154
- Mendes de Oliveira, Zukowski, Palivec, Henin, Martinez-Seara, Ben-Amotz, Jungwirth & Duboue-Dijon. Binding of divalent cations to acetate: molecular simulations guided by Raman spectroscopy. Phys. Chem. Chem. Phys. 22 (2020) 24014. doi:10.1039/d0cp02987d
- Vazquez de Vasquez, Wellen Rudd, Baer, Beasley & Allen. J. Phys. Chem. B 125 (2021) 11308. doi:10.1021/acs.jpcb.1c06108
- Stumpff & McGuigan. Anal. Biochem. 459 (2014) 46. doi:10.1016/j.ab.2014.05.001
- Hefter. Pure Appl. Chem. 78 (2006) 1571. doi:10.1351/pac200678081571
- Buchner, Chen & Hefter. J. Phys. Chem. B 108 (2004) 2365. doi:10.1021/jp034870p
- Riopedre-Fernandez, Kostal, Martinek, Martinez-Seara & Biriukov. J. Chem. Inf. Model. 64 (2024) 7122. doi:10.1021/acs.jcim.4c00981
- Wernersson, Heyda, Kubickova, Krizek, Coufal & Jungwirth. J. Phys. Chem. B 114 (2010) 11934. doi:10.1021/jp1054342
- Fromm, Hileman, Caldwell, Weiler & Linhardt. Arch. Biochem. Biophys. 323 (1995) 279. doi:10.1006/abbi.1995.9963
- Herman, Valiya Parambathu, Asthagiri & Lenhoff. J. Phys. Chem. Lett. 14 (2023) 7020. doi:10.1021/acs.jpclett.3c01566
- Hunger, Neueder, Buchner & Apelblat. J. Phys. Chem. B 117 (2013) 615. doi:10.1021/jp311425v
- PHREEQC version 3 database files (phreeqc.dat, minteq.v4.dat), U.S. Geological Survey.

Coordination geometry and water exchange
- Harding. Small revisions to predicted distances around metal sites in proteins. Acta Cryst. D62 (2006) 678. doi:10.1107/S0907444906014594
- Bazayeva, Andreini & Rosato. A database overview of metal-coordination distances in metalloproteins. Acta Cryst. D80 (2024) 362. doi:10.1107/S2059798324003152
- Katz, Glusker, Beebe & Bock. Calcium ion coordination: a comparison with that of beryllium, magnesium, and zinc. J. Am. Chem. Soc. 118 (1996) 5752. doi:10.1021/ja953943i
- Zheng, Chordia, Cooper, Chruszcz, Muller, Sheldrick & Minor. Nat. Protoc. 9 (2014) 156. doi:10.1038/nprot.2013.172
- Zheng, Cooper, Porebski, Shabalin, Handing & Minor. Acta Cryst. D73 (2017) 223. doi:10.1107/S2059798317001061
- Gagne & Hawthorne. Acta Cryst. B72 (2016) 602. doi:10.1107/S2052520616008507
- Persson. Structures of hydrated metal ions in solid state and aqueous solution. Liquids 2 (2022) 210. doi:10.3390/liquids2030014
- Jalilehvand, Spangberg, Lindqvist-Reis, Hermansson, Persson & Sandstrom. J. Am. Chem. Soc. 123 (2001) 431. doi:10.1021/ja001533a
- Schwenk & Rode. Pure Appl. Chem. 76 (2004) 37. doi:10.1351/pac200476010037
- Vchirawongkwin, Rode & Persson. J. Phys. Chem. B 111 (2007) 4150. doi:10.1021/jp0702402
- Copley & Barton. J. Mol. Biol. 242 (1994) 321. doi:10.1006/jmbi.1994.1583
- Bleuzen, Pittet, Helm & Merbach. Water exchange on magnesium(II) in aqueous solution. Magn. Reson. Chem. 35 (1997) 765.
- Neely & Connick. J. Am. Chem. Soc. 92 (1970) 3476. doi:10.1021/ja00714a048
- Diebler, Eigen, Ilgenfritz, Maass & Winkler. Pure Appl. Chem. 20 (1969) 93. doi:10.1351/pac196920010093
- Remsing & Klein. J. Phys. Chem. A 123 (2019) 1077. doi:10.1021/acs.jpca.8b09667

Multisite ion models
- Aqvist & Warshel. J. Am. Chem. Soc. 112 (1990) 2860. doi:10.1021/ja00164a003
- Oelschlaeger, Klahn, Beard, Wilson & Warshel. J. Mol. Biol. 366 (2007) 687. doi:10.1016/j.jmb.2006.10.095
- Saxena & Sept. Multisite ion models that improve coordination and free energy calculations in molecular dynamics simulations. J. Chem. Theory Comput. 9 (2013) 3538. doi:10.1021/ct400177g
- Saxena & Garcia. J. Phys. Chem. B 119 (2015) 219. doi:10.1021/jp507008x
- Duarte, Bauer, Barrozo, Amrein, Purg, Aqvist & Kamerlin. Force field independent metal parameters using a nonbonded dummy model. J. Phys. Chem. B 118 (2014) 4351. doi:10.1021/jp501737x
- Li & Merz. Metal ion modeling using classical mechanics. Chem. Rev. 117 (2017) 1564. doi:10.1021/acs.chemrev.6b00440
- Zeron, Abascal & Vega. J. Chem. Phys. 151 (2019) 134504. doi:10.1063/1.5121392
- Nguyen, Hori & Thirumalai. Proc. Natl. Acad. Sci. USA 116 (2019) 21022. doi:10.1073/pnas.1911632116

Methods
- Weeks, Chandler & Andersen. J. Chem. Phys. 54 (1971) 5237.
- Thurlkill, Grimsley, Scholtz & Pace. Protein Sci. 15 (2006) 1214. doi:10.1110/ps.051840806
- Platzer, Okon & McIntosh. J. Biomol. NMR 60 (2014) 109. doi:10.1007/s10858-014-9862-y
