"""Site model: composition, geometry and agreement with the existing tools.

Oracles: residue counts published in the README, the telopeptide sequences
of mature human type I collagen, the 7/2 helix parameters the README quotes
from PDB 1K6F, and the per-residue surface positions computed by the existing
tools/azimuthal_profile.py.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from modcoll import sites
from modcoll.sites import SiteType, native_molecule

ROOT = Path(__file__).resolve().parents[2]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def azimuthal_tool():
    sites.ensure_structure()
    return load_tool("azimuthal_profile")


def test_tool_telopeptide_model_reproduces_readme_counts():
    # README: "94 D / 143 E / 18 H / 108 K / 163 R per trimer"
    m = native_molecule("tool")
    got = [m.count(t) for t in (SiteType.ASP, SiteType.GLU, SiteType.HIS,
                                SiteType.LYS, SiteType.ARG)]
    assert got == [94, 143, 18, 108, 163]


def test_mature_telopeptides_are_the_published_sequences():
    # N- and C-telopeptides of mature human alpha1(I) and alpha2(I)
    segs = sites.chain_segments(ROOT / "data", "mature")
    assert segs[0]["ntel"] == "QLSYGYDEKSTGGISVP"
    assert segs[0]["ctel"] == "SAGFDFSFLPQPPQEKAHDGGRYYRA"
    assert segs[1]["ntel"] == "QYDGKGVGLGP"
    assert segs[1]["ctel"] == "GGGYDFGYDGDFYRA"
    for seg in segs:
        assert len(seg["helix"]) == 1014
        assert seg["helix"][::3] == "G" * 338      # uninterrupted Gly-X-Y


def test_mature_counts_match_a_direct_count_of_the_sequences():
    segs = sites.chain_segments(ROOT / "data", "mature")
    full = [s["ntel"] + s["helix"] + s["ctel"] for s in segs]
    m = native_molecule("mature")
    for letter, st in (("D", SiteType.ASP), ("E", SiteType.GLU), ("H", SiteType.HIS),
                       ("K", SiteType.LYS), ("R", SiteType.ARG), ("N", SiteType.ASN),
                       ("Y", SiteType.TYR)):
        assert m.count(st) == 2 * full[0].count(letter) + full[1].count(letter)
    # the three N-terminal Gln are pyroglutamate: not deamidatable
    assert m.count(SiteType.GLN) == 2 * full[0].count("Q") + full[1].count("Q") - 3


def test_helix_positions_match_existing_tool(azimuthal_tool):
    """Same side-chain-centroid reduction as load_residue_surface; the only
    difference is the documented axial origin (helix N-end at s = 0)."""
    surf = azimuthal_tool.load_residue_surface()
    m = native_molecule("none")
    helix = m.is_residue
    zmin_nm = sites.read_helix_geometry(sites.ensure_structure())["zmin"] / 10.0
    assert helix.sum() == len(surf) == 3042
    np.testing.assert_allclose(m.s[helix], surf[:, 0] - zmin_nm, atol=1e-9)
    np.testing.assert_allclose(m.phi[helix], surf[:, 1], atol=1e-12)
    np.testing.assert_allclose(m.hydro[helix], surf[:, 3], atol=1e-12)
    # tool titratable index D,E,H,K,R = 0..4 <-> SiteType 1..5; -1 = none
    tool_t = surf[:, 4].astype(int)
    mine = np.where(np.isin(m.site_type[helix], [1, 2, 3, 4, 5]), m.site_type[helix] - 1, -1)
    np.testing.assert_array_equal(mine, tool_t)
    assert m.s[helix].min() >= 0.0 and m.s[helix].max() <= m.helix_length_nm


def test_tool_telopeptides_match_existing_tool(azimuthal_tool):
    m = native_molecule("tool")
    tool = np.array(azimuthal_tool.telopeptides(m.helix_length_nm))
    tel = m.smeared & m.is_residue
    assert tel.sum() == len(tool) == 123
    np.testing.assert_allclose(m.s[tel], tool[:, 0], atol=1e-9)
    np.testing.assert_allclose(m.hydro[tel], tool[:, 2], atol=1e-12)
    mine = np.where(np.isin(m.site_type[tel], [1, 2, 3, 4, 5]), m.site_type[tel] - 1, -1)
    np.testing.assert_array_equal(mine, tool[:, 3].astype(int))


def test_seven_two_helix_rise_and_twist():
    # README: rise 2.858 A per residue, +51.37 degrees per triplet (PDB 1K6F)
    m = native_molecule("none")
    gly = (m.chain == 0) & (m.res_index % 3 == 0) & m.is_residue   # the Gly of each triplet
    assert np.all(m.aa[gly] == "G")
    order = np.argsort(m.res_index[gly])
    idx, s, phi = m.res_index[gly][order], m.s[gly][order], m.phi[gly][order]
    assert np.all(np.diff(idx) == 3)
    np.testing.assert_allclose(np.diff(s), 3 * 0.2858, atol=2e-4)
    twist = np.degrees(np.mod(np.diff(phi), 2 * np.pi))
    # single steps carry the 0.001 A rounding of the PDB coordinates
    np.testing.assert_allclose(twist, 51.37, atol=0.05)
    assert twist.mean() == pytest.approx(51.371, abs=0.002)


def test_d_period_is_234_residue_rises():
    m = native_molecule()
    assert m.d_period_nm == pytest.approx(234 * 0.2858, abs=0.02)   # 66.9 nm
    assert m.helix_length_nm / m.d_period_nm == pytest.approx(4.34, abs=0.01)
    assert m.rise_nm == pytest.approx(0.2858, abs=1e-4)


def test_glycine_is_in_the_core_and_charged_groups_reach_further_out():
    m = native_molecule("none")
    res = m.is_residue
    assert m.r[res & (m.aa == "G")].max() < m.r[res & (m.aa != "G")].min() + 0.02
    lys = m.site_type == int(SiteType.LYS)
    # the amine nitrogen is further from the axis than the side-chain centroid
    assert np.all(m.r_group[lys] > m.r[lys])
    other = m.site_type == int(SiteType.OTHER)
    np.testing.assert_array_equal(m.r_group[other], m.r[other])


def test_n_terminus_modes():
    auto, free, blocked = (native_molecule("mature", n) for n in ("auto", "free", "blocked"))
    assert auto.count(SiteType.NTERM) == 0          # both mature chains start with Gln
    assert free.count(SiteType.NTERM) == 3
    assert blocked.count(SiteType.NTERM) == 0
    assert free.count(SiteType.GLN) == auto.count(SiteType.GLN) + 3
    # the tool's window does not start on Gln, so "auto" leaves it free
    assert native_molecule("tool", "auto").count(SiteType.NTERM) == 3
    for m in (auto, free, blocked):
        assert m.count(SiteType.CTERM) == 3


def test_termini_sit_at_the_chain_ends():
    m = native_molecule("mature", "free")
    for chain in range(3):
        rows = m.chain == chain
        n_term = rows & (m.site_type == int(SiteType.NTERM))
        c_term = rows & (m.site_type == int(SiteType.CTERM))
        assert m.s[n_term][0] == pytest.approx(m.s[rows & m.is_residue].min())
        assert m.s[c_term][0] == pytest.approx(m.s[rows & m.is_residue].max())
        assert not m.is_residue[n_term][0] and m.hydro[c_term][0] == 0.0


def test_centered_hydrophobicity():
    m = native_molecule()
    h = m.centered_hydrophobicity()
    assert h[m.is_residue].mean() == pytest.approx(0.0, abs=1e-12)
    assert np.all(h[~m.is_residue] == 0.0)
    leu = m.is_residue & (m.aa == "L")
    arg = m.is_residue & (m.aa == "R")
    # Kyte-Doolittle: Leu 3.8, Arg -4.5; centring shifts both by the same amount
    assert h[leu][0] - h[arg][0] == pytest.approx(3.8 + 4.5)


def test_hydroxyproline_only_in_y_position():
    m = native_molecule("none")
    hyp = m.is_residue & (m.aa == "O")
    assert hyp.sum() > 300
    assert np.all(m.res_index[hyp] % 3 == 2)
    assert np.all(m.site_type[hyp] == int(SiteType.OTHER))


def test_unknown_options_are_rejected():
    with pytest.raises(ValueError):
        sites.chain_segments(ROOT / "data", "trypsin")
    with pytest.raises(ValueError):
        sites.build_native_molecule(n_terminus="maybe")
    with pytest.raises(ValueError):
        native_molecule().positions("backbone")


def test_helix_is_anchored_on_the_annotated_region():
    # UniProt: triple-helical region 179-1192 of P02452 and 91-1104 of P08123
    segs = sites.chain_segments(ROOT / "data", "mature")
    assert [s["helix_start"] + 1 for s in segs] == [179, 91]
    assert [s["helix_start"] + len(s["helix"]) for s in segs] == [1192, 1104]
    # the search rule of the existing builder over-runs into alpha2's
    # C-telopeptide (GGG...); it is right for human only after truncation
    alpha2 = sites.read_fasta(ROOT / "data" / "P08123.fasta")
    assert sites.helical_domain(alpha2) == (90, 339)


def test_anchoring_does_not_follow_glycines_of_the_telopeptide(tmp_path):
    """A chain whose N-telopeptide happens to carry Gly in phase with the
    helix (as bovine alpha2 does): the longest-Gly-run rule starts the helix
    early, the anchored helix does not move."""
    ntel, helix, ctel = "QAGAAGAA", "GPP" * 338, "GGGYA"
    seq = "MMMMM" + ntel + helix + ctel + "DDD"
    (tmp_path / "toy.fasta").write_text(">toy\n" + seq + "\n")
    assert sites.helical_domain(seq)[0] == 7            # two triplets too early
    spec = sites.ChainSpec("toy.fasta", 1, mature_first=6, mature_last=5 + 8 + 1014 + 5,
                           helix_first=14)
    seg = sites.chain_segments(tmp_path, "mature", chains=(spec,))[0]
    assert (seg["ntel"], seg["helix"], seg["ctel"]) == (ntel, helix, ctel)
    shifted = sites.ChainSpec("toy.fasta", 1, 6, 5 + 8 + 1014 + 5, helix_first=13)
    with pytest.raises(ValueError):
        sites.chain_segments(tmp_path, "mature", chains=(shifted,))


def test_pepsin_chain_ends():
    """Preferred pepsin cuts: alpha1 loses pQLSYGY and the distal 19 residues
    of its C-telopeptide, alpha2 loses two N-terminal residues."""
    segs = sites.chain_segments(ROOT / "data", "pepsin")
    assert (segs[0]["ntel"], segs[0]["ctel"]) == ("DEKSTGGISVP", "SAGFDFS")
    assert (segs[1]["ntel"], segs[1]["ctel"]) == ("DGKGVGLGP", "GGGYDFGYDGDFYRA")
    m = native_molecule("pepsin")
    # each alpha1 C-telopeptide stub lost 1 Lys, 2 Arg, 1 His, 1 Asp, 1 Glu
    intact = native_molecule("mature")
    for st, lost in ((SiteType.LYS, 2), (SiteType.ARG, 4), (SiteType.HIS, 2),
                     (SiteType.ASP, 2), (SiteType.GLU, 2)):
        assert m.count(st) == intact.count(st) - lost
    assert [m.count(t) for t in (SiteType.LYS, SiteType.ARG, SiteType.HIS, SiteType.ASP,
                                 SiteType.GLU)] == [105, 157, 16, 90, 141]
    assert m.count(SiteType.NTERM) == 3        # new alpha-amines on Asp
    helix_only = native_molecule("none")
    assert [helix_only.count(t) for t in (SiteType.LYS, SiteType.ARG, SiteType.HIS,
                                          SiteType.ASP, SiteType.GLU)] == [102, 156, 16, 82, 139]


def test_tool_window_carries_a_propeptide_lysine_the_mature_chain_lacks():
    tool, mature = native_molecule("tool"), native_molecule("mature")
    assert tool.count(SiteType.LYS) == mature.count(SiteType.LYS) + 1
    tool_seg = sites.chain_segments(ROOT / "data", "tool")[1]
    assert tool_seg["ntel"].startswith("GNFAA")          # N-propeptide residues
    assert tool_seg["ctel"].endswith("DQPRSAPSLRPKD")    # C-propeptide residues


def test_tyrosines_are_typed():
    m = native_molecule("mature")
    assert m.count(SiteType.TYR) == 13                   # 1 in the helices, 12 in telopeptides
    assert np.all(m.aa[m.site_type == int(SiteType.TYR)] == "Y")
    assert int(np.sum((m.site_type == int(SiteType.TYR)) & ~m.smeared)) == 1
