"""The eps_h fits of aamd.compare on synthetic rows with known parameters."""
import numpy as np

from aamd.compare import fit_eps_h, fit_eps_h_offset


def rows(eps, c, H=(65.0, -3.0, 56.0, -2.5), el=(-0.9, 0.2, 1.3, 1.3), err=(0.6, 1.0, 0.5, 1.5)):
    return [{"dG_contact_kT": e - eps * h + c, "E_mean_el_kT": e, "H_overlap": h, "err_block_kT": s}
            for h, e, s in zip(H, el, err)]


def test_offset_fit_recovers_slope_and_constant():
    f = fit_eps_h_offset(rows(0.05, -7.0))
    assert np.isclose(f["eps_h"], 0.05) and np.isclose(f["c_kT"], -7.0)
    assert f["chi2"] < 1e-9 and f["dof"] == 2


def test_register_independent_attraction_is_not_read_as_eps_h():
    # equal depth at both staggers: the slope-only fit invents an eps_h, the offset fit does not
    r = rows(0.0, -8.0)
    assert fit_eps_h(r)["eps_h_fit"] > 0.05
    assert abs(fit_eps_h_offset(r)["eps_h"]) < 1e-9


def test_offset_fit_needs_three_rows():
    assert fit_eps_h_offset(rows(0.05, -7.0)[:2]) is None
