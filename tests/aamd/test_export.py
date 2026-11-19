"""Chain labels of the exported topology: 32 residues per chain, ions chain I."""
from aamd import export


def _atom(i, name, resn, resi):
    return f"ATOM  {i:5d} {name:<4s} {resn:>3s} A{resi:4d}    {0:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00\n"


def test_chained_pdb_labels_chains_in_residue_order(tmp_path):
    lines, i = [], 1
    for resi in range(1, 6 * export.RES_PER_CHAIN + 1):
        resn = "ACE" if resi % 32 == 1 else ("NME" if resi % 32 == 0 else "GLY")
        for name in ("N", "CA"):
            lines.append(_atom(i, name, resn, resi))
            i += 1
    for resi, resn in ((193, "CA"), (194, "Cl-")):
        lines.append(_atom(i, resn, resn, resi))
        i += 1
    src, dst = tmp_path / "s.pdb", tmp_path / "c.pdb"
    src.write_text("".join(lines) + "END\n")
    export.chained_pdb(src, dst)
    chains = [l[21] for l in dst.read_text().splitlines() if l.startswith("ATOM")]
    assert chains[:64] == ["A"] * 64
    assert chains[64 * 5:64 * 6] == ["F"] * 64
    assert chains[-2:] == ["I", "I"]
    assert len(chains) == len(lines)
