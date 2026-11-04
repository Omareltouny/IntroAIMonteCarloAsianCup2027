"""Replays the real 2023 Asian Cup group stage through the format code (needs data/)."""
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(not Path("data/results.csv").exists(), reason="data/ not present")


def test_2023_group_stage_replay_reproduces_the_real_round_of_16():
    import validation as v
    res, sho = v.load_raw(v.DATA)
    grp, ko = v.load_2023(res, sho)
    r = v.replay(grp, ko, sho)
    assert r["matched_r16_ties"] == 8                                   # all eight real R16 ties
    assert r["winners"] == dict(A="Qatar", B="Australia", C="Iran", D="Iraq", E="Bahrain", F="Saudi Arabia")
    assert r["runners_up"] == dict(A="Tajikistan", B="Uzbekistan", C="United Arab Emirates", D="Japan",
                                   E="South Korea", F="Thailand")
    assert set(r["best_thirds"].values()) == {"Jordan", "Palestine", "Syria", "Indonesia"}
