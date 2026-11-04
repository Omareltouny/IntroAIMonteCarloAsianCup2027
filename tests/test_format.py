import itertools, numpy as np, pytest
from afc_sim.config import ALLOWED_THIRD, GROUPS, THIRD_PLACE_TABLE, R16, QF, SF, FINAL, MATCHDAYS
from afc_sim.groupstage import rank_group, rank_thirds
from afc_sim.knockout import build_r16

RNG = lambda s=0: np.random.default_rng(s)

# ---- format constants -------------------------------------------------------
def test_groups_are_six_by_four_and_unique():
    assert len(GROUPS) == 6 and all(len(v) == 4 for v in GROUPS.values())
    allt = [t for v in GROUPS.values() for t in v]
    assert len(allt) == len(set(allt)) == 24

def test_round_robin_covers_all_six_pairs_once():
    pairs = [frozenset(p) for md in MATCHDAYS for p in md]
    assert len(pairs) == 6 and len(set(pairs)) == 6

def test_third_place_table_complete_and_valid():
    assert set(THIRD_PLACE_TABLE) == {frozenset(c) for c in itertools.combinations("ABCDEF", 4)}
    for combo, row in THIRD_PLACE_TABLE.items():
        assert set(row.values()) == set(combo)                      # uses exactly the qualifying groups
        for w, tg in row.items():
            assert tg in ALLOWED_THIRD[w]                           # respects the bracket's slot limits

def test_third_place_table_matches_wiki_examples():
    assert THIRD_PLACE_TABLE[frozenset("ABCD")] == dict(A="C", B="D", C="A", D="B")
    assert THIRD_PLACE_TABLE[frozenset("CDEF")] == dict(A="C", B="D", C="F", D="E")

def test_bracket_is_a_proper_tree():
    feeders = [f for rnd in (QF, SF, FINAL) for pair in rnd.values() for f in pair]
    assert sorted(feeders) == sorted(list(R16) + list(QF) + list(SF))   # each match feeds exactly one later match

@pytest.mark.parametrize("combo", list(itertools.combinations("ABCDEF", 4)))
def test_r16_has_16_distinct_teams_and_no_same_group_clash(combo):
    W = {g: f"W{g}" for g in GROUPS}; R = {g: f"R{g}" for g in GROUPS}; T = {g: f"T{g}" for g in combo}
    r16 = build_r16(W, R, T)
    teams = [t for p in r16.values() for t in p]
    assert len(teams) == len(set(teams)) == 16
    for a, b in r16.values():
        assert a[1] != b[1], (a, b)                                  # never a same-group R16 rematch

# ---- tiebreakers ---------------------------------------------------------------
def rg(results, seed=0, **kw):
    return rank_group([0, 1, 2, 3], results, RNG(seed), **kw)[0]

def test_points_order():
    res = [(0,1,1,0),(0,2,1,0),(0,3,1,0),(1,2,1,0),(1,3,1,0),(2,3,1,0)]
    assert rg(res) == [0, 1, 2, 3]

def test_head_to_head_beats_overall_goal_difference():
    # 0 and 1 both on 6 pts; 1 won the meeting although 0 has the far better overall GD (+5 v +1)
    res = [(0,1,0,1),(0,2,5,0),(0,3,1,0),(1,2,1,0),(1,3,0,1),(2,3,0,0)]
    assert rg(res) == [1, 0, 3, 2]

def test_h2h_goals_scored_splits_when_points_and_gd_level():
    # all six games drawn: 0 and 1 scored in their meeting, so they rank above 2 and 3 (rule 3)
    res = [(0,1,1,1),(0,2,0,0),(0,3,0,0),(1,2,0,0),(1,3,0,0),(2,3,0,0)]
    for s in range(20):
        o = rg(res, seed=s)
        assert set(o[:2]) == {0, 1} and set(o[2:]) == {2, 3}

def test_criterion4_reapplies_h2h_to_the_remaining_subset():
    # four teams on 4 pts, GD 0.  Team 1 has fewest goals and drops to the bottom of the
    # four-team mini-league; the other three are level on overall GD/GF, so only the
    # re-applied head-to-head among {0,2,3} (2: 4 pts, 0: 3 pts, 3: 1 pt) can order them.
    res = [(0,1,1,1),(0,2,1,2),(0,3,2,1),(1,2,1,0),(1,3,0,1),(2,3,2,2)]
    for s in range(20):
        assert rg(res, seed=s) == [2, 0, 3, 1]

def test_criterion4_second_case():
    res = [(0,1,2,1),(0,2,2,2),(0,3,0,1),(1,2,2,1),(1,3,1,1),(2,3,1,0)]
    for s in range(20):
        assert rg(res, seed=s) == [0, 1, 2, 3]

def test_circular_head_to_head_falls_to_overall_gd():
    res = [(0,1,1,0),(1,2,1,0),(2,0,1,0),(0,3,3,0),(1,3,2,0),(2,3,1,0)]
    assert rg(res) == [0, 1, 2, 3]                       # GD +3, +2, +1

def test_overall_goals_scored_after_equal_gd():
    # trio level on points and h2h; all +3 overall; goals scored 6 > 5 > 4
    res = [(0,1,1,0),(1,2,1,0),(2,0,1,0),(0,3,4,1),(1,3,3,0),(2,3,5,2)]
    assert rg(res) == [2, 0, 1, 3]

def test_lots_when_everything_equal_is_uniform():
    res = [(a, b, 0, 0) for a in range(4) for b in range(a + 1, 4)]
    first = [rg(res, seed=s)[0] for s in range(800)]
    assert np.bincount(first, minlength=4).min() > 140   # ~200 each

def test_rule7_shootout_only_for_two_teams_that_met_in_last_round():
    # 0 and 1 drew 0-0 and are identical in every other respect
    res = [(0,1,0,0),(0,2,1,0),(0,3,1,0),(1,2,1,0),(1,3,1,0),(2,3,0,0)]
    called = []
    def sw(a, b):
        called.append((a, b)); return b                  # always let the second team win
    out = rg(res, last_round={frozenset((0, 1))}, shootout_winner=sw)
    assert called and out[:2] == [1, 0]
    called.clear()
    rg(res, last_round={frozenset((0, 2))}, shootout_winner=sw)   # tied pairs (0,1),(2,3) not in last round
    assert not called                                    # did not meet in the last round -> lots

def test_rank_thirds_by_points_then_gd_then_gf():
    ent = [("A",3,0,2),("B",4,-1,1),("C",3,1,1),("D",3,1,3),("E",1,-2,0),("F",3,0,2)]
    order = [e[0] for e in rank_thirds(ent, RNG())]
    assert order[:3] == ["B", "D", "C"] and order[-1] == "E" and set(order[3:5]) == {"A", "F"}
