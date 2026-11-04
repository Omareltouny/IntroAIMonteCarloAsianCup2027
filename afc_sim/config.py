"""Static description of the 2027 AFC Asian Cup (Saudi Arabia, 7 Jan - 5 Feb 2027).

Source: https://en.wikipedia.org/wiki/2027_AFC_Asian_Cup (draw held 9 May 2026).
Team names follow the Kaggle ``results.csv`` spelling (e.g. "China", not "China PR").
"""
from __future__ import annotations

HOST = "Saudi Arabia"

# Draw-position order (A1..A4 etc.).  Position matters only for the (assumed)
# matchday schedule below; results do not depend on it otherwise.
GROUPS: dict[str, list[str]] = {
    "A": ["Saudi Arabia", "Kuwait", "Oman", "Palestine"],
    "B": ["Uzbekistan", "Bahrain", "North Korea", "Jordan"],
    "C": ["Iran", "Syria", "Kyrgyzstan", "China"],
    "D": ["Australia", "Tajikistan", "Iraq", "Singapore"],
    "E": ["South Korea", "United Arab Emirates", "Vietnam", "Yemen"],
    "F": ["Japan", "Qatar", "Thailand", "Indonesia"],
}
TEAMS: list[str] = [t for g in GROUPS.values() for t in g]
GROUP_OF: dict[str, str] = {t: g for g, ts in GROUPS.items() for t in ts}

# Round-robin pairs by draw position (1-based).  The official matchday order was
# NOT available on the wiki page, so the FIFA four-team template is assumed.
# It only matters for tiebreak rule 7 (shoot-out between two teams that met in
# the last round) -- an extremely rare event -- and not for any other result.
MATCHDAYS: list[list[tuple[int, int]]] = [
    [(1, 2), (3, 4)],
    [(4, 2), (1, 3)],
    [(4, 1), (2, 3)],
]
LAST_ROUND_PAIRS = {frozenset(p) for p in MATCHDAYS[-1]}

N_THIRD_ADVANCING = 4

# ---------------------------------------------------------------- knockout bracket
# Slot kinds: ("W", g) winner of group g, ("R", g) runner-up of group g,
#             ("T", g) the third-placed team assigned to the opponent of 1g
#             (resolved through THIRD_PLACE_TABLE).
R16: dict[int, tuple[tuple[str, str], tuple[str, str]]] = {
    37: (("R", "A"), ("R", "C")),
    38: (("W", "B"), ("T", "B")),   # 1B v 3A/C/D
    39: (("W", "D"), ("T", "D")),   # 1D v 3B/E/F
    40: (("W", "A"), ("T", "A")),   # 1A v 3C/D/E
    41: (("W", "F"), ("R", "E")),
    42: (("R", "B"), ("R", "F")),
    43: (("W", "E"), ("R", "D")),
    44: (("W", "C"), ("T", "C")),   # 1C v 3A/B/F
}
# Later rounds: match -> (feeder match, feeder match).
QF: dict[int, tuple[int, int]] = {45: (37, 39), 46: (38, 41), 47: (44, 43), 48: (40, 42)}
SF: dict[int, tuple[int, int]] = {49: (45, 46), 50: (47, 48)}
FINAL: dict[int, tuple[int, int]] = {51: (49, 50)}  # no third-place play-off

# Which third-placed groups qualify -> third-placed group meeting 1A, 1B, 1C, 1D.
_RAW_THIRD = """
ABCD CDAB
ABCE CABE
ABCF CABF
ABDE DABE
ABDF DABF
ABEF EABF
ACDE CDAE
ACDF CDAF
ACEF CAFE
ADEF DAFE
BCDE CDBE
BCDF CDBF
BCEF ECBF
BDEF EDBF
CDEF CDFE
"""
THIRD_PLACE_TABLE: dict[frozenset, dict[str, str]] = {}
for _line in _RAW_THIRD.strip().splitlines():
    _combo, _opp = _line.split()
    THIRD_PLACE_TABLE[frozenset(_combo)] = dict(zip("ABCD", _opp))

# Groups whose third-placed team each R16 slot may receive (from the bracket page).
ALLOWED_THIRD = {"A": set("CDE"), "B": set("ACD"), "C": set("ABF"), "D": set("BEF")}
