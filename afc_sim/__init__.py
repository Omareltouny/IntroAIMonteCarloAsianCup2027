"""Monte Carlo engine for the 2027 AFC Asian Cup (format, score layer, knockout resolution)."""
from .config import GROUPS, HOST, TEAMS
from .data import build_features, load_raw
from .engine import MatchEngine, train_xgb
from .knockout import ET_FACTOR
from .scores import ScoreModel
from .shootouts import ShootoutModel
from .simulate import Simulator
