from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "example_files"


@pytest.fixture
def example_files():
    return sorted(EXAMPLES.glob("*.csv"))


def synthetic_results(n_rounds: int = 12, seed: int = 1) -> pd.DataFrame:
    """A small round-robin league where team strength drives the scores."""
    rng = np.random.default_rng(seed)
    teams = [f"Team {c}" for c in "ABCDEFGH"]
    strength = dict(zip(teams, np.linspace(0.6, 2.0, len(teams))))
    rows = []
    date = pd.Timestamp("2026-08-01")
    for r in range(n_rounds):
        order = rng.permutation(teams)
        for i in range(0, len(order), 2):
            home, away = order[i], order[i + 1]
            hg = rng.poisson(strength[home] * 1.1)
            ag = rng.poisson(strength[away] * 0.9)
            odds_h = 1 / max(0.05, min(0.9, 0.45 + (strength[home] - strength[away]) * 0.2))
            rows.append({"Div": "T1", "Date": date + pd.Timedelta(days=7 * r), "Time": "15:00",
                         "HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag,
                         "FTR": "H" if hg > ag else ("A" if hg < ag else "D"),
                         "HST": hg + 2, "AST": ag + 2, "HxG": hg * 0.9 + 0.2, "AxG": ag * 0.9 + 0.2,
                         "AvgH": odds_h, "AvgD": 3.4, "AvgA": 1 / max(0.05, 1 - 1 / odds_h - 0.27)})
    return pd.DataFrame(rows)


@pytest.fixture
def results():
    return synthetic_results()
