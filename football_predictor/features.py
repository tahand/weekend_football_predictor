"""Pre-match feature engineering.

Matches are processed in date order and every feature is computed from
matches played *before* the one being described, so backtests never see
the future.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import implied_probabilities, market_odds, over_25_probability, sort_matches

FEATURE_GROUPS: dict[str, str] = {
    "form": "Recent form (points, goals)",
    "xg": "Expected goals & shots",
    "venue": "Home / away form",
    "elo": "Elo strength rating",
    "market": "Bookmaker odds",
}

GROUP_COLUMNS: dict[str, list[str]] = {
    "form": ["h_pts", "h_gf", "h_ga", "a_pts", "a_gf", "a_ga", "diff_pts", "diff_gd"],
    "xg": ["h_xgf", "h_xga", "h_sotf", "h_sota", "a_xgf", "a_xga", "a_sotf", "a_sota", "diff_xgd"],
    "venue": ["h_home_pts", "h_home_gf", "h_home_ga", "a_away_pts", "a_away_gf", "a_away_ga"],
    "elo": ["elo_h", "elo_a", "elo_diff"],
    "market": ["mkt_ph", "mkt_pd", "mkt_pa", "mkt_over25", "ah_line"],
}

INFO_COLUMNS = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "odds_h", "odds_d", "odds_a",
                "h_played", "a_played"]
TARGET_COLUMNS = ["FTHG", "FTAG", "FTR"]

ELO_START = 1500.0
ELO_HOME_ADVANTAGE = 60.0


def feature_columns(groups: list[str]) -> list[str]:
    cols: list[str] = []
    for group in FEATURE_GROUPS:
        if group in groups:
            cols.extend(GROUP_COLUMNS[group])
    return cols


@dataclass
class _TeamState:
    window: int
    elo: float = ELO_START
    played: int = 0
    recent: deque = field(default=None)
    home: deque = field(default=None)
    away: deque = field(default=None)

    def __post_init__(self):
        self.recent = deque(maxlen=self.window)
        self.home = deque(maxlen=self.window)
        self.away = deque(maxlen=self.window)


def _mean(records, key) -> float:
    values = [r[key] for r in records if r[key] is not None and not math.isnan(r[key])]
    return float(np.mean(values)) if values else math.nan


def _points(gf: int, ga: int) -> int:
    return 3 if gf > ga else (1 if gf == ga else 0)


def _num(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value


class FeatureBuilder:
    """Walks through results in order, keeping rolling per-team state."""

    def __init__(self, window: int = 6, elo_k: float = 20.0):
        self.window = max(1, int(window))
        self.elo_k = elo_k
        self.teams: dict[str, _TeamState] = defaultdict(lambda: _TeamState(self.window))

    def describe(self, home: str, away: str) -> dict:
        h, a = self.teams[home], self.teams[away]
        row = {
            "h_played": h.played,
            "a_played": a.played,
            "h_pts": _mean(h.recent, "pts"),
            "h_gf": _mean(h.recent, "gf"),
            "h_ga": _mean(h.recent, "ga"),
            "a_pts": _mean(a.recent, "pts"),
            "a_gf": _mean(a.recent, "gf"),
            "a_ga": _mean(a.recent, "ga"),
            "h_xgf": _mean(h.recent, "xgf"),
            "h_xga": _mean(h.recent, "xga"),
            "h_sotf": _mean(h.recent, "sotf"),
            "h_sota": _mean(h.recent, "sota"),
            "a_xgf": _mean(a.recent, "xgf"),
            "a_xga": _mean(a.recent, "xga"),
            "a_sotf": _mean(a.recent, "sotf"),
            "a_sota": _mean(a.recent, "sota"),
            "h_home_pts": _mean(h.home, "pts"),
            "h_home_gf": _mean(h.home, "gf"),
            "h_home_ga": _mean(h.home, "ga"),
            "a_away_pts": _mean(a.away, "pts"),
            "a_away_gf": _mean(a.away, "gf"),
            "a_away_ga": _mean(a.away, "ga"),
            "elo_h": h.elo,
            "elo_a": a.elo,
            "elo_diff": h.elo + ELO_HOME_ADVANTAGE - a.elo,
        }
        row["diff_pts"] = row["h_pts"] - row["a_pts"]
        row["diff_gd"] = (row["h_gf"] - row["h_ga"]) - (row["a_gf"] - row["a_ga"])
        row["diff_xgd"] = (row["h_xgf"] - row["h_xga"]) - (row["a_xgf"] - row["a_xga"])
        return row

    def update(self, match) -> None:
        home, away = match["HomeTeam"], match["AwayTeam"]
        hg, ag = int(match["FTHG"]), int(match["FTAG"])
        hx, ax = _num(match.get("HxG")), _num(match.get("AxG"))
        hst, ast = _num(match.get("HST")), _num(match.get("AST"))
        h, a = self.teams[home], self.teams[away]

        home_rec = {"gf": hg, "ga": ag, "pts": _points(hg, ag), "xgf": hx, "xga": ax, "sotf": hst, "sota": ast}
        away_rec = {"gf": ag, "ga": hg, "pts": _points(ag, hg), "xgf": ax, "xga": hx, "sotf": ast, "sota": hst}
        h.recent.append(home_rec)
        h.home.append(home_rec)
        a.recent.append(away_rec)
        a.away.append(away_rec)
        h.played += 1
        a.played += 1

        expected_home = 1.0 / (1.0 + 10 ** ((a.elo - h.elo - ELO_HOME_ADVANTAGE) / 400.0))
        actual_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        margin = abs(hg - ag)
        multiplier = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11.0 + margin) / 8.0)
        delta = self.elo_k * multiplier * (actual_home - expected_home)
        h.elo += delta
        a.elo -= delta


def _market_features(df: pd.DataFrame) -> pd.DataFrame:
    odds = market_odds(df)
    probs = implied_probabilities(odds)
    out = pd.concat([odds, probs], axis=1)
    out["mkt_over25"] = over_25_probability(df)
    out["ah_line"] = df["AHh"].astype(float) if "AHh" in df.columns else np.nan
    return out


def build_features(results: pd.DataFrame, fixtures: pd.DataFrame | None = None,
                   window: int = 6, elo_k: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (training rows for every result, feature rows for every fixture)."""
    builder = FeatureBuilder(window=window, elo_k=elo_k)
    all_feature_cols = feature_columns(list(FEATURE_GROUPS))

    train_rows = []
    results = sort_matches(results) if results is not None and not results.empty else pd.DataFrame()
    for match in results.to_dict("records"):
        train_rows.append(builder.describe(match["HomeTeam"], match["AwayTeam"]))
        builder.update(match)

    train = _assemble(results, train_rows, all_feature_cols, with_targets=True)

    fixture_rows = []
    fixtures = fixtures if fixtures is not None else pd.DataFrame()
    if not fixtures.empty:
        fixtures = sort_matches(fixtures)
        for match in fixtures.to_dict("records"):
            fixture_rows.append(builder.describe(match["HomeTeam"], match["AwayTeam"]))
    upcoming = _assemble(fixtures, fixture_rows, all_feature_cols, with_targets=False)
    return train, upcoming


def _assemble(matches: pd.DataFrame, rows: list[dict], feature_cols: list[str], with_targets: bool) -> pd.DataFrame:
    columns = INFO_COLUMNS + feature_cols + (TARGET_COLUMNS if with_targets else [])
    if matches.empty:
        return pd.DataFrame(columns=columns)
    feats = pd.DataFrame(rows, index=matches.index)
    market = _market_features(matches)
    out = pd.concat([matches[[c for c in ("Div", "Date", "HomeTeam", "AwayTeam") if c in matches]],
                     feats, market], axis=1)
    out["Time"] = matches["Time"] if "Time" in matches.columns else ""
    if with_targets:
        for col in TARGET_COLUMNS:
            out[col] = matches[col]
    return out[columns].reset_index(drop=True)
