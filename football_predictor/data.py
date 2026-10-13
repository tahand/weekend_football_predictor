"""Reading and merging football-data.co.uk style CSV files.

A file can hold played matches (with FTHG/FTAG), upcoming fixtures, or a mix.
Played rows become results; rows without a score become fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLUMNS = ["Div", "Date", "HomeTeam", "AwayTeam"]
GOAL_COLUMNS = ["FTHG", "FTAG"]
TEXT_COLUMNS = {"Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTR", "HTR", "Referee"}

# Odds prefixes tried in order when picking the market price for a match.
# Opening prices are used (not closing 'C' prices) because fixture files
# are published before the closing prices exist.
ODDS_PREFIXES = ["Avg", "B365", "BW", "PS", "WH", "Max", "BFE", "BFD", "BV", "PP", "SKB"]
OVER_UNDER_PREFIXES = ["Avg", "B365", "Max", "BFE", "P"]


class DataFormatError(ValueError):
    """Raised when a file does not look like a match data file."""


@dataclass
class ImportSummary:
    files: list[str] = field(default_factory=list)
    results_read: int = 0
    results_added: int = 0
    fixtures_read: int = 0
    errors: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [f"Read {len(self.files)} file(s)"]
        if self.results_read:
            parts.append(f"{self.results_read} results ({self.results_added} new)")
        if self.fixtures_read:
            parts.append(f"{self.fixtures_read} fixtures")
        text = ", ".join(parts) + "."
        if self.errors:
            text += " Problems: " + "; ".join(self.errors)
        return text


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse dd/mm/yyyy or dd/mm/yy dates (football-data format), falling back to day-first parsing."""
    text = values.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed[missing] = pd.to_datetime(text[missing], format="%d/%m/%y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed[missing] = pd.to_datetime(text[missing], dayfirst=True, errors="coerce")
    return parsed


def read_match_file(path: str | Path) -> pd.DataFrame:
    """Read one CSV file into a clean DataFrame (dates parsed, numbers numeric)."""
    path = Path(path)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise DataFormatError(f"{path.name}: file is empty") from exc

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise DataFormatError(f"{path.name}: missing column(s) {', '.join(missing)}")

    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    for col in ("Div", "HomeTeam", "AwayTeam"):
        df[col] = df[col].astype(str).str.strip()
    df["Date"] = parse_dates(df["Date"])
    df = df.dropna(subset=["Date"])

    for col in df.columns:
        if col not in TEXT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def split_results_and_fixtures(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows with a full-time score are results; the rest are fixtures."""
    if all(c in df.columns for c in GOAL_COLUMNS):
        played = df[GOAL_COLUMNS].notna().all(axis=1)
    else:
        played = pd.Series(False, index=df.index)
    results = df[played].copy()
    fixtures = df[~played].copy()
    if not results.empty:
        results["FTHG"] = results["FTHG"].astype(int)
        results["FTAG"] = results["FTAG"].astype(int)
        results["FTR"] = result_codes(results["FTHG"], results["FTAG"])
    fixtures = fixtures.drop(columns=[c for c in GOAL_COLUMNS + ["FTR"] if c in fixtures.columns])
    return results.reset_index(drop=True), fixtures.reset_index(drop=True)


def result_codes(home_goals, away_goals) -> np.ndarray:
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    return np.where(home_goals > away_goals, "H", np.where(home_goals < away_goals, "A", "D"))


def merge_matches(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Combine match tables, newer rows replacing older rows for the same match."""
    frames = [f for f in (existing, new) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=KEY_COLUMNS)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    return sort_matches(merged)


def sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Date", "Time"] if "Time" in df.columns else ["Date"]
    return df.sort_values(order + ["Div", "HomeTeam"], kind="stable").reset_index(drop=True)


def remove_played(fixtures: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Drop fixtures that now appear in the results (same teams, same date)."""
    if fixtures.empty or results.empty:
        return fixtures
    played = set(zip(results["Div"], results["Date"], results["HomeTeam"], results["AwayTeam"]))
    keep = [
        (d, dt, h, a) not in played
        for d, dt, h, a in zip(fixtures["Div"], fixtures["Date"], fixtures["HomeTeam"], fixtures["AwayTeam"])
    ]
    return fixtures[keep].reset_index(drop=True)


def market_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Best available home/draw/away decimal odds for each match (NaN when none)."""
    odds = pd.DataFrame(np.nan, index=df.index, columns=["odds_h", "odds_d", "odds_a"])
    for prefix in ODDS_PREFIXES:
        cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
        if not all(c in df.columns for c in cols):
            continue
        candidate = df[cols].astype(float)
        usable = (candidate > 1).all(axis=1) & odds["odds_h"].isna()
        odds.loc[usable, :] = candidate[usable].to_numpy()
    return odds


def implied_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    """Convert decimal odds into probabilities with the bookmaker margin removed."""
    inverse = 1.0 / odds[["odds_h", "odds_d", "odds_a"]]
    total = inverse.sum(axis=1, min_count=3)
    probs = inverse.div(total, axis=0)
    probs.columns = ["mkt_ph", "mkt_pd", "mkt_pa"]
    return probs


def over_25_probability(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=df.index)
    for prefix in OVER_UNDER_PREFIXES:
        over, under = f"{prefix}>2.5", f"{prefix}<2.5"
        if over not in df.columns or under not in df.columns:
            continue
        o = df[over].astype(float)
        u = df[under].astype(float)
        usable = (o > 1) & (u > 1) & result.isna()
        result[usable] = (1 / o[usable]) / (1 / o[usable] + 1 / u[usable])
    return result
