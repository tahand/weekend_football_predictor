"""Walk-forward backtesting and accuracy reporting."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from .features import build_features
from .models import OUTCOMES, ModelConfig, fit_model, predict_rows, training_rows

CONFIDENCE_BANDS = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 101)]
VALUE_EDGE = 0.05

Progress = Callable[[float, str], None]


def backtest(config: ModelConfig, results: pd.DataFrame, min_train: int = 100, step_days: int = 7,
             progress: Progress | None = None) -> pd.DataFrame:
    """Predict every match after the first `min_train` using only earlier matches.

    The model is retrained at the start of each `step_days` block, which mirrors
    retraining before each weekend's fixtures.
    """
    config.validate()
    features, _ = build_features(results, window=config.form_window)
    features = features.sort_values("Date", kind="stable").reset_index(drop=True)
    eligible = training_rows(config, features)
    if len(eligible) <= min_train:
        raise ValueError(f"Backtesting needs more than {min_train} results "
                         f"for '{config.name}' (has {len(eligible)})")

    start = eligible["Date"].iloc[min_train]
    end = features["Date"].max()
    blocks = []
    block_start = start
    while block_start <= end:
        blocks.append((block_start, block_start + timedelta(days=step_days)))
        block_start += timedelta(days=step_days)

    frames = []
    for i, (lo, hi) in enumerate(blocks):
        if progress:
            progress(i / len(blocks), f"{config.name}: block {i + 1} of {len(blocks)}")
        test = features[(features["Date"] >= lo) & (features["Date"] < hi)]
        if config.divisions:
            test = test[test["Div"].isin(config.divisions)]
        if test.empty:
            continue
        model = fit_model(config, features[features["Date"] < lo])
        frames.append(predict_rows(config, model, test))
    if progress:
        progress(1.0, f"{config.name}: done")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out.insert(0, "model_id", config.id)
        out.insert(1, "model_name", config.name)
    return out


def _bet_profit(rows: pd.DataFrame, picks: pd.Series) -> pd.Series:
    odds = np.select([picks == "H", picks == "D", picks == "A"],
                     [rows["odds_h"], rows["odds_d"], rows["odds_a"]], default=np.nan)
    won = picks == rows["FTR"]
    profit = np.where(won, odds - 1, -1.0)
    return pd.Series(np.where(np.isnan(odds), np.nan, profit), index=rows.index)


def evaluate(pred: pd.DataFrame) -> dict:
    """Accuracy metrics for predictions that have a known result (FTR/FTHG/FTAG)."""
    pred = pred.dropna(subset=["FTR"])
    n = len(pred)
    if n == 0:
        return {"matches": 0}

    probs = pred[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)
    actual = np.zeros_like(probs)
    actual_idx = pred["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    actual[np.arange(n), actual_idx] = 1
    correct = (pred["prediction"] == pred["FTR"]).to_numpy()

    p_actual = np.clip(probs[np.arange(n), actual_idx], 1e-15, 1)
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    cum_o = np.cumsum(actual, axis=1)[:, :2]

    metrics = {
        "matches": n,
        "accuracy": float(correct.mean()),
        "log_loss": float(-np.log(p_actual).mean()),
        "brier": float(((probs - actual) ** 2).sum(axis=1).mean()),
        "rps": float((0.5 * ((cum_p - cum_o) ** 2).sum(axis=1)).mean()),
        "exact_score": float(((pred["score_h"] == pred["FTHG"]) & (pred["score_a"] == pred["FTAG"])).mean()),
        "goal_diff_correct": float(((pred["score_h"] - pred["score_a"]) == (pred["FTHG"] - pred["FTAG"])).mean()),
        "goals_mae": float((np.abs(pred["xg_h"] - pred["FTHG"]) + np.abs(pred["xg_a"] - pred["FTAG"])).mean() / 2),
        "avg_confidence": float(pred["confidence"].mean()),
        "baseline_home": float((pred["FTR"] == "H").mean()),
    }

    profit = _bet_profit(pred, pred["prediction"])
    metrics["bets"] = int(profit.notna().sum())
    metrics["roi"] = float(profit.mean()) if profit.notna().any() else None

    pick_prob = probs.max(axis=1)
    pick_odds = np.select([pred["prediction"] == o for o in OUTCOMES],
                          [pred["odds_h"], pred["odds_d"], pred["odds_a"]], default=np.nan)
    value = pick_prob * pick_odds > 1 + VALUE_EDGE
    value_profit = profit[value]
    metrics["value_bets"] = int(value_profit.notna().sum())
    metrics["value_roi"] = float(value_profit.mean()) if value_profit.notna().any() else None

    market = pred[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    has_market = ~np.isnan(market).any(axis=1)
    if has_market.any():
        fav = np.array(OUTCOMES)[np.argmin(np.where(has_market[:, None], market, np.inf), axis=1)]
        metrics["baseline_favourite"] = float((fav[has_market] == pred["FTR"].to_numpy()[has_market]).mean())
    else:
        metrics["baseline_favourite"] = None

    metrics["confusion"] = {
        a: {p: int(((pred["FTR"] == a) & (pred["prediction"] == p)).sum()) for p in OUTCOMES} for a in OUTCOMES
    }
    metrics["by_outcome"] = []
    for o in OUTCOMES:
        picked = pred["prediction"] == o
        happened = pred["FTR"] == o
        metrics["by_outcome"].append({
            "outcome": o,
            "predicted": int(picked.sum()),
            "actual": int(happened.sum()),
            "precision": float((picked & happened).sum() / picked.sum()) if picked.any() else None,
            "recall": float((picked & happened).sum() / happened.sum()) if happened.any() else None,
        })

    metrics["by_confidence"] = []
    for lo, hi in CONFIDENCE_BANDS:
        band = (pred["confidence"] >= lo) & (pred["confidence"] < hi)
        label = f"{lo}%+" if hi > 100 else f"{lo}-{hi}%"
        metrics["by_confidence"].append({
            "band": label,
            "matches": int(band.sum()),
            "accuracy": float(correct[band.to_numpy()].mean()) if band.any() else None,
        })

    metrics["by_division"] = []
    for div, grp in pred.groupby("Div"):
        metrics["by_division"].append({
            "division": div,
            "matches": int(len(grp)),
            "accuracy": float((grp["prediction"] == grp["FTR"]).mean()),
        })

    # Calibration: pool every outcome probability and compare with how often it happened.
    flat_p = probs.ravel()
    flat_o = actual.ravel()
    edges = np.linspace(0, 1, 11)
    metrics["calibration"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (flat_p >= lo) & (flat_p < hi if hi < 1 else flat_p <= hi)
        if mask.sum():
            metrics["calibration"].append({
                "bin": f"{lo:.0%}-{hi:.0%}",
                "count": int(mask.sum()),
                "predicted": float(flat_p[mask].mean()),
                "observed": float(flat_o[mask].mean()),
            })
    return metrics


def attach_results(predictions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Join saved predictions to results that have since been imported."""
    if predictions.empty:
        return predictions.assign(FTHG=pd.Series(dtype=float), FTAG=pd.Series(dtype=float),
                                  FTR=pd.Series(dtype=object))
    keys = ["Div", "HomeTeam", "AwayTeam", "Date"]
    played = results[keys + ["FTHG", "FTAG", "FTR"]].drop_duplicates(keys, keep="last")
    base = predictions.drop(columns=[c for c in ("FTHG", "FTAG", "FTR") if c in predictions.columns])
    return base.merge(played, on=keys, how="left")


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _num(value, digits=3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_report(name: str, description: str, metrics: dict, title: str = "Backtest") -> str:
    """Plain-text report suitable for saving or sharing."""
    lines = [f"{title} report: {name}", "=" * (len(title) + len(name) + 9)]
    if description:
        lines.append(description)
    lines.append(f"Generated {datetime.now():%d %b %Y %H:%M}")
    lines.append("")
    if not metrics.get("matches"):
        lines.append("No matches with known results to evaluate yet.")
        return "\n".join(lines)
    lines += [
        f"Matches evaluated      {metrics['matches']}",
        f"Result accuracy        {_pct(metrics['accuracy'])}",
        f"  always-home baseline {_pct(metrics['baseline_home'])}",
        f"  bookies' favourite   {_pct(metrics['baseline_favourite'])}",
        f"Exact score correct    {_pct(metrics['exact_score'])}",
        f"Goal difference right  {_pct(metrics['goal_diff_correct'])}",
        f"Log loss (lower=better){_num(metrics['log_loss']):>7}",
        f"Brier score            {_num(metrics['brier'])}",
        f"Ranked prob. score     {_num(metrics['rps'])}",
        f"Goals error per team   {_num(metrics['goals_mae'], 2)}",
        f"Average confidence     {metrics['avg_confidence']:.1f}%",
        f"ROI backing every pick {_pct(metrics['roi'])} over {metrics['bets']} bets",
        f"ROI on value picks     {_pct(metrics['value_roi'])} over {metrics['value_bets']} bets",
        "",
        "Accuracy by confidence",
    ]
    for row in metrics["by_confidence"]:
        lines.append(f"  {row['band']:<8} {row['matches']:>5} matches  {_pct(row['accuracy'])}")
    lines += ["", "By outcome            predicted  actual  precision  recall"]
    for row in metrics["by_outcome"]:
        lines.append(f"  {row['outcome']:<20} {row['predicted']:>9} {row['actual']:>7} "
                     f"{_pct(row['precision']):>10} {_pct(row['recall']):>7}")
    lines += ["", "Confusion matrix (rows = actual, columns = predicted)", "       H     D     A"]
    for a in OUTCOMES:
        c = metrics["confusion"][a]
        lines.append(f"  {a}  {c['H']:>4}  {c['D']:>4}  {c['A']:>4}")
    lines += ["", "By division"]
    for row in metrics["by_division"]:
        lines.append(f"  {row['division']:<6} {row['matches']:>5} matches  {_pct(row['accuracy'])}")
    lines += ["", "Calibration (predicted vs observed frequency)"]
    for row in metrics["calibration"]:
        lines.append(f"  {row['bin']:<9} n={row['count']:<5} predicted {row['predicted']:.1%}  "
                     f"observed {row['observed']:.1%}")
    return "\n".join(lines)
