"""On-disk storage for imported data, model configurations, saved predictions and reports."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data import (ImportSummary, DataFormatError, merge_matches, read_match_file, remove_played,
                   sort_matches, split_results_and_fixtures)
from .models import ModelConfig, default_models

HISTORY_COLUMNS = ["saved_at", "model_id", "model_name", "Div", "Date", "Time", "HomeTeam", "AwayTeam",
                   "p_h", "p_d", "p_a", "xg_h", "xg_a", "prediction", "score_h", "score_a", "confidence",
                   "odds_h", "odds_d", "odds_a"]


def default_data_dir() -> Path:
    override = os.environ.get("WFP_DATA_DIR")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or Path.home()
    return Path(base) / "WeekendFootballPredictor"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


class Workspace:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else default_data_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.exports_dir = self.root / "exports"
        self.exports_dir.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self.results = _read_csv(self.root / "results.csv")
        self.fixtures = _read_csv(self.root / "fixtures.csv")
        self.history = _read_csv(self.root / "prediction_history.csv")
        self.models = self._load_models()
        self.settings = self._load_json(self.root / "settings.json", {})

    # ------------------------------------------------------------ data

    def import_files(self, paths: list[str | Path]) -> ImportSummary:
        summary = ImportSummary()
        new_results, new_fixtures = [], []
        for path in paths:
            try:
                df = read_match_file(path)
            except (DataFormatError, OSError, ValueError) as exc:
                summary.errors.append(str(exc))
                continue
            results, fixtures = split_results_and_fixtures(df)
            summary.files.append(Path(path).name)
            summary.results_read += len(results)
            summary.fixtures_read += len(fixtures)
            if not results.empty:
                new_results.append(results)
            if not fixtures.empty:
                new_fixtures.append(fixtures)

        with self._lock:
            if new_results:
                before = len(self.results)
                incoming = pd.concat(new_results, ignore_index=True, sort=False)
                self.results = merge_matches(self.results, incoming)
                summary.results_added = len(self.results) - before
                _write_csv(self.results, self.root / "results.csv")
            if new_fixtures:
                # A fresh fixture import replaces the previous fixture list.
                incoming = pd.concat(new_fixtures, ignore_index=True, sort=False)
                self.fixtures = merge_matches(pd.DataFrame(), incoming)
            if new_results or new_fixtures:
                self.fixtures = remove_played(self.fixtures, self.results)
                _write_csv(self.fixtures, self.root / "fixtures.csv")
        return summary

    def clear_data(self) -> None:
        with self._lock:
            self.results = pd.DataFrame()
            self.fixtures = pd.DataFrame()
            for name in ("results.csv", "fixtures.csv"):
                (self.root / name).unlink(missing_ok=True)

    def clear_fixtures(self) -> None:
        with self._lock:
            self.fixtures = pd.DataFrame()
            (self.root / "fixtures.csv").unlink(missing_ok=True)

    def division_summary(self) -> pd.DataFrame:
        if self.results.empty:
            return pd.DataFrame(columns=["Div", "matches", "teams", "first", "last", "fixtures"])
        rows = []
        fixture_counts = self.fixtures["Div"].value_counts() if not self.fixtures.empty else {}
        for div, grp in self.results.groupby("Div"):
            teams = set(grp["HomeTeam"]) | set(grp["AwayTeam"])
            rows.append({"Div": div, "matches": len(grp), "teams": len(teams),
                         "first": grp["Date"].min(), "last": grp["Date"].max(),
                         "fixtures": int(fixture_counts.get(div, 0))})
        for div in (set(fixture_counts.index) if len(fixture_counts) else set()) - set(self.results["Div"]):
            rows.append({"Div": div, "matches": 0, "teams": 0, "first": pd.NaT, "last": pd.NaT,
                         "fixtures": int(fixture_counts[div])})
        return pd.DataFrame(rows).sort_values("Div").reset_index(drop=True)

    @property
    def divisions(self) -> list[str]:
        divs = set()
        for df in (self.results, self.fixtures):
            if not df.empty:
                divs |= set(df["Div"].astype(str))
        return sorted(divs)

    # ------------------------------------------------------------ models

    @staticmethod
    def _load_json(path: Path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    def _save_json(self, path: Path, data) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)

    def _load_models(self) -> list[ModelConfig]:
        data = self._load_json(self.root / "models.json", None)
        if data is None:
            models = default_models()
            self._save_json(self.root / "models.json", [m.to_dict() for m in models])
            return models
        return [ModelConfig.from_dict(d) for d in data]

    def save_models(self) -> None:
        with self._lock:
            self._save_json(self.root / "models.json", [m.to_dict() for m in self.models])

    def get_model(self, model_id: str) -> ModelConfig | None:
        return next((m for m in self.models if m.id == model_id), None)

    def upsert_model(self, config: ModelConfig) -> ModelConfig:
        config.validate()
        clash = next((m for m in self.models if m.name.lower() == config.name.lower() and m.id != config.id), None)
        if clash:
            raise ValueError(f"Another model is already called '{config.name}'")
        config.updated = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            for i, existing in enumerate(self.models):
                if existing.id == config.id:
                    self.models[i] = config
                    break
            else:
                self.models.append(config)
            self.save_models()
        return config

    def delete_model(self, model_id: str) -> None:
        with self._lock:
            self.models = [m for m in self.models if m.id != model_id]
            self.save_models()
            (self.reports_dir / f"{model_id}.json").unlink(missing_ok=True)
            (self.reports_dir / f"{model_id}_predictions.csv").unlink(missing_ok=True)

    def unique_name(self, base: str) -> str:
        names = {m.name.lower() for m in self.models}
        if base.lower() not in names:
            return base
        n = 2
        while f"{base} ({n})".lower() in names:
            n += 1
        return f"{base} ({n})"

    # ------------------------------------------------------------ backtest reports

    def save_backtest(self, config: ModelConfig, predictions: pd.DataFrame, metrics: dict, settings: dict) -> None:
        record = {"model_id": config.id, "model_name": config.name, "algorithm": config.algorithm_label,
                  "run_at": datetime.now().isoformat(timespec="seconds"), "settings": settings,
                  "config_updated": config.updated, "metrics": metrics}
        with self._lock:
            self._save_json(self.reports_dir / f"{config.id}.json", record)
            _write_csv(predictions, self.reports_dir / f"{config.id}_predictions.csv")

    def load_backtest(self, model_id: str) -> dict | None:
        return self._load_json(self.reports_dir / f"{model_id}.json", None)

    def load_backtest_predictions(self, model_id: str) -> pd.DataFrame:
        return _read_csv(self.reports_dir / f"{model_id}_predictions.csv")

    # ------------------------------------------------------------ prediction history

    def save_predictions(self, predictions: pd.DataFrame) -> int:
        rows = predictions.copy()
        rows["saved_at"] = datetime.now().isoformat(timespec="seconds")
        rows = rows[[c for c in HISTORY_COLUMNS if c in rows.columns]]
        with self._lock:
            combined = pd.concat([self.history, rows], ignore_index=True, sort=False) if not self.history.empty else rows
            # Keep the latest saved prediction per model and fixture.
            combined = combined.drop_duplicates(["model_id", "Div", "Date", "HomeTeam", "AwayTeam"], keep="last")
            self.history = sort_matches(combined)
            _write_csv(self.history, self.root / "prediction_history.csv")
        return len(rows)

    def clear_history(self) -> None:
        with self._lock:
            self.history = pd.DataFrame()
            (self.root / "prediction_history.csv").unlink(missing_ok=True)

    # ------------------------------------------------------------ settings & exports

    def set_setting(self, key: str, value) -> None:
        self.settings[key] = value
        self._save_json(self.root / "settings.json", self.settings)

    def export_frame(self, df: pd.DataFrame, stem: str) -> Path:
        path = self.exports_dir / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        _write_csv(df, path)
        return path

    def export_text(self, text: str, stem: str) -> Path:
        path = self.exports_dir / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path.write_text(text, encoding="utf-8")
        return path
