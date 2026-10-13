"""Prediction model pipelines and their user-editable configuration.

Every algorithm produces home/draw/away probabilities plus expected goals
for each side. Expected goals drive the predicted scoreline.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_GROUPS, build_features, feature_columns

OUTCOMES = ["H", "D", "A"]
OUTCOME_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}
MIN_TRAINING_MATCHES = 30


class InsufficientDataError(RuntimeError):
    pass


# ---------------------------------------------------------------- parameters

@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: str  # "int", "float", "choice", "multi"
    default: object
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple = ()
    help: str = ""

    def coerce(self, value):
        if self.kind == "int":
            value = int(float(value))
        elif self.kind == "float":
            value = float(value)
        elif self.kind == "choice":
            value = str(value)
            if value not in self.choices:
                raise ValueError(f"{self.label} must be one of {', '.join(self.choices)}")
            return value
        elif self.kind == "multi":
            items = value.split(",") if isinstance(value, str) else list(value)
            value = [str(v).strip() for v in items if str(v).strip()]
            bad = [v for v in value if v not in self.choices]
            if bad or not value:
                raise ValueError(f"{self.label} must be chosen from {', '.join(self.choices)}")
            return value
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.label} must be at least {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.label} must be at most {self.maximum}")
        return value


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    label: str
    description: str
    params: tuple[ParamSpec, ...]

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}


_CLASS_WEIGHT = ParamSpec("class_weight", "Class weighting", "choice", "none", choices=("none", "balanced"),
                          help="'balanced' makes the model predict draws more often")
_DRAW_RHO = ParamSpec("rho", "Low-score correlation (rho)", "float", -0.05, -0.3, 0.3,
                      help="Dixon-Coles adjustment; negative values add draws like 0-0 and 1-1")

ALGORITHMS: dict[str, AlgorithmSpec] = {
    spec.key: spec for spec in (
        AlgorithmSpec(
            "market_odds", "Bookmaker odds baseline",
            "Uses the bookmakers' implied probabilities. A benchmark the other models should try to beat.",
            (),
        ),
        AlgorithmSpec(
            "logistic", "Logistic regression",
            "Multinomial logistic regression for the result, Poisson regression for goals.",
            (ParamSpec("C", "Regularisation strength (C)", "float", 0.02, 0.0001, 100,
                       help="Smaller values give simpler, more cautious models"),
             _CLASS_WEIGHT),
        ),
        AlgorithmSpec(
            "random_forest", "Random forest",
            "An ensemble of decision trees for the result and for each team's goals.",
            (ParamSpec("n_estimators", "Number of trees", "int", 200, 10, 2000),
             ParamSpec("max_depth", "Maximum tree depth", "int", 6, 1, 50),
             ParamSpec("min_samples_leaf", "Minimum matches per leaf", "int", 5, 1, 200),
             _CLASS_WEIGHT),
        ),
        AlgorithmSpec(
            "gradient_boosting", "Gradient boosting",
            "Histogram gradient-boosted trees; copes with missing statistics natively.",
            (ParamSpec("learning_rate", "Learning rate", "float", 0.02, 0.001, 1.0),
             ParamSpec("max_iter", "Boosting rounds", "int", 100, 10, 2000),
             ParamSpec("max_depth", "Maximum tree depth", "int", 2, 1, 20),
             ParamSpec("l2_regularization", "L2 regularisation", "float", 10.0, 0.0, 100.0)),
        ),
        AlgorithmSpec(
            "poisson", "Poisson goals model",
            "Predicts each team's goals, then derives result probabilities from the scoreline grid.",
            (ParamSpec("alpha", "Regularisation (alpha)", "float", 3.0, 0.0, 100.0),
             _DRAW_RHO),
        ),
        AlgorithmSpec(
            "ensemble", "Blended ensemble",
            "Averages the probabilities and expected goals of several algorithms.",
            (ParamSpec("components", "Blend components", "multi",
                       ["logistic", "poisson", "gradient_boosting", "market_odds"],
                       choices=("market_odds", "logistic", "random_forest", "gradient_boosting", "poisson")),),
        ),
    )
}


# ---------------------------------------------------------------- algorithms

def _encode(results: pd.Series) -> np.ndarray:
    return results.map({"H": 0, "D": 1, "A": 2}).to_numpy()


def _class_probs(model, X) -> np.ndarray:
    """predict_proba re-ordered to H/D/A, padding classes absent from training."""
    raw = model.predict_proba(X)
    probs = np.zeros((len(X), 3))
    for i, cls in enumerate(model.classes_):
        probs[:, int(cls)] = raw[:, i]
    return probs


def _linear(estimator):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), estimator)


def _imputed(estimator):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), estimator)


class _DropEmptyColumns(TransformerMixin, BaseEstimator):
    """Remove columns with no values in training (e.g. xG before any xG data exists)."""

    def fit(self, X, y=None):
        self.keep_ = ~np.isnan(np.asarray(X, dtype=float)).all(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if not self.keep_.any():
            return np.zeros((len(X), 1))
        return X[:, self.keep_]


def _native_nan(estimator):
    return make_pipeline(_DropEmptyColumns(), estimator)


class Algorithm:
    def __init__(self, params: dict):
        self.params = params

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "Algorithm":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (probabilities n x 3 in H/D/A order, expected goals n x 2)."""
        raise NotImplementedError


class _ClassifierPlusGoals(Algorithm):
    def make_classifier(self):
        raise NotImplementedError

    def make_regressor(self):
        raise NotImplementedError

    def fit(self, X, y):
        self.classifier = self.make_classifier().fit(X, _encode(y["FTR"]))
        self.home_goals = self.make_regressor().fit(X, y["FTHG"])
        self.away_goals = self.make_regressor().fit(X, y["FTAG"])
        return self

    def predict(self, X):
        goals = np.column_stack([self.home_goals.predict(X), self.away_goals.predict(X)])
        return _class_probs(self.classifier, X), np.clip(goals, 0.05, None)


def _class_weight(params):
    return None if params.get("class_weight", "none") == "none" else "balanced"


class LogisticAlgorithm(_ClassifierPlusGoals):
    def make_classifier(self):
        return _linear(LogisticRegression(C=self.params["C"], class_weight=_class_weight(self.params),
                                          max_iter=2000))

    def make_regressor(self):
        # Tie the goals model's regularisation to the same C setting.
        return _linear(PoissonRegressor(alpha=0.1 / max(self.params["C"], 1e-6), max_iter=1000))


class RandomForestAlgorithm(_ClassifierPlusGoals):
    def _common(self):
        return dict(n_estimators=self.params["n_estimators"], max_depth=self.params["max_depth"],
                    min_samples_leaf=self.params["min_samples_leaf"], random_state=42, n_jobs=-1)

    def make_classifier(self):
        return _imputed(RandomForestClassifier(class_weight=_class_weight(self.params), **self._common()))

    def make_regressor(self):
        return _imputed(RandomForestRegressor(**self._common()))


class GradientBoostingAlgorithm(_ClassifierPlusGoals):
    def _common(self):
        return dict(learning_rate=self.params["learning_rate"], max_iter=self.params["max_iter"],
                    max_depth=self.params["max_depth"], l2_regularization=self.params["l2_regularization"],
                    random_state=42)

    def make_classifier(self):
        return _native_nan(HistGradientBoostingClassifier(**self._common()))

    def make_regressor(self):
        return _native_nan(HistGradientBoostingRegressor(loss="poisson", **self._common()))


class PoissonAlgorithm(Algorithm):
    def fit(self, X, y):
        self.home_goals = _linear(PoissonRegressor(alpha=self.params["alpha"], max_iter=1000)).fit(X, y["FTHG"])
        self.away_goals = _linear(PoissonRegressor(alpha=self.params["alpha"], max_iter=1000)).fit(X, y["FTAG"])
        return self

    def predict(self, X):
        goals = np.clip(np.column_stack([self.home_goals.predict(X), self.away_goals.predict(X)]), 0.05, None)
        probs = np.array([outcome_probabilities(score_matrix(h, a, rho=self.params["rho"])) for h, a in goals])
        return probs, goals


class MarketOddsAlgorithm(Algorithm):
    """Bookmaker probabilities; falls back to training base rates when a match has no odds."""

    def fit(self, X, y):
        counts = y["FTR"].value_counts(normalize=True)
        self.base_rates = np.array([counts.get(o, 0.0) for o in OUTCOMES])
        self.home_goals = _linear(PoissonRegressor(alpha=1.0, max_iter=1000)).fit(X, y["FTHG"])
        self.away_goals = _linear(PoissonRegressor(alpha=1.0, max_iter=1000)).fit(X, y["FTAG"])
        return self

    def predict(self, X):
        n = len(X)
        if all(c in X.columns for c in ("mkt_ph", "mkt_pd", "mkt_pa")):
            probs = X[["mkt_ph", "mkt_pd", "mkt_pa"]].to_numpy(dtype=float, copy=True)
        else:
            probs = np.full((n, 3), np.nan)
        missing = np.isnan(probs).any(axis=1)
        probs[missing] = self.base_rates
        goals = np.clip(np.column_stack([self.home_goals.predict(X), self.away_goals.predict(X)]), 0.05, None)
        return probs, goals


class EnsembleAlgorithm(Algorithm):
    def fit(self, X, y):
        self.members = []
        for key in self.params["components"]:
            spec = ALGORITHMS[key]
            self.members.append(make_algorithm(key, spec.defaults()).fit(X, y))
        return self

    def predict(self, X):
        outputs = [m.predict(X) for m in self.members]
        probs = np.mean([p for p, _ in outputs], axis=0)
        goals = np.mean([g for _, g in outputs], axis=0)
        return probs, goals


_IMPLEMENTATIONS = {
    "market_odds": MarketOddsAlgorithm,
    "logistic": LogisticAlgorithm,
    "random_forest": RandomForestAlgorithm,
    "gradient_boosting": GradientBoostingAlgorithm,
    "poisson": PoissonAlgorithm,
    "ensemble": EnsembleAlgorithm,
}


def make_algorithm(key: str, params: dict) -> Algorithm:
    spec = ALGORITHMS[key]
    merged = spec.defaults()
    merged.update({k: v for k, v in params.items() if k in merged})
    return _IMPLEMENTATIONS[key](merged)


# ---------------------------------------------------------------- scorelines

def score_matrix(home_xg: float, away_xg: float, max_goals: int = 10, rho: float = 0.0) -> np.ndarray:
    """Probability grid P[home goals, away goals] with an optional Dixon-Coles adjustment."""
    goals = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(goals, home_xg), poisson.pmf(goals, away_xg))
    if rho:
        grid[0, 0] *= max(1 - home_xg * away_xg * rho, 0)
        grid[0, 1] *= max(1 + home_xg * rho, 0)
        grid[1, 0] *= max(1 + away_xg * rho, 0)
        grid[1, 1] *= max(1 - rho, 0)
    return grid / grid.sum()


def outcome_probabilities(grid: np.ndarray) -> np.ndarray:
    return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])


def likely_score(home_xg: float, away_xg: float, outcome: str) -> tuple[int, int]:
    """Most likely scoreline that agrees with the predicted outcome."""
    grid = score_matrix(home_xg, away_xg)
    h, a = np.indices(grid.shape)
    mask = {"H": h > a, "D": h == a, "A": h < a}[outcome]
    masked = np.where(mask, grid, -1)
    i, j = np.unravel_index(np.argmax(masked), grid.shape)
    return int(i), int(j)


def confidence_label(confidence: float) -> str:
    if confidence >= 60:
        return "High"
    if confidence >= 47:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------- configuration

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class ModelConfig:
    name: str
    algorithm: str
    params: dict = field(default_factory=dict)
    feature_groups: list[str] = field(default_factory=lambda: list(FEATURE_GROUPS))
    form_window: int = 6
    divisions: list[str] = field(default_factory=list)  # empty = train on every division
    description: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)

    def validate(self) -> "ModelConfig":
        if not self.name.strip():
            raise ValueError("The model needs a name")
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{self.algorithm}'")
        spec = ALGORITHMS[self.algorithm]
        clean = spec.defaults()
        for p in spec.params:
            if p.name in self.params:
                clean[p.name] = p.coerce(self.params[p.name])
        self.params = clean
        self.feature_groups = [g for g in FEATURE_GROUPS if g in self.feature_groups]
        if not self.feature_groups:
            raise ValueError("Choose at least one feature group")
        self.form_window = int(self.form_window)
        if not 1 <= self.form_window <= 50:
            raise ValueError("Form window must be between 1 and 50 matches")
        self.divisions = [d.strip() for d in self.divisions if d.strip()]
        self.name = self.name.strip()
        return self

    @property
    def algorithm_label(self) -> str:
        spec = ALGORITHMS.get(self.algorithm)
        return spec.label if spec else self.algorithm

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def default_models() -> list[ModelConfig]:
    return [
        ModelConfig("Bookie Benchmark", "market_odds", feature_groups=["market"],
                    description="What the bookmakers think - the bar to beat."),
        ModelConfig("Form Finder", "logistic", feature_groups=["form", "venue", "elo"],
                    description="Recent form, home/away form and Elo rating; ignores the odds."),
        ModelConfig("xG Forest", "random_forest", feature_groups=["form", "xg", "elo", "market"],
                    description="Random forest leaning on expected goals and shot data."),
        ModelConfig("Goal Machine", "poisson", feature_groups=["form", "xg", "venue", "elo"],
                    description="Models each side's goals to find the most likely scoreline."),
        ModelConfig("Boosted All-Rounder", "gradient_boosting", feature_groups=list(FEATURE_GROUPS),
                    description="Gradient boosting using every available feature."),
        ModelConfig("The Committee", "ensemble", feature_groups=list(FEATURE_GROUPS),
                    description="Blends several algorithms for a consensus view."),
    ]


# ---------------------------------------------------------------- training & prediction

def training_rows(config: ModelConfig, train: pd.DataFrame) -> pd.DataFrame:
    if config.divisions:
        train = train[train["Div"].isin(config.divisions)]
    return train


def fit_model(config: ModelConfig, train: pd.DataFrame) -> Algorithm:
    train = training_rows(config, train)
    if len(train) < MIN_TRAINING_MATCHES:
        raise InsufficientDataError(
            f"'{config.name}' needs at least {MIN_TRAINING_MATCHES} results to train (has {len(train)})")
    cols = feature_columns(config.feature_groups)
    return make_algorithm(config.algorithm, config.params).fit(train[cols], train[["FTR", "FTHG", "FTAG"]])


def predict_rows(config: ModelConfig, model: Algorithm, rows: pd.DataFrame) -> pd.DataFrame:
    """Add probability, expected goals, prediction, scoreline and confidence columns."""
    out = rows.copy()
    if rows.empty:
        for col in ("p_h", "p_d", "p_a", "xg_h", "xg_a", "prediction", "score_h", "score_a", "confidence"):
            out[col] = []
        return out
    probs, goals = model.predict(rows[feature_columns(config.feature_groups)])
    probs = np.clip(probs, 1e-6, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return attach_predictions(out, probs, goals)


def attach_predictions(out: pd.DataFrame, probs: np.ndarray, goals: np.ndarray) -> pd.DataFrame:
    out["p_h"], out["p_d"], out["p_a"] = probs[:, 0], probs[:, 1], probs[:, 2]
    out["xg_h"], out["xg_a"] = goals[:, 0], goals[:, 1]
    picks = [OUTCOMES[i] for i in probs.argmax(axis=1)]
    scores = [likely_score(h, a, o) for (h, a), o in zip(goals, picks)]
    out["prediction"] = picks
    out["score_h"] = [s[0] for s in scores]
    out["score_a"] = [s[1] for s in scores]
    out["confidence"] = probs.max(axis=1) * 100
    return out


def predict_fixtures(config: ModelConfig, results: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    config.validate()
    train, upcoming = build_features(results, fixtures, window=config.form_window)
    model = fit_model(config, train)
    out = predict_rows(config, model, upcoming)
    out.insert(0, "model_id", config.id)
    out.insert(1, "model_name", config.name)
    return out


def consensus(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average every model's probabilities and expected goals per fixture."""
    keys = ["Div", "Date", "Time", "HomeTeam", "AwayTeam"]
    grouped = predictions.groupby(keys, sort=False, dropna=False)
    agg = grouped[["p_h", "p_d", "p_a", "xg_h", "xg_a", "odds_h", "odds_d", "odds_a",
                   "h_played", "a_played"]].mean().reset_index()
    probs = agg[["p_h", "p_d", "p_a"]].to_numpy()
    goals = agg[["xg_h", "xg_a"]].to_numpy()
    out = attach_predictions(agg, probs, goals)
    out.insert(0, "model_id", "consensus")
    out.insert(1, "model_name", "Consensus (average)")
    return out
