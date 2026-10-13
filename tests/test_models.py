import numpy as np
import pandas as pd
import pytest

from football_predictor.evaluation import attach_results, backtest, evaluate, format_report
from football_predictor.features import ELO_START, build_features, feature_columns
from football_predictor.models import (ALGORITHMS, InsufficientDataError, ModelConfig, consensus,
                                       likely_score, outcome_probabilities, predict_fixtures, score_matrix)


def _fixtures(results):
    last = results["Date"].max() + pd.Timedelta(days=7)
    return pd.DataFrame({"Div": ["T1", "T1"], "Date": [last, last], "Time": ["15:00", "15:00"],
                         "HomeTeam": ["Team H", "Team A"], "AwayTeam": ["Team A", "Unknown FC"],
                         "AvgH": [1.5, 2.5], "AvgD": [4.0, 3.3], "AvgA": [6.0, 2.9]})


def test_features_use_only_earlier_matches(results):
    train, _ = build_features(results)
    first_round = train.iloc[:4]
    assert (first_round["h_played"] == 0).all()
    assert first_round["h_pts"].isna().all()
    assert (first_round["elo_h"] == ELO_START).all()
    # Changing a later score must not change earlier features.
    altered = results.copy()
    altered.loc[altered.index[-1], "FTHG"] = 9
    train2, _ = build_features(altered)
    cols = feature_columns(["form", "elo"])
    pd.testing.assert_frame_equal(train[cols].iloc[:-1], train2[cols].iloc[:-1])


def test_score_matrix_and_likely_score():
    grid = score_matrix(1.6, 0.9, rho=-0.05)
    assert grid.sum() == pytest.approx(1.0)
    probs = outcome_probabilities(grid)
    assert probs.sum() == pytest.approx(1.0) and probs[0] > probs[2]
    h, a = likely_score(1.6, 0.9, "A")
    assert a > h


@pytest.mark.parametrize("algorithm", list(ALGORITHMS))
def test_every_algorithm_predicts_fixtures(results, algorithm):
    config = ModelConfig(f"Test {algorithm}", algorithm, feature_groups=["form", "xg", "venue", "elo", "market"])
    out = predict_fixtures(config, results, _fixtures(results))
    assert len(out) == 2
    probs = out[["p_h", "p_d", "p_a"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1)
    assert set(out["prediction"]) <= {"H", "D", "A"}
    assert ((out["confidence"] >= 33) & (out["confidence"] <= 100)).all()
    # The predicted scoreline agrees with the predicted result.
    for r in out.itertuples():
        expected = "H" if r.score_h > r.score_a else ("A" if r.score_h < r.score_a else "D")
        assert expected == r.prediction


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(" ", "logistic").validate()
    with pytest.raises(ValueError):
        ModelConfig("x", "logistic", feature_groups=[]).validate()
    with pytest.raises(ValueError):
        ModelConfig("x", "logistic", params={"C": -1}).validate()
    config = ModelConfig("x", "random_forest", params={"n_estimators": "50"}).validate()
    assert config.params["n_estimators"] == 50 and "max_depth" in config.params


def test_too_little_data(results):
    config = ModelConfig("x", "logistic")
    with pytest.raises(InsufficientDataError):
        predict_fixtures(config, results.head(10), _fixtures(results))


def test_backtest_and_metrics(results):
    config = ModelConfig("Poisson", "poisson", feature_groups=["form", "elo"])
    preds = backtest(config, results, min_train=40, step_days=7)
    assert len(preds) == len(results) - 40
    metrics = evaluate(preds)
    assert metrics["matches"] == len(preds)
    assert 0 <= metrics["accuracy"] <= 1
    assert sum(sum(row.values()) for row in metrics["confusion"].values()) == len(preds)
    assert sum(b["matches"] for b in metrics["by_confidence"]) == len(preds)
    assert "Result accuracy" in format_report("Poisson", "", metrics)


def test_consensus_and_tracking(results):
    fixtures = _fixtures(results)
    a = predict_fixtures(ModelConfig("A", "logistic"), results, fixtures)
    b = predict_fixtures(ModelConfig("B", "poisson"), results, fixtures)
    combined = consensus(pd.concat([a, b]))
    assert len(combined) == 2
    assert np.allclose(combined["p_h"], (a["p_h"].to_numpy() + b["p_h"].to_numpy()) / 2)

    played = fixtures.iloc[:1].assign(FTHG=2, FTAG=0, FTR="H")
    settled = attach_results(a, played)
    metrics = evaluate(settled)
    assert metrics["matches"] == 1
