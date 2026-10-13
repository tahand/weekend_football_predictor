import pandas as pd
import pytest

from football_predictor.data import (DataFormatError, implied_probabilities, market_odds, merge_matches,
                                     parse_dates, read_match_file, remove_played, split_results_and_fixtures)


def test_reads_example_results_and_fixtures(example_files):
    by_name = {p.name: p for p in example_files}
    results, fixtures = split_results_and_fixtures(read_match_file(by_name["E0.csv"]))
    assert len(results) == 60 and fixtures.empty
    assert results["Date"].dtype.kind == "M"
    assert set(results["FTR"]) <= {"H", "D", "A"}

    results, fixtures = split_results_and_fixtures(read_match_file(by_name["fixtures.csv"]))
    assert results.empty and len(fixtures) == 14
    assert "FTHG" not in fixtures.columns


def test_parse_dates_handles_both_year_formats():
    parsed = parse_dates(pd.Series(["13/10/2026", "01/02/26"]))
    assert list(parsed) == [pd.Timestamp("2026-10-13"), pd.Timestamp("2026-02-01")]


def test_missing_columns_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(DataFormatError):
        read_match_file(bad)


def test_merge_replaces_duplicates():
    a = pd.DataFrame({"Div": ["E0"], "Date": [pd.Timestamp("2026-08-01")], "HomeTeam": ["X"],
                      "AwayTeam": ["Y"], "FTHG": [1], "FTAG": [0]})
    b = a.assign(FTHG=[2])
    merged = merge_matches(a, b)
    assert len(merged) == 1 and merged["FTHG"].iloc[0] == 2


def test_remove_played_fixtures():
    played = pd.DataFrame({"Div": ["E0"], "Date": [pd.Timestamp("2026-08-01")], "HomeTeam": ["X"],
                           "AwayTeam": ["Y"]})
    fixtures = pd.concat([played, played.assign(HomeTeam=["Z"])], ignore_index=True)
    assert list(remove_played(fixtures, played)["HomeTeam"]) == ["Z"]


def test_implied_probabilities_remove_margin():
    df = pd.DataFrame({"AvgH": [2.0], "AvgD": [3.4], "AvgA": [3.6], "B365H": [1.5], "B365D": [4], "B365A": [6]})
    probs = implied_probabilities(market_odds(df))
    assert probs.sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert probs["mkt_ph"].iloc[0] > probs["mkt_pa"].iloc[0]
