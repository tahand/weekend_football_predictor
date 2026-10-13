import pytest

from football_predictor.models import ModelConfig
from football_predictor.workspace import Workspace


def test_import_and_reload(tmp_path, example_files):
    ws = Workspace(tmp_path)
    summary = ws.import_files(example_files)
    assert not summary.errors
    assert summary.results_added == 538 and summary.fixtures_read == 14

    again = ws.import_files(example_files)
    assert again.results_added == 0

    reloaded = Workspace(tmp_path)
    assert len(reloaded.results) == 538 and len(reloaded.fixtures) == 14
    assert reloaded.results["Date"].dtype.kind == "M"
    assert set(reloaded.division_summary()["Div"]) == {"E0", "E1", "E2", "E3", "EC", "SC1"}


def test_bad_file_reported(tmp_path):
    bad = tmp_path / "notes.csv"
    bad.write_text("hello,world\n1,2\n")
    summary = Workspace(tmp_path / "ws").import_files([bad])
    assert summary.errors and summary.results_read == 0


def test_model_crud(tmp_path):
    ws = Workspace(tmp_path)
    assert len(ws.models) == 6  # defaults seeded
    config = ws.upsert_model(ModelConfig("My model", "poisson", params={"alpha": 2}))
    assert Workspace(tmp_path).get_model(config.id).params["alpha"] == 2

    config.name = "Renamed"
    ws.upsert_model(config)
    assert Workspace(tmp_path).get_model(config.id).name == "Renamed"

    with pytest.raises(ValueError):
        ws.upsert_model(ModelConfig("renamed", "logistic"))  # names are unique, ignoring case

    ws.delete_model(config.id)
    assert Workspace(tmp_path).get_model(config.id) is None
    assert ws.unique_name("Form Finder") == "Form Finder (2)"
