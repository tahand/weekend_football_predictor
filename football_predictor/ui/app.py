"""Kivy application: Data, Models, Predictions and Analysis screens."""

from __future__ import annotations

import copy
import os
import threading
import traceback
import uuid
from pathlib import Path

import pandas as pd
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.screenmanager import NoTransition, Screen
from kivy.uix.spinner import Spinner
from kivy.uix.togglebutton import ToggleButton

from ..evaluation import attach_results, backtest, evaluate, format_report
from ..features import FEATURE_GROUPS
from ..models import (ALGORITHMS, OUTCOME_NAMES, InsufficientDataError, ModelConfig, consensus,
                      predict_fixtures)
from ..workspace import Workspace
from . import theme
from .widgets import (ImportPopup, ListToggle, MutedLabel, confirm, message)

KV_FILE = Path(__file__).with_name("layout.kv")
ALL_MODELS = "All models + consensus"


def fmt_date(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%a %d %b %Y")


def short_date(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%d %b %y")


def pct(value, digits=1) -> str:
    return "-" if value is None or pd.isna(value) else f"{value * 100:.{digits}f}%"


def num(value, digits=3) -> str:
    return "-" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def odds_text(value) -> str:
    return "" if value is None or pd.isna(value) else f"{value:.2f}"


def confidence_colour(confidence: float):
    if confidence >= 60:
        return theme.HIGH
    if confidence >= 47:
        return theme.MEDIUM
    return None


def app() -> "PredictorApp":
    return App.get_running_app()


class RootLayout(BoxLayout):
    def show(self, name: str) -> None:
        self.ids.sm.current = name
        for key, nav in (("data", "nav_data"), ("models", "nav_models"), ("predict", "nav_predict"),
                         ("analysis", "nav_analysis")):
            self.ids[nav].state = "down" if key == name else "normal"


class BaseScreen(Screen):
    ready = BooleanProperty(False)

    def on_kv_post(self, base_widget):
        self.ready = True

    def on_enter(self, *args):
        # The first screen is entered before its KV rules have been applied.
        if self.ready:
            self.refresh()

    def refresh(self):
        pass


# ---------------------------------------------------------------- data

class DataScreen(BaseScreen):
    def refresh(self):
        ws = app().workspace
        summary = ws.division_summary()
        rows = [[r.Div, r.matches, r.teams, short_date(r.first), short_date(r.last), r.fixtures]
                for r in summary.itertuples()]
        self.ids.summary_title.text = f"Leagues - {len(ws.results)} results imported"
        self.ids.div_table.set_data(["Div", "Results", "Teams", "From", "To", "Fixtures"], rows,
                                    weights=[0.6, 0.9, 0.8, 1.2, 1.2, 0.9],
                                    empty_text="No results yet - import some CSV files")

        fixtures = ws.fixtures
        rows = []
        if not fixtures.empty:
            for f in fixtures.to_dict("records"):
                rows.append([fmt_date(f["Date"]), f.get("Time", "") if pd.notna(f.get("Time")) else "",
                             f["Div"], f["HomeTeam"], f["AwayTeam"],
                             odds_text(f.get("AvgH", f.get("B365H"))), odds_text(f.get("AvgD", f.get("B365D"))),
                             odds_text(f.get("AvgA", f.get("B365A")))])
        self.ids.fixtures_title.text = f"Fixtures to predict - {len(fixtures)}"
        self.ids.fixtures_table.set_data(["Date", "Time", "Div", "Home", "Away", "H odds", "D odds", "A odds"],
                                         rows, weights=[1.6, 0.6, 0.5, 1.6, 1.6, 0.7, 0.7, 0.7],
                                         empty_text="No fixtures - import a fixtures CSV")

    def open_import(self):
        start = app().workspace.settings.get("last_import_dir") or _default_import_dir()
        ImportPopup(start, app().import_files).open()

    def clear_fixtures(self):
        confirm("Clear fixtures", "Remove the current fixture list?",
                lambda: (app().workspace.clear_fixtures(), self.refresh()), yes_text="Clear")

    def clear_all(self):
        def do():
            app().workspace.clear_data()
            self.refresh()
            app().set_status("All imported results and fixtures removed.")
        confirm("Clear all data", "Remove every imported result and fixture? Models, backtest reports "
                                  "and saved predictions are kept.", do, yes_text="Clear everything")


def _default_import_dir() -> str:
    repo_examples = Path(__file__).resolve().parents[2] / "example_files"
    if repo_examples.is_dir():
        return str(repo_examples)
    downloads = Path.home() / "Downloads"
    return str(downloads if downloads.is_dir() else Path.home())


# ---------------------------------------------------------------- models

def _fitted_label(text: str, colour, size: str) -> Label:
    label = Label(text=text, color=colour, font_size=size, halign="left", valign="middle")
    label.bind(size=lambda w, sz: setattr(w, "text_size", sz))
    return label


class ModelsScreen(BaseScreen):
    selected_id = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._param_getters = {}
        self._feature_boxes = {}
        self._loading = False
        self._label_to_algo = {spec.label: key for key, spec in ALGORITHMS.items()}

    def refresh(self):
        ws = app().workspace
        if not ws.get_model(self.selected_id) and ws.models:
            self.selected_id = ws.models[0].id
        self._build_list()
        self.load_selected()

    def _build_list(self):
        box = self.ids.model_list
        box.clear_widgets()
        for m in app().workspace.models:
            btn = ListToggle(text=f"[b]{m.name}[/b]\n[size=12sp][color=9fa6b2]{m.algorithm_label}[/color][/size]",
                             item_id=m.id, state="down" if m.id == self.selected_id else "normal")
            btn.bind(on_release=lambda b: self.select(b.item_id))
            box.add_widget(btn)

    def select(self, model_id: str):
        self.selected_id = model_id
        for btn in self.ids.model_list.children:
            btn.state = "down" if btn.item_id == model_id else "normal"
        self.load_selected()

    @property
    def selected(self) -> ModelConfig | None:
        return app().workspace.get_model(self.selected_id)

    def load_selected(self, *_):
        config = self.selected
        self._loading = True
        try:
            ids = self.ids
            ids.algo_spinner.values = [spec.label for spec in ALGORITHMS.values()]
            if config is None:
                ids.name_input.text = ""
                ids.desc_input.text = ""
                ids.params_box.clear_widgets()
                return
            ids.name_input.text = config.name
            ids.desc_input.text = config.description
            ids.algo_spinner.text = config.algorithm_label
            ids.window_input.text = str(config.form_window)
            ids.div_input.text = ", ".join(config.divisions)
            divisions = app().workspace.divisions
            ids.div_label.text = "Train on divisions (blank = all" + (f"; have {', '.join(divisions)})" if divisions else ")")
            self._build_features(config.feature_groups)
            self._build_params(config.algorithm, config.params)
            self._show_backtest(config)
        finally:
            self._loading = False

    def _show_backtest(self, config):
        record = app().workspace.load_backtest(config.id)
        if not record:
            self.ids.last_backtest.text = "Not backtested yet. Use 'Save & backtest' or the Analysis page."
            return
        m = record["metrics"]
        stale = " (settings changed since - rerun to update)" if record.get("config_updated") != config.updated else ""
        self.ids.last_backtest.text = (
            f"Last backtest {record['run_at'].replace('T', ' ')}{stale}: "
            f"{m.get('matches', 0)} matches, accuracy {pct(m.get('accuracy'))}, "
            f"log loss {num(m.get('log_loss'))}, exact score {pct(m.get('exact_score'))}.")

    def _build_features(self, groups):
        box = self.ids.feature_box
        box.clear_widgets()
        self._feature_boxes = {}
        for key, label in FEATURE_GROUPS.items():
            row = BoxLayout(spacing=dp(4))
            cb = CheckBox(active=key in groups, size_hint_x=None, width=dp(32), color=theme.ACCENT)
            row.add_widget(cb)
            text = Label(text=label, color=theme.TEXT, font_size="14sp", halign="left", valign="middle")
            text.bind(size=lambda w, size: setattr(w, "text_size", size))
            row.add_widget(text)
            self._feature_boxes[key] = cb
            box.add_widget(row)

    def _build_params(self, algorithm, values):
        box = self.ids.params_box
        box.clear_widgets()
        self._param_getters = {}
        spec = ALGORITHMS[algorithm]
        self.ids.algo_desc.text = spec.description
        if not spec.params:
            box.add_widget(MutedLabel(text="This algorithm has no settings."))
            box.add_widget(Label())
            return
        for p in spec.params:
            value = values.get(p.name, p.default)
            label_box = BoxLayout(orientation="vertical", size_hint_y=None, height=dp(48))
            label_box.add_widget(_fitted_label(p.label, theme.TEXT, "14sp"))
            if p.help:
                label_box.add_widget(_fitted_label(p.help, theme.MUTED, "11sp"))
            box.add_widget(label_box)

            if p.kind == "choice":
                widget = Spinner(text=str(value), values=list(p.choices), size_hint_y=None, height=dp(36))
                self._param_getters[p.name] = lambda w=widget: w.text
            elif p.kind == "multi":
                widget = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(4))
                toggles = []
                for choice in p.choices:
                    t = ToggleButton(text=ALGORITHMS[choice].label if choice in ALGORITHMS else choice,
                                     state="down" if choice in value else "normal", font_size="11sp",
                                     halign="center", valign="middle")
                    t.bind(size=lambda w, s: setattr(w, "text_size", (s[0] - dp(6), None)))
                    toggles.append((choice, t))
                    widget.add_widget(t)
                self._param_getters[p.name] = lambda ts=toggles: [c for c, t in ts if t.state == "down"]
            else:
                widget = Factory.Field(text=str(value))
                self._param_getters[p.name] = lambda w=widget: w.text
            box.add_widget(widget)

    def on_algorithm_label(self, label):
        if self._loading or label not in self._label_to_algo:
            return
        config = self.selected
        algorithm = self._label_to_algo[label]
        params = config.params if config and config.algorithm == algorithm else {}
        self._build_params(algorithm, params)

    def reset_params(self):
        label = self.ids.algo_spinner.text
        if label in self._label_to_algo:
            self._build_params(self._label_to_algo[label], {})

    def _collect(self) -> ModelConfig:
        base = self.selected
        config = copy.deepcopy(base)
        config.name = self.ids.name_input.text
        config.description = self.ids.desc_input.text.strip()
        config.algorithm = self._label_to_algo.get(self.ids.algo_spinner.text, base.algorithm)
        config.params = {name: getter() for name, getter in self._param_getters.items()}
        config.feature_groups = [k for k, cb in self._feature_boxes.items() if cb.active]
        config.form_window = int(self.ids.window_input.text or 0)
        config.divisions = [d.strip() for d in self.ids.div_input.text.replace(";", ",").split(",")]
        return config

    def save_model(self) -> ModelConfig | None:
        if self.selected is None:
            return None
        try:
            config = app().workspace.upsert_model(self._collect())
        except (ValueError, TypeError) as exc:
            message("Can't save model", str(exc))
            return None
        self.refresh()
        app().set_status(f"Saved model '{config.name}'.")
        return config

    def save_and_backtest(self):
        config = self.save_model()
        if config:
            app().run_backtests([config.id], switch_to_analysis=True)

    def new_model(self):
        ws = app().workspace
        config = ModelConfig(ws.unique_name("New model"), "logistic", feature_groups=["form", "venue", "elo"],
                             description="")
        ws.upsert_model(config)
        self.selected_id = config.id
        self.refresh()

    def duplicate_model(self):
        base = self.selected
        if base is None:
            return
        ws = app().workspace
        config = copy.deepcopy(base)
        config.id = uuid.uuid4().hex[:12]
        config.name = ws.unique_name(f"{base.name} copy")
        ws.upsert_model(config)
        self.selected_id = config.id
        self.refresh()

    def delete_model(self):
        config = self.selected
        if config is None:
            return
        if len(app().workspace.models) == 1:
            message("Can't delete", "Keep at least one model.")
            return

        def do():
            app().workspace.delete_model(config.id)
            self.selected_id = ""
            self.refresh()
            app().set_status(f"Deleted model '{config.name}'.")

        confirm("Delete model", f"Delete '{config.name}' and its backtest report?", do, yes_text="Delete")


# ---------------------------------------------------------------- predictions

PRED_COLUMNS = ["Date", "Div", "Home", "Away", "Model", "Pick", "Score", "Conf.", "Home", "Draw", "Away", "xG",
                "Value"]
PRED_WEIGHTS = [1.25, 0.45, 1.3, 1.3, 1.5, 0.9, 0.55, 0.6, 0.55, 0.55, 0.55, 0.75, 0.6]
PRED_ALIGNS = ["left"] * 5 + ["left", "center"] + ["right"] * 6


class PredictScreen(BaseScreen):
    has_predictions = BooleanProperty(False)
    predictions = None

    def refresh(self):
        ws = app().workspace
        names = [m.name for m in ws.models]
        values = ([ALL_MODELS] if len(names) > 1 else []) + names
        spinner = self.ids.model_spinner
        spinner.values = values
        if spinner.text not in values:
            spinner.text = values[0] if values else ""
        self.render()

    def _selected_configs(self):
        ws = app().workspace
        if self.ids.model_spinner.text == ALL_MODELS:
            return list(ws.models)
        return [m for m in ws.models if m.name == self.ids.model_spinner.text]

    def predict(self):
        ws = app().workspace
        if ws.fixtures.empty:
            message("No fixtures", "Import a fixtures CSV on the Data page first.")
            return
        if ws.results.empty:
            message("No results", "Import some historical results on the Data page first.")
            return
        configs = [copy.deepcopy(c) for c in self._selected_configs()]
        results, fixtures = ws.results.copy(), ws.fixtures.copy()

        def work(progress):
            frames, errors = [], []
            for i, config in enumerate(configs):
                progress(i / len(configs), f"Training '{config.name}'...")
                try:
                    frames.append(predict_fixtures(config, results, fixtures))
                except (InsufficientDataError, ValueError) as exc:
                    errors.append(f"{config.name}: {exc}")
            out = pd.concat(frames, ignore_index=True) if frames else None
            if out is not None and len(frames) > 1:
                out = pd.concat([out, consensus(out)], ignore_index=True)
            return out, errors

        def done(result):
            out, errors = result
            self.predictions = out
            self.render()
            n = 0 if out is None else out[["HomeTeam", "AwayTeam", "Date"]].drop_duplicates().shape[0]
            app().set_status(f"Predicted {n} fixtures with {len(configs) - len(errors)} model(s).")
            if errors:
                message("Some models could not run", "\n".join(errors))

        app().run_job("Predicting fixtures...", work, done)

    def render(self):
        df = self.predictions
        self.has_predictions = df is not None and not df.empty
        if not self.has_predictions:
            self.ids.pred_title.text = "Predictions"
            self.ids.pred_table.set_data(PRED_COLUMNS, [], PRED_WEIGHTS, PRED_ALIGNS,
                                         empty_text="Choose a model and press 'Predict fixtures'")
            return
        order = {m.id: i for i, m in enumerate(app().workspace.models)}
        df = df.assign(_order=df["model_id"].map(order).fillna(len(order)))
        df = df.sort_values(["Date", "Time", "Div", "HomeTeam", "_order"], kind="stable")
        multi = df["model_id"].nunique() > 1
        rows, colours, bold = [], [], []
        last_fixture = None
        for r in df.to_dict("records"):
            fixture = (r["Date"], r["HomeTeam"], r["AwayTeam"])
            first = fixture != last_fixture
            last_fixture = fixture
            low_data = min(r.get("h_played", 0) or 0, r.get("a_played", 0) or 0) < 3
            pick = OUTCOME_NAMES[r["prediction"]] + (" *" if low_data else "")
            odds = {"H": r.get("odds_h"), "D": r.get("odds_d"), "A": r.get("odds_a")}[r["prediction"]]
            prob = {"H": r["p_h"], "D": r["p_d"], "A": r["p_a"]}[r["prediction"]]
            value = "" if odds is None or pd.isna(odds) else f"{(prob * odds - 1) * 100:+.0f}%"
            show_fixture = first or not multi
            rows.append([fmt_date(r["Date"]) if show_fixture else "", r["Div"] if show_fixture else "",
                         r["HomeTeam"] if show_fixture else "", r["AwayTeam"] if show_fixture else "",
                         r["model_name"], pick, f"{int(r['score_h'])}-{int(r['score_a'])}",
                         f"{r['confidence']:.0f}%", pct(r["p_h"], 0), pct(r["p_d"], 0), pct(r["p_a"], 0),
                         f"{r['xg_h']:.1f}-{r['xg_a']:.1f}", value])
            is_consensus = r["model_id"] == "consensus"
            colours.append(theme.CONSENSUS if is_consensus else confidence_colour(r["confidence"]))
            bold.append(is_consensus)
        fixtures = df[["Date", "HomeTeam", "AwayTeam"]].drop_duplicates().shape[0]
        self.ids.pred_title.text = f"Predictions - {fixtures} fixtures"
        self.ids.pred_table.set_data(PRED_COLUMNS, rows, PRED_WEIGHTS, PRED_ALIGNS, row_colors=colours,
                                     bold_rows=bold)

    def save_history(self):
        if not self.has_predictions:
            return
        n = app().workspace.save_predictions(self.predictions)
        app().set_status(f"Saved {n} predictions to the tracker. They'll be scored on the Analysis page "
                         f"once results are imported.")

    def export(self):
        if not self.has_predictions:
            return
        cols = ["model_name", "Div", "Date", "Time", "HomeTeam", "AwayTeam", "prediction", "score_h", "score_a",
                "confidence", "p_h", "p_d", "p_a", "xg_h", "xg_a", "odds_h", "odds_d", "odds_a"]
        df = self.predictions[[c for c in cols if c in self.predictions.columns]].round(4)
        path = app().workspace.export_frame(df, "predictions")
        message("Exported", f"Predictions saved to:\n{path}", folder=path.parent)


# ---------------------------------------------------------------- analysis

class AnalysisScreen(BaseScreen):
    def on_kv_post(self, base_widget):
        super().on_kv_post(base_widget)
        settings = app().workspace.settings
        self.ids.min_train.text = str(settings.get("min_train", 100))
        self.ids.step_days.text = str(settings.get("step_days", 7))

    def backtest_settings(self) -> tuple[int, int]:
        min_train = max(30, int(self.ids.min_train.text or 100))
        step = max(1, int(self.ids.step_days.text or 7))
        ws = app().workspace
        ws.set_setting("min_train", min_train)
        ws.set_setting("step_days", step)
        return min_train, step

    def backtest_all(self):
        app().run_backtests([m.id for m in app().workspace.models])

    def refresh(self):
        ws = app().workspace
        rows, colours, bars = [], [], []
        baselines = []
        for m in ws.models:
            record = ws.load_backtest(m.id)
            if not record or not record["metrics"].get("matches"):
                rows.append([m.name, m.algorithm_label, "-", "-", "-", "-", "-", "-", "-", "-", "Not run"])
                colours.append(None)
                continue
            r = record["metrics"]
            stale = record.get("config_updated") != m.updated
            rows.append([m.name, m.algorithm_label, r["matches"], pct(r["accuracy"]), num(r["log_loss"]),
                         num(r["brier"]), num(r["rps"]), pct(r["exact_score"]), pct(r.get("roi")),
                         pct(r.get("value_roi")), "Out of date" if stale else record["run_at"][:10]])
            colours.append(None)
            bars.append((m.name, r["accuracy"], r))
            baselines.append(r)
        self.ids.compare_table.set_data(
            ["Model", "Algorithm", "Matches", "Accuracy", "Log loss", "Brier", "RPS", "Exact score", "ROI",
             "Value ROI", "Run"], rows,
            weights=[1.6, 1.5, 0.7, 0.8, 0.7, 0.7, 0.7, 0.9, 0.7, 0.8, 0.9],
            aligns=["left", "left"] + ["right"] * 8 + ["left"], row_colors=colours)

        if bars:
            best = max(b[1] for b in bars)
            top = max(0.7, best)
            ordered = sorted(bars, key=lambda b: b[1], reverse=True)
            self.ids.accuracy_chart.set_bars([
                (name, acc / top, pct(acc), theme.BAR_GOOD if acc == best else theme.BAR_INFO)
                for name, acc, _ in ordered])
            ref = baselines[0]
            self.ids.compare_note.text = (
                f"For reference on the same matches: always picking the home team scores "
                f"{pct(ref['baseline_home'])}, and backing the bookmakers' favourite scores "
                f"{pct(ref.get('baseline_favourite'))}. ROI = profit per 1 unit staked on every pick at average "
                f"odds; Value ROI only backs picks the model rates at least 5% above the odds.")
        else:
            self.ids.accuracy_chart.set_bars([])
            self.ids.compare_note.text = "Press 'Backtest all models' to measure how accurate each model is."

        names = [m.name for m in ws.models]
        if self.ids.detail_spinner.text not in names:
            self.ids.detail_spinner.text = names[0] if names else ""
        self.ids.detail_spinner.values = names
        self.show_detail()
        self._render_live()

    def _detail_metrics(self):
        ws = app().workspace
        config = next((m for m in ws.models if m.name == self.ids.detail_spinner.text), None)
        if config is None:
            return None, None, "No model selected."
        if self.ids.source_spinner.text == "Backtest":
            record = ws.load_backtest(config.id)
            if not record:
                return config, None, f"'{config.name}' has not been backtested yet."
            note = f"Backtest run {record['run_at'].replace('T', ' ')}"
            if record.get("config_updated") != config.updated:
                note += " - model settings have changed since, rerun to update"
            return config, record["metrics"], note
        history = ws.history
        if history.empty:
            return config, None, "No saved predictions yet - use 'Save to tracker' on the Predictions page."
        mine = history[history["model_id"] == config.id]
        settled = attach_results(mine, ws.results)
        metrics = evaluate(settled)
        return config, metrics, (f"{len(mine)} saved predictions, {metrics.get('matches', 0)} with results so far")

    def show_detail(self, *_):
        if "detail_summary" not in self.ids:
            return
        config, m, note = self._detail_metrics()
        ids = self.ids
        if not m or not m.get("matches"):
            ids.detail_summary.text = note if not m else note + ". Nothing to score yet."
            ids.confidence_chart.set_bars([])
            for table in (ids.confusion_table, ids.outcome_table, ids.division_table, ids.calibration_table):
                table.set_data([""], [], empty_text="No data")
            return
        ids.detail_summary.text = (
            f"[color=9fa6b2]{note}[/color]\n"
            f"[b]{m['matches']}[/b] matches   accuracy [b]{pct(m['accuracy'])}[/b] "
            f"(home-only {pct(m['baseline_home'])}, favourite {pct(m.get('baseline_favourite'))})   "
            f"exact score [b]{pct(m['exact_score'])}[/b]   correct goal difference {pct(m['goal_diff_correct'])}\n"
            f"log loss {num(m['log_loss'])}   Brier {num(m['brier'])}   RPS {num(m['rps'])}   "
            f"goals error {num(m['goals_mae'], 2)} per team   average confidence {m['avg_confidence']:.1f}%\n"
            f"ROI backing every pick {pct(m.get('roi'))} ({m['bets']} bets)   "
            f"value picks {pct(m.get('value_roi'))} ({m['value_bets']} bets)")

        bars = []
        for band in m["by_confidence"]:
            acc = band["accuracy"]
            if band["matches"]:
                colour = theme.BAR_GOOD if acc >= 0.55 else (theme.BAR_MID if acc >= 0.4 else theme.BAR_BAD)
                bars.append((f"{band['band']} ({band['matches']})", acc, pct(acc), colour))
        ids.confidence_chart.set_bars(bars, label_width=0.36)

        c = m["confusion"]
        ids.confusion_table.set_data(
            ["Actual \\ Predicted", "Home", "Draw", "Away"],
            [[OUTCOME_NAMES[a], c[a]["H"], c[a]["D"], c[a]["A"]] for a in ("H", "D", "A")],
            weights=[1.6, 1, 1, 1], aligns=["left", "right", "right", "right"])
        ids.outcome_table.set_data(
            ["Outcome", "Predicted", "Happened", "Precision", "Recall"],
            [[OUTCOME_NAMES[r["outcome"]], r["predicted"], r["actual"], pct(r["precision"]), pct(r["recall"])]
             for r in m["by_outcome"]],
            weights=[1.3, 1, 1, 1, 1], aligns=["left"] + ["right"] * 4)
        ids.division_table.set_data(
            ["Division", "Matches", "Accuracy"],
            [[r["division"], r["matches"], pct(r["accuracy"])] for r in m["by_division"]],
            aligns=["left", "right", "right"])
        ids.calibration_table.set_data(
            ["Predicted probability", "Outcomes", "Average predicted", "Actually happened", "Gap"],
            [[r["bin"], r["count"], pct(r["predicted"]), pct(r["observed"]),
              f"{(r['observed'] - r['predicted']) * 100:+.1f} pts"] for r in m["calibration"]],
            aligns=["left", "right", "right", "right", "right"])

    def _render_live(self):
        ws = app().workspace
        rows = []
        if not ws.history.empty:
            settled = attach_results(ws.history, ws.results)
            for (model_id, name), grp in settled.groupby(["model_id", "model_name"], sort=False):
                m = evaluate(grp)
                rows.append([name, len(grp), m.get("matches", 0), pct(m.get("accuracy")),
                             pct(m.get("exact_score")), num(m.get("log_loss")), pct(m.get("roi"))])
        self.ids.live_table.set_data(
            ["Model", "Saved", "With results", "Accuracy", "Exact score", "Log loss", "ROI"], rows,
            weights=[1.8, 0.8, 1, 1, 1, 1, 0.8], aligns=["left"] + ["right"] * 6,
            empty_text="No saved predictions yet")

    def export_report(self):
        config, m, note = self._detail_metrics()
        ws = app().workspace
        lines = ["Model comparison (walk-forward backtest)", ""]
        for model in ws.models:
            record = ws.load_backtest(model.id)
            if record and record["metrics"].get("matches"):
                r = record["metrics"]
                lines.append(f"  {model.name:<28} acc {pct(r['accuracy']):>6}  logloss {num(r['log_loss'])}  "
                             f"rps {num(r['rps'])}  exact {pct(r['exact_score']):>6}  roi {pct(r.get('roi')):>7}")
            else:
                lines.append(f"  {model.name:<28} not backtested")
        lines.append("")
        if config is not None and m:
            title = "Backtest" if self.ids.source_spinner.text == "Backtest" else "Tracker"
            lines.append(format_report(config.name, config.description, m, title=title))
        path = ws.export_text("\n".join(lines), "model_report")
        if config is not None and self.ids.source_spinner.text == "Backtest":
            preds = ws.load_backtest_predictions(config.id)
            if not preds.empty:
                ws.export_frame(preds, f"backtest_{config.name.replace(' ', '_')}")
        message("Report exported", f"Report saved to:\n{path}", folder=path.parent)

    def clear_tracker(self):
        def do():
            app().workspace.clear_history()
            self.refresh()
        confirm("Clear tracker", "Delete every saved prediction from the tracker?", do, yes_text="Clear")


# ---------------------------------------------------------------- app

class PredictorApp(App):
    title = "Weekend Football Predictor"
    busy = BooleanProperty(False)
    status = StringProperty("Ready")
    progress = NumericProperty(0)

    def __init__(self, workspace: Workspace | None = None, **kwargs):
        super().__init__(**kwargs)
        self.workspace = workspace or Workspace()
        self._dropped: list[str] = []

    def build(self):
        Window.clearcolor = theme.BACKGROUND
        Window.bind(on_drop_file=self._on_drop_file)
        Builder.load_file(str(KV_FILE))
        root = RootLayout()
        root.ids.sm.transition = NoTransition()
        ws = self.workspace
        if ws.results.empty:
            self.status = "Welcome! Start by importing results and fixtures CSV files on the Data page."
        else:
            self.status = f"Loaded {len(ws.results)} results and {len(ws.fixtures)} fixtures from {ws.root}"
        return root

    def on_start(self):
        self.root.ids.data_screen.refresh()

    # -------------------------------------------------- helpers

    def set_status(self, text: str) -> None:
        self.status = text

    def open_folder(self, path) -> None:
        if hasattr(os, "startfile"):
            os.startfile(str(path))

    def refresh_current(self) -> None:
        self.root.ids.sm.current_screen.refresh()

    def run_job(self, label: str, work, done) -> None:
        """Run work(progress) on a background thread, then done(result) on the UI thread."""
        if self.busy:
            self.set_status("Please wait for the current task to finish.")
            return
        self.busy = True
        self.progress = 0
        self.status = label

        def progress(fraction: float, text: str) -> None:
            def apply(_dt):
                self.progress = fraction
                self.status = text
            Clock.schedule_once(apply)

        def worker():
            try:
                result = work(progress)
            except Exception as exc:  # report anything unexpected rather than crash the UI
                details = traceback.format_exc()
                Clock.schedule_once(lambda _dt, e=exc, d=details: self._job_failed(e, d))
            else:
                Clock.schedule_once(lambda _dt: self._job_done(done, result))

        threading.Thread(target=worker, daemon=True).start()

    def _job_done(self, done, result):
        self.busy = False
        self.progress = 1
        done(result)

    def _job_failed(self, exc, details):
        self.busy = False
        self.progress = 0
        self.status = f"Error: {exc}"
        print(details)
        message("Something went wrong", f"{type(exc).__name__}: {exc}")

    # -------------------------------------------------- actions

    def import_files(self, paths: list[str]) -> None:
        if not paths:
            return
        self.workspace.set_setting("last_import_dir", str(Path(paths[0]).parent))

        def done(summary):
            self.set_status(summary.describe())
            self.refresh_current()
            if summary.errors:
                message("Import finished with problems", summary.describe())

        self.run_job(f"Importing {len(paths)} file(s)...", lambda progress: self.workspace.import_files(paths), done)

    def _on_drop_file(self, _window, filename, *_args):
        path = filename.decode("utf-8") if isinstance(filename, bytes) else str(filename)
        if path.lower().endswith(".csv"):
            self._dropped.append(path)
            Clock.unschedule(self._import_dropped)
            Clock.schedule_once(self._import_dropped, 0.4)

    def _import_dropped(self, _dt):
        paths, self._dropped = self._dropped, []
        self.root.show("data")
        self.import_files(paths)

    def run_backtests(self, model_ids: list[str], switch_to_analysis: bool = False) -> None:
        ws = self.workspace
        analysis = self.root.ids.analysis_screen
        try:
            min_train, step = analysis.backtest_settings()
        except ValueError:
            message("Backtest settings", "Warm-up matches and retrain days must be whole numbers.")
            return
        if ws.results.empty:
            message("No results", "Import some historical results on the Data page first.")
            return
        configs = [copy.deepcopy(ws.get_model(i)) for i in model_ids if ws.get_model(i)]
        results = ws.results.copy()

        def work(progress):
            errors = []
            for i, config in enumerate(configs):
                def scaled(fraction, text, i=i):
                    progress((i + fraction) / len(configs), f"Backtesting {text}")
                try:
                    preds = backtest(config, results, min_train=min_train, step_days=step, progress=scaled)
                    metrics = evaluate(preds)
                    ws.save_backtest(config, preds, metrics, {"min_train": min_train, "step_days": step})
                except (InsufficientDataError, ValueError) as exc:
                    errors.append(f"{config.name}: {exc}")
            return errors

        def done(errors):
            self.set_status(f"Backtested {len(configs) - len(errors)} model(s).")
            if switch_to_analysis:
                self.root.show("analysis")
            else:
                self.refresh_current()
            if errors:
                message("Some backtests could not run", "\n".join(errors))

        self.run_job("Backtesting...", work, done)
