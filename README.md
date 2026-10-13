# weekend_football_predictor
Modeling historical data to predict football results.

A Windows desktop app (Python + Kivy) that imports football-data.co.uk style CSV files,
trains several machine-learning models on the results and predicts upcoming fixtures,
with a predicted result, scoreline and confidence score for each match. Models can be
renamed and customised, and their accuracy is measured with walk-forward backtests and
by tracking saved predictions against real results.

## Setup

Kivy needs Python 3.10–3.13 (there are no Kivy wheels for 3.14 yet).

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Then start the app with `run.bat`, or `.venv\Scripts\python main.py`.

## Using it

1. **Data** – import results files (e.g. `E0.csv`) and a fixtures file (`fixtures.csv`)
   using *Import CSV files...* or drag and drop. Played matches are added to the history;
   rows without a score become the fixtures to predict. Re-importing is safe: duplicates
   are replaced.
2. **Models** – six models are created to start with. Rename them, change the algorithm,
   choose which feature groups they use (form, xG & shots, home/away form, Elo, bookmaker
   odds), set the form window, limit training to certain divisions, and tune the algorithm
   settings. *New* and *Duplicate* create more.
3. **Predictions** – pick a model (or all models plus a consensus) and press
   *Predict fixtures*. Each fixture gets a pick (home/draw/away), the most likely
   scoreline, a confidence score, the H/D/A probabilities, expected goals and the value
   against the bookmakers' odds. *Save to tracker* keeps the predictions so they can be
   scored once the results are imported.
4. **Analysis** – *Backtest all models* replays the season: each week every model is
   trained only on earlier matches and predicts that week. The comparison table shows
   accuracy, log loss, Brier score, ranked probability score, exact-score hit rate and
   betting ROI; the detail view breaks accuracy down by confidence band, outcome,
   division and calibration. *Export report* writes a text report and CSV.

Data, models and reports are stored in `%LOCALAPPDATA%\WeekendFootballPredictor`
(set `WFP_DATA_DIR` to use another folder).

## Algorithms

| Algorithm | What it does |
|---|---|
| Bookmaker odds baseline | Uses the bookmakers' implied probabilities – the benchmark to beat |
| Logistic regression | Multinomial logistic regression for the result, Poisson regression for goals |
| Random forest | Tree ensembles for the result and each side's goals |
| Gradient boosting | Histogram gradient boosting; handles missing stats natively |
| Poisson goals model | Predicts each side's goals; result probabilities come from the scoreline grid (Dixon-Coles adjusted) |
| Blended ensemble | Averages any combination of the above |

## Tests

```
.venv\Scripts\python -m pytest
```
