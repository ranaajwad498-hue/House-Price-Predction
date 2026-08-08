# House Price Prediction Pipeline

An end-to-end pipeline that loads a messy real-estate dataset, cleans it, engineers features, trains a **Linear Regression** model to predict house prices, evaluates it, and saves the trained model for reuse.

## What it does

The `PricePrediction` class wraps a full ML workflow in a single object:

1. **`load_data()`** — reads the raw CSV, prints a preview and `.info()` summary.
2. **`cleaning_data()`**
   - Drops duplicate rows.
   - Fills missing numeric values with the column median, and missing categorical values with the column mode.
   - Standardizes inconsistent category labels (e.g. `"SUBURBS"` / `"downtown"` → `"Suburbs"` / `"Downtown"`; `"FAIR"` / `"good"` → `"Fair"` / `"Good"`; `Y`/`N`/`TRUE`/`FALSE` → `Yes`/`No`).
   - Rounds specific half-value bathroom counts (e.g. `1.5` → `2`).
   - Extracts numeric values out of a messy `Area_SqFt` column (e.g. strings with units) and fills any leftover missing values with the median.
   - Clips `Bedrooms` to `[0, 5]` and `Price` to `[0, 900000]` to control outliers.
   - Drops the `Id` column.
   - Saves the cleaned dataset to `Cleaned Data.csv`.
3. **`split_data()`** — splits into train/test sets (80/20, `random_state=42`).
4. **`fe()`** — builds a `ColumnTransformer` that:
   - Scales numerical columns (`Area_SqFt`, `Bedrooms`, `Bathrooms`, `Floors`, `Year_Built`, `Garage_Capacity`, `Distance_City_Center_Miles`) with `StandardScaler`.
   - Ordinally encodes `Condition` using the order `Poor < Fair < Good < Very Good < Excellent`.
   - One-hot encodes `Location` and `Has_Pool`.
5. **`create_model()`** — wraps the preprocessor and a `LinearRegression` regressor into a single `sklearn.pipeline.Pipeline`.
6. **`train_model()`** — fits the pipeline on the training data.
7. **`prediction()`** — predicts prices on the test set and prints predictions vs. actuals.
8. **`evaluation()`** — reports MAE, MSE, RMSE, and R².
9. **`save()`** — serializes the trained pipeline to `Price Predictor Model.pkl` via `joblib`.
10. **`predict_price()`** — runs a single example house through the trained pipeline and prints its predicted price.

`pipeline()` runs all of the above steps in order, end to end.

## Requirements

```
pandas
matplotlib
scikit-learn
joblib
```

Install with:

```bash
pip install pandas matplotlib scikit-learn joblib
```

## Expected input

A CSV file (default: `house_prices_messy_dataset.csv`) containing at least the following columns:

| Column | Type | Notes |
|---|---|---|
| `Id` | any | dropped during cleaning |
| `Area_SqFt` | numeric-ish | may contain non-numeric characters/units |
| `Bedrooms` | numeric | clipped to 0–5 |
| `Bathrooms` | numeric | half-values (1.5, 2.5, 3.5, 4.5) rounded up |
| `Floors` | numeric | |
| `Year_Built` | numeric | |
| `Location` | categorical | inconsistent casing normalized |
| `Condition` | categorical | ordinal: Poor–Excellent |
| `Garage_Capacity` | numeric | |
| `Has_Pool` | categorical | normalized to Yes/No |
| `Distance_City_Center_Miles` | numeric | |
| `Price` | numeric | target variable, clipped to 0–900,000 |

## Output

Running the script produces:

- `Cleaned Data.csv` — the cleaned dataset.
- `Price Predictor Model.pkl` — the trained `sklearn` pipeline (preprocessing + `LinearRegression`), loadable with `joblib.load(...)`.
- Console output: dataset preview/info, null-value checks, train/test shapes, dtypes, predicted vs. actual prices, evaluation metrics, and a sample prediction.

## Usage

```bash
python price_prediction.py
```

Or import and use programmatically:

```python
from price_prediction import PricePrediction

predictor = PricePrediction("house_prices_messy_dataset.csv")
predictor.pipeline()
```

To reuse the saved model later without retraining:

```python
import joblib
import pandas as pd

model = joblib.load("Price Predictor Model.pkl")
new_house = pd.DataFrame([{
    "Area_SqFt": 700, "Bedrooms": 3, "Bathrooms": 4, "Floors": 2,
    "Year_Built": 2006, "Location": "Rural", "Condition": "Good",
    "Garage_Capacity": 1, "Has_Pool": "Yes", "Distance_City_Center_Miles": 10
}])
print(model.predict(new_house))
```

## Known issues / things to review

- **`Bathrooms` "rounding" is a fixed lookup, not general rounding** — `replace([1.5, 2.5, 3.5, 4.5], [2, 3, 4, 5])` only handles those four exact values. Any other half-value (e.g. `5.5`, `0.5`) will pass through unchanged.
- **Outlier clipping vs. removal** — `Bedrooms` and `Price` are clipped rather than treated as outliers/removed, which silently caps extreme values instead of flagging or investigating them; confirm this is the intended strategy.
- **No train/test leakage check on cleaning** — cleaning (median/mode imputation, clipping) happens on the *full* dataset before the train/test split, so statistics like the median used to fill nulls are computed using test-set rows too. For a strict evaluation, this should happen after splitting (fit on train, apply to test).
- **Model choice** — a plain `LinearRegression` is used with no regularization, cross-validation, or hyperparameter search; consider `Ridge`/`Lasso`/tree-based models and cross-validation for a more robust benchmark.
- **No output directory handling** — `Cleaned Data.csv` and `Price Predictor Model.pkl` are written to the current working directory; consider parameterizing the output path.
- **Unused import** — `matplotlib.pyplot` is imported but never used in the current code.
- **Silent column assumptions** — `cleaning_data()` assumes specific columns (`Location`, `Condition`, `Has_Pool`, `Area_SqFt`, `Bedrooms`, `Price`, `Id`) exist; missing columns will raise a `KeyError` with no custom error message.

## Suggested improvements

- Move cleaning/imputation logic to fit only on the training split, then transform both train and test to avoid leakage.
- Replace the hardcoded bathroom value mapping with generic rounding logic (e.g. `np.ceil`).
- Add cross-validation and compare against at least one regularized or non-linear model.
- Add logging instead of `print()` for production use.
- Parameterize file paths (input dataset, cleaned data output, model output) via constructor arguments or a config file.