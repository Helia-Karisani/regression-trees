# Regression Trees (Decision Tree Regressor) — NYC Taxi Tip Prediction

This project trains a **regression tree** using Scikit-Learn’s `DecisionTreeRegressor` to predict **tip amount** (`tip_amount`) from a subset of the publicly available **NYC Taxi & Limousine Commission (TLC)** trip dataset.

I will:
- preprocess real-world trip data,
- train a regression tree model,
- run inference (predictions),
- and evaluate model quality using regression metrics.

---

## Project Purpose

Given trip-level features (inputs) from NYC taxi rides, predict the **amount of tip paid**. This is a **supervised regression** problem:

- Input: a vector of trip features `x`
- Output: a real number `y = tip_amount`

---

## Dataset

- Source: subset of the NYC TLC dataset (as used in the lab).
- Target (label): `tip_amount`
- Features: all other columns besides `tip_amount`

The notebook computes correlations between numeric features and `tip_amount` to provide a quick sanity check on which variables may be informative.

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

## Workflow Summary

1. Load the TLC subset dataset.
2. Separate:
   - `X` = features (all columns except `tip_amount`)
   - `y` = target (`tip_amount`)
3. Normalize features (L1 normalization).
4. Split into train/test.
5. Train a regression tree:
   - `DecisionTreeRegressor(criterion="squared_error", max_depth=8)`
6. Evaluate using MSE and R².
7. Compare with a deeper tree (e.g., `max_depth=12`) to observe overfitting.

---

## Data Preprocessing

### Feature/Label split

- `y = tip_amount`
- `X = dataset.drop("tip_amount")`

### L1 normalization (row-wise)

Each example `x` is normalized so the sum of absolute feature values is 1:

`x_normalized = x / sum_k |x_k|`

This reduces the impact of scale differences across rows and makes learning more stable when features have different magnitudes.

---

## Model: Regression Tree (Decision Tree Regressor)

A regression tree recursively splits the feature space into regions. Each region corresponds to a **leaf node**, and the prediction for any point in that leaf is a constant.

At one node, training does this:

- For each feature j
- Try many candidate thresholds s (typically thresholds between sorted unique values of that feature among the samples that reached the node)
- Compute the split score, e.g.
   - SSE(split) = SSE(left) + SSE(right)
   - and pick the (j, s) with minimum SSE.

### Prediction rule at a leaf

If a point falls into leaf `L`, the predicted value is the mean label value of training samples in that leaf:

`y_hat(L) = (1/|L|) * sum_{i in L} y_i`

### How the tree chooses splits

At any internal node, the tree tries candidate splits of the form:

- choose feature index `j`
- choose threshold `s`
- send points left if `x_j <= s`, right otherwise

For each split, compute the sum of squared errors (SSE) within the two children:

`SSE(split) = sum_{i in left} (y_i - y_left_mean)^2 + sum_{i in right} (y_i - y_right_mean)^2`

The algorithm picks the split that **minimizes** `SSE(split)` (equivalently, maximizes reduction in variance / squared error).

### Hyperparameters used

- `criterion = "squared_error"` (standard regression-tree objective)
- `max_depth = 8` (main model)
- `max_depth = 12` (deeper comparison model)

Depth controls complexity:
- small depth: simpler model, less overfitting, may underfit
- large depth: more complex model, can overfit training data

---

## Evaluation Metrics

### Mean Squared Error (MSE)

`MSE = (1/n) * sum_{i=1..n} (y_i - y_hat_i)^2`

Lower is better.

### R² score

`R2 = 1 - (sum_i (y_i - y_hat_i)^2) / (sum_i (y_i - y_mean)^2)`

- `R2 = 1`: perfect predictions
- `R2 = 0`: same as predicting the mean
- `R2 < 0`: worse than predicting the mean

---

## How to Run

### Option A — Run the notebook

1. Clone the repo:
   `git clone <YOUR_REPO_URL>`
2. Open `regression-trees.ipynb`
3. Run cells top-to-bottom

### Option B — Run locally (recommended setup)

Create and activate a virtual environment:

**Windows (PowerShell)**
`python -m venv .venv`
`.venv\Scripts\Activate.ps1`

**macOS / Linux**
`python3 -m venv .venv`
`source .venv/bin/activate`

Install dependencies:

`pip install -U pip`
`pip install numpy pandas matplotlib scikit-learn notebook`

Launch Jupyter:

`jupyter notebook`

Open `regression-trees.ipynb` and run all cells.

---

## Repo Structure

- `regression-trees.ipynb` — main notebook
- `README.md` — project overview (this file)

---

## Expected Outputs

Running the notebook should produce:
- a correlation view of features vs `tip_amount`
- a trained `DecisionTreeRegressor`
- predictions on the test set
- printed evaluation metrics (MSE and R²)
- a comparison showing how increasing `max_depth` affects performance

---

## Notes / Limitations

- A single decision tree is interpretable but can overfit.
- Better performance often comes from ensembles:
  - Random Forest Regressor
  - Gradient Boosting / XGBoost
- Feature engineering (and handling categorical fields properly) can improve results substantially.

---

## License / Attribution

Dataset credited to NYC Taxi & Limousine Commission (TLC). This project is a learning exercise built on a publicly available dataset subset.

