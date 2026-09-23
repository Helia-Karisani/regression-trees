# Regression Trees: NYC Taxi Tip Prediction

This project trains a **regression tree** with Scikit-Learn's `DecisionTreeRegressor` to predict **tip amount** (`tip_amount`) from a subset of the public **NYC Taxi & Limousine Commission (TLC)** trip dataset.

It is a supervised regression problem: the input is a vector of trip features `x`, and the output is a real number `y = tip_amount`.

---

## Dataset

- Source: subset of the NYC TLC dataset
- Target: `tip_amount`
- Features: all other columns

The notebook also computes correlations between the numeric features and `tip_amount` as a quick check of which variables may be useful.

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

## Workflow

1. Load the TLC subset.
2. Split into `X` (all columns except `tip_amount`) and `y` (`tip_amount`).
3. Normalize features (L1, row-wise).
4. Split into train/test.
5. Train `DecisionTreeRegressor(criterion="squared_error", max_depth=8)`.
6. Evaluate with MSE and R².
7. Compare with a deeper tree (`max_depth=12`) to see overfitting.

---

## L1 Normalization

Each row is scaled so the sum of absolute feature values is 1:

`x_normalized = x / sum_k |x_k|`

---

## Regression Tree

A regression tree splits the feature space into regions. Each region is a leaf, and the prediction for any point in a leaf is the mean label of the training samples in that leaf:

`y_hat(L) = (1/|L|) * sum_{i in L} y_i`

At each node, the tree tries splits of the form `x_j <= s` for every feature `j` and candidate threshold `s`, and picks the split with the lowest sum of squared errors:

`SSE(split) = sum_{i in left} (y_i - y_left_mean)^2 + sum_{i in right} (y_i - y_right_mean)^2`

### Hyperparameters

- `criterion = "squared_error"`
- `max_depth = 8` (main model)
- `max_depth = 12` (deeper comparison)

A small depth gives a simpler model that may underfit, and a large depth can overfit the training data.

---

## Evaluation Metrics

**Mean Squared Error:**

`MSE = (1/n) * sum_i (y_i - y_hat_i)^2`

**R² score:**

`R2 = 1 - sum_i (y_i - y_hat_i)^2 / sum_i (y_i - y_mean)^2`

- `R2 = 1`: perfect predictions
- `R2 = 0`: same as predicting the mean
- `R2 < 0`: worse than predicting the mean

---

## How to Run

```bash
git clone https://github.com/Helia-Karisani/regression-trees.git
cd regression-trees
pip install numpy pandas matplotlib scikit-learn notebook
jupyter notebook regression-trees.ipynb
```

Then run all cells.

---

## Limitations

- A single decision tree is interpretable but can overfit.
- Ensembles like Random Forest or XGBoost usually perform better.
- Better feature engineering and handling of categorical fields could improve results.

---

## Attribution

Dataset from the NYC Taxi & Limousine Commission (TLC).
