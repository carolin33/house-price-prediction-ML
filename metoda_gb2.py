import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. WCZYTANIE DANYCH
# ============================================================
df = pd.read_csv("housing.csv")

# ============================================================
# 2. DEFINICJA ZADANIA - REGRESJA
# ============================================================
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {len(X_train):,} próbek | Test: {len(X_test):,} próbek")

# ============================================================
# 4. PREPROCESSING
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ],
    remainder="drop"
)

# ============================================================
# 5. WALIDACJA KRZYŻOWA
# ============================================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

current_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 1.0,
    "random_state": 42
}

all_results = []

def evaluate_and_update(param_name, values):
    global current_params

    print(f"\n{'='*60}")
    print(f"OPTYMALIZACJA: {param_name}")
    print(f"{'='*60}")

    best_val = None
    best_score = -np.inf

    for val in values:
        params = current_params.copy()
        params[param_name] = val

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", GradientBoostingRegressor(**params))
        ])

        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "r2": "r2",
                "neg_mae": "neg_mean_absolute_error",
                "neg_rmse": "neg_root_mean_squared_error"
            },
            n_jobs=-1
        )

        r2 = scores["test_r2"].mean()
        mae = -scores["test_neg_mae"].mean()
        rmse = -scores["test_neg_rmse"].mean()

        result = {
            "parametr": param_name,
            "wartosc": str(val),
            "r2_mean": round(r2, 4),
            "r2_std": round(scores["test_r2"].std(), 4),
            "mae_mean": round(mae, 2),
            "mae_std": round(scores["test_neg_mae"].std(), 2),
            "rmse_mean": round(rmse, 2),
            "rmse_std": round(scores["test_neg_rmse"].std(), 2),
            "params_snapshot": str(params)
        }

        all_results.append(result)

        marker = ""
        if r2 > best_score:
            best_score = r2
            best_val = val
            marker = "  <-- NOWE NAJLEPSZE"

        print(
            f"{param_name}={val} | "
            f"R2={r2:.4f} ± {scores['test_r2'].std():.4f} | "
            f"MAE={mae:.0f} | RMSE={rmse:.0f}{marker}"
        )

    current_params[param_name] = best_val
    print(f"\nNajlepsza wartość: {best_val}")
    print(f"Zaktualizowane parametry: {current_params}")

# ============================================================
# 6. GREEDY OPTYMALIZACJA
# ============================================================
evaluate_and_update("n_estimators", [50, 100, 200, 300])
evaluate_and_update("learning_rate", [0.01, 0.05, 0.1, 0.2])
evaluate_and_update("max_depth", [2, 3, 4, 5])
evaluate_and_update("subsample", [0.6, 0.8, 0.9, 1.0])

results_df = pd.DataFrame(all_results)
results_df.to_csv("wyniki_gradient_boosting_regresja.csv", index=False)

print("\nFINALNE PARAMETRY GRADIENT BOOSTING:")
print(current_params)

# ============================================================
# 7. FINALNY MODEL
# ============================================================
final_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingRegressor(**current_params))
])

final_pipe.fit(X_train, y_train)

y_pred_train = final_pipe.predict(X_train)
y_pred_test = final_pipe.predict(X_test)

metrics_df = pd.DataFrame({
    "split": ["Train", "Test"],
    "R2": [
        r2_score(y_train, y_pred_train),
        r2_score(y_test, y_pred_test)
    ],
    "MAE": [
        mean_absolute_error(y_train, y_pred_train),
        mean_absolute_error(y_test, y_pred_test)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_train, y_pred_train)),
        np.sqrt(mean_squared_error(y_test, y_pred_test))
    ]
})

metrics_df.to_csv("metryki_train_test_gradient_boosting.csv", index=False)

print("\nMetryki Train vs Test:")
print(metrics_df.to_string(index=False))

# ============================================================
# 8. WYKRESY
# ============================================================

# Wykres 1: greedy optymalizacja
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, param in zip(axes, ["n_estimators", "learning_rate", "max_depth", "subsample"]):
    sub = results_df[results_df["parametr"] == param].copy()
    x = range(len(sub))

    ax.errorbar(
        x,
        sub["r2_mean"],
        yerr=sub["r2_std"],
        fmt="o-",
        capsize=4
    )

    best_idx = sub["r2_mean"].values.argmax()
    ax.scatter(best_idx, sub["r2_mean"].iloc[best_idx], s=80)

    ax.set_title(f"Parametr: {param}")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["wartosc"], rotation=20)
    ax.set_ylabel("R²")
    ax.set_xlabel("Wartość parametru")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gb_wykres_greedy.png", bbox_inches="tight")
plt.close()

# Wykres 2: Train vs Test
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

for ax, metric in zip(axes, ["R2", "MAE", "RMSE"]):
    ax.bar(metrics_df["split"], metrics_df[metric])
    ax.set_title(metric)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gb_wykres_train_test.png", bbox_inches="tight")
plt.close()

print("\nZapisano wyniki i wykresy dla Gradient Boosting.")