import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ============================================================
# 1. WCZYTANIE DANYCH
# ============================================================
df = pd.read_csv("housing.csv")

# ============================================================
# 2. DEFINICJA ZADANIA
# ============================================================
X_reg = df.drop(columns=["median_house_value"])
y_reg = df["median_house_value"]

num_cols = X_reg.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_reg.select_dtypes(exclude=[np.number]).columns.tolist()

# ============================================================
# 3. PREPROCESSING
# ============================================================
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ],
    remainder="drop"
)

# ============================================================
# 4. WALIDACJA KRZYŻOWA
# ============================================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 5. PARAMETRY BAZOWE (aktualizowane po każdym parametrze)
# ============================================================
current_params = {
    "n_neighbors": 5,
    "weights": "uniform",
    "metric": "minkowski",
    "p": 2,
    "n_jobs": 1
}

# ============================================================
# 6. REJESTR WYNIKÓW
# ============================================================
all_results = []

# ============================================================
# 7. FUNKCJA OCENY + AKTUALIZACJA
# ============================================================
def evaluate_and_update(param_name, values):
    global current_params

    print(f"\n{'='*60}")
    print(f"  OPTYMALIZACJA PARAMETRU: {param_name}")
    print(f"  Aktualne parametry: {current_params}")
    print(f"{'='*60}")

    best_val   = None
    best_score = -np.inf

    for val in values:
        params = current_params.copy()
        params[param_name] = val

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", KNeighborsRegressor(**params))
        ])

        scores = cross_validate(
            pipe, X_reg, y_reg, cv=cv,
            scoring={
                "r2":       "r2",
                "neg_mae":  "neg_mean_absolute_error",
                "neg_rmse": "neg_root_mean_squared_error"
            },
            n_jobs=1
        )

        r2   = scores["test_r2"].mean()
        mae  = -scores["test_neg_mae"].mean()
        rmse = -scores["test_neg_rmse"].mean()

        result = {
            "parametr":        param_name,
            "wartosc":         str(val),
            "r2_mean":         round(r2,   4),
            "r2_std":          round(scores["test_r2"].std(), 4),
            "mae_mean":        round(mae,  2),
            "mae_std":         round(scores["test_neg_mae"].std(), 2),
            "rmse_mean":       round(rmse, 2),
            "rmse_std":        round(scores["test_neg_rmse"].std(), 2),
            "params_snapshot": str(params),
        }

        all_results.append(result)

        marker = ""
        if r2 > best_score:
            best_score = r2
            best_val   = val
            marker     = "  ◄ NOWE NAJLEPSZE"

        print(
            f"  {param_name}={str(val):>12} | "
            f"R2={r2:.4f} ± {scores['test_r2'].std():.4f} | "
            f"MAE={mae:.0f} | "
            f"RMSE={rmse:.0f}"
            f"{marker}"
        )

    # --- AKTUALIZACJA PARAMETRÓW BAZOWYCH ---
    current_params[param_name] = best_val

    print(f"\n  ✔ Najlepsza wartość dla '{param_name}': {best_val}  (R2={best_score:.4f})")
    print(f"  Zaktualizowane parametry: {current_params}")

# ============================================================
# 8. SEKWENCYJNA GREEDY OPTYMALIZACJA
# ============================================================
evaluate_and_update("n_neighbors", [1, 3, 5, 10, 20, 50])
evaluate_and_update("weights",     ["uniform", "distance"])
evaluate_and_update("metric",      ["euclidean", "manhattan", "chebyshev", "minkowski"])

# ============================================================
# 9. ZAPIS WYNIKÓW
# ============================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv("wyniki_knn_regresja_greedy.csv", index=False)
print("\n\nZapisano wyniki do: wyniki_knn_regresja_greedy.csv")

print("\n" + "="*60)
print("FINALNE PARAMETRY PO GREEDY OPTYMALIZACJI")
print(f"  {current_params}")
print("="*60)