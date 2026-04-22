import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
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
# 4. WALIDACJA KRZYŻOWA
# ============================================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 5. PARAMETRY BAZOWE
# ============================================================
current_params = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "max_features": 1.0
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
            ("model", RandomForestRegressor(
                **params, random_state=42, n_jobs=-1
            ))
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
            "parametr":          param_name,
            "wartosc":           str(val),
            "r2_mean":           round(r2,   4),
            "r2_std":            round(scores["test_r2"].std(), 4),
            "mae_mean":          round(mae,  2),
            "mae_std":           round(scores["test_neg_mae"].std(), 2),
            "rmse_mean":         round(rmse, 2),
            "rmse_std":          round(scores["test_neg_rmse"].std(), 2),
            "params_snapshot":   str(params),
        }

        all_results.append(result)

        marker = ""
        if r2 > best_score:
            best_score = r2
            best_val   = val
            marker     = "  ◄ NOWE NAJLEPSZE"

        print(
            f"  {param_name}={str(val):>6} | "
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
evaluate_and_update("n_estimators",     [50, 100, 200, 300, 500])
evaluate_and_update("max_depth",        [5, 10, 15, 25, None])
evaluate_and_update("min_samples_split",[2, 5, 10, 20, 50])
evaluate_and_update("max_features",     [0.3, 0.5, 0.7, 0.9, 1.0])

# ============================================================
# 9. ZAPIS WYNIKÓW
# ============================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv("wyniki_rf_regresja_greedy.csv", index=False)
print("\n\nZapisano wyniki do: wyniki_rf_regresja_greedy.csv")

print("\n" + "="*60)
print("FINALNE PARAMETRY PO GREEDY OPTYMALIZACJI")
print(f"  {current_params}")
print("="*60)

# ============================================================
# 10. FEATURE IMPORTANCE NA FINALNYM MODELU
# ============================================================
final_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        **current_params, random_state=42, n_jobs=-1
    ))
])

final_pipe.fit(X_reg, y_reg)

feature_names = final_pipe.named_steps["preprocess"].get_feature_names_out()
importances   = final_pipe.named_steps["model"].feature_importances_

fi_df = pd.DataFrame({
    "cecha":      feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

fi_df.to_csv("feature_importance_rf_regresja_greedy.csv", index=False)
print("Zapisano ważność cech do: feature_importance_rf_regresja_greedy.csv")