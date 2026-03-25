import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ============================================================
# 1. WCZYTANIE DANYCH
# ============================================================
df = pd.read_csv("housing.csv")

# ============================================================
# 2. DEFINICJA ZADAŃ
# ============================================================

# ---------- KLASYFIKACJA: przewidywanie ocean_proximity ----------
# Celowo usuwamy longitude i latitude, aby model nie rozwiązywał
# zadania bezpośrednio po współrzędnych geograficznych.
df_clf = df.copy()
X_clf = df_clf.drop(columns=["ocean_proximity", "longitude", "latitude"])
y_clf_raw = df_clf["ocean_proximity"]

le = LabelEncoder()
y_clf = le.fit_transform(y_clf_raw)

# ---------- REGRESJA: przewidywanie median_house_value ----------
df_reg = df.copy()
X_reg = df_reg.drop(columns=["median_house_value"])
y_reg = df_reg["median_house_value"]

# Kolumny do preprocessingu
num_cols_clf = X_clf.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_clf = X_clf.select_dtypes(exclude=[np.number]).columns.tolist()

num_cols_reg = X_reg.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_reg = X_reg.select_dtypes(exclude=[np.number]).columns.tolist()

# ============================================================
# 3. PREPROCESSING
# ============================================================

# Dla klasyfikacji: po usunięciu ocean_proximity, longitude i latitude
# .
preprocessor_clf = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols_clf),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols_clf)
    ],
    remainder="drop"
)

# Dla regresji: ocean_proximity zostaje jako cecha kategoryczna
preprocessor_reg = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols_reg),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols_reg)
    ],
    remainder="drop"
)

# ============================================================
# 4. WALIDACJA KRZYŻOWA
# ============================================================
cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 5. PARAMETRY BAZOWE
# ============================================================
BASE_PARAMS_CLF = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "max_features": "sqrt"
}

BASE_PARAMS_REG = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "max_features": 1.0
}

# ============================================================
# 6. FUNKCJA OCENY JEDNEGO PARAMETRU
# ============================================================
all_results = []

def evaluate_parameter(param_name, values):
    print(f"\n{'='*20} TEST PARAMETRU: {param_name} {'='*20}")

    for val in values:
        params_clf = BASE_PARAMS_CLF.copy()
        params_reg = BASE_PARAMS_REG.copy()

        params_clf[param_name] = val
        params_reg[param_name] = val

        clf_pipe = Pipeline([
            ("preprocess", preprocessor_clf),
            ("model", RandomForestClassifier(
                **params_clf,
                random_state=42,
                n_jobs=-1
            ))
        ])

        reg_pipe = Pipeline([
            ("preprocess", preprocessor_reg),
            ("model", RandomForestRegressor(
                **params_reg,
                random_state=42,
                n_jobs=-1
            ))
        ])

        clf_scores = cross_validate(
            clf_pipe,
            X_clf,
            y_clf,
            cv=cv_clf,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "f1_macro": "f1_macro"
            },
            n_jobs=-1
        )

        reg_scores = cross_validate(
            reg_pipe,
            X_reg,
            y_reg,
            cv=cv_reg,
            scoring={
                "r2": "r2",
                "neg_mae": "neg_mean_absolute_error",
                "neg_rmse": "neg_root_mean_squared_error"
            },
            n_jobs=-1
        )

        result = {
            "model": "RandomForest",
            "parametr": param_name,
            "wartosc": str(val),

            "clf_accuracy_mean": round(clf_scores["test_accuracy"].mean(), 4),
            "clf_accuracy_std": round(clf_scores["test_accuracy"].std(), 4),

            "clf_balanced_accuracy_mean": round(clf_scores["test_balanced_accuracy"].mean(), 4),
            "clf_balanced_accuracy_std": round(clf_scores["test_balanced_accuracy"].std(), 4),

            "clf_f1_macro_mean": round(clf_scores["test_f1_macro"].mean(), 4),
            "clf_f1_macro_std": round(clf_scores["test_f1_macro"].std(), 4),

            "reg_r2_mean": round(reg_scores["test_r2"].mean(), 4),
            "reg_r2_std": round(reg_scores["test_r2"].std(), 4),

            "reg_mae_mean": round(-reg_scores["test_neg_mae"].mean(), 2),
            "reg_mae_std": round(reg_scores["test_neg_mae"].std(), 2),

            "reg_rmse_mean": round(-reg_scores["test_neg_rmse"].mean(), 2),
            "reg_rmse_std": round(reg_scores["test_neg_rmse"].std(), 2)
        }

        all_results.append(result)

        print(
            f"Wartość={val} | "
            f"Acc={result['clf_accuracy_mean']:.4f} ± {result['clf_accuracy_std']:.4f} | "
            f"BalAcc={result['clf_balanced_accuracy_mean']:.4f} | "
            f"F1_macro={result['clf_f1_macro_mean']:.4f} | "
            f"R2={result['reg_r2_mean']:.4f} ± {result['reg_r2_std']:.4f} | "
            f"MAE={result['reg_mae_mean']:.2f} | "
            f"RMSE={result['reg_rmse_mean']:.2f}"
        )

# ============================================================
# 7. EKSPERYMENTY
# Każdy parametr ma >= 4 wartości
# ============================================================
evaluate_parameter("n_estimators", [50, 100, 200, 300, 500])
evaluate_parameter("max_depth", [5, 10, 15, 25, None])
evaluate_parameter("min_samples_split", [2, 5, 10, 20, 50])
evaluate_parameter("max_features", [0.3, 0.5, 0.7, 0.9, 1.0])

# ============================================================
# 8. ZAPIS WYNIKÓW
# ============================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv("wyniki_rf_poprawione.csv", index=False)
print("\nZapisano wyniki do: wyniki_rf_poprawione.csv")

# ============================================================
# 9. FEATURE IMPORTANCE (EKSPERORACYJNIE, NA CAŁYM ZBIORZE)
# ============================================================
# 
final_reg_pipe = Pipeline([
    ("preprocess", preprocessor_reg),
    ("model", RandomForestRegressor(
        **BASE_PARAMS_REG,
        random_state=42,
        n_jobs=-1
    ))
])

final_reg_pipe.fit(X_reg, y_reg)

feature_names = final_reg_pipe.named_steps["preprocess"].get_feature_names_out()
importances = final_reg_pipe.named_steps["model"].feature_importances_

fi_df = pd.DataFrame({
    "cecha": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

fi_df.to_csv("feature_importance_rf_poprawione.csv", index=False)
print("Zapisano ważność cech do: feature_importance_rf_poprawione.csv")