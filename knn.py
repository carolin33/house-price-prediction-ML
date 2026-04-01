import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
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

# ---------- KLASYFIKACJA ----------
df_clf = df.copy()
X_clf = df_clf.drop(columns=["ocean_proximity", "longitude", "latitude"])
y_clf_raw = df_clf["ocean_proximity"]

le = LabelEncoder()
y_clf = le.fit_transform(y_clf_raw)

# ---------- REGRESJA ----------
df_reg = df.copy()
X_reg = df_reg.drop(columns=["median_house_value"])
y_reg = df_reg["median_house_value"]

num_cols_clf = X_clf.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_clf = X_clf.select_dtypes(exclude=[np.number]).columns.tolist()

num_cols_reg = X_reg.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_reg = X_reg.select_dtypes(exclude=[np.number]).columns.tolist()

# ============================================================
# 3. PREPROCESSING

# StandardScaler jest zawarty w Pipeline, co eliminuje ryzyko
# data leakage — scaler jest fitowany tylko na danych treningowych.
# ============================================================

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_clf = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols_clf),
        ("cat", categorical_pipe, cat_cols_clf)
    ],
    remainder="drop"
)

preprocessor_reg = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols_reg),
        ("cat", categorical_pipe, cat_cols_reg)
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
    "n_neighbors": 5,
    "weights": "uniform",
    "metric": "minkowski",
    "p": 2,
    "n_jobs": -1
}

BASE_PARAMS_REG = {
    "n_neighbors": 5,
    "weights": "uniform",
    "metric": "minkowski",
    "p": 2,
    "n_jobs": -1
}

# ============================================================
# 6. FUNKCJA OCENY
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
            ("model", KNeighborsClassifier(**params_clf))
        ])

        reg_pipe = Pipeline([
            ("preprocess", preprocessor_reg),
            ("model", KNeighborsRegressor(**params_reg))
        ])

        clf_scores = cross_validate(
            clf_pipe, X_clf, y_clf, cv=cv_clf,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "f1_macro": "f1_macro"
            },
            n_jobs=-1
        )

        reg_scores = cross_validate(
            reg_pipe, X_reg, y_reg, cv=cv_reg,
            scoring={
                "r2": "r2",
                "neg_mae": "neg_mean_absolute_error",
                "neg_rmse": "neg_root_mean_squared_error"
            },
            n_jobs=-1
        )

        result = {
            "model": "KNN",
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
            f"F1={result['clf_f1_macro_mean']:.4f} | "
            f"R2={result['reg_r2_mean']:.4f} ± {result['reg_r2_std']:.4f} | "
            f"MAE={result['reg_mae_mean']:.2f} | "
            f"RMSE={result['reg_rmse_mean']:.2f}"
        )

# ============================================================
# 7. EKSPERYMENTY
# Badane są trzy hiperparametry:
#   - n_neighbors: kluczowy parametr kontrolujący bias-variance tradeoff
#   - weights: sposób ważenia głosów sąsiadów
#   - metric: miara odległości używana do wyznaczania sąsiadów
#

# ============================================================

# 1. Liczba sąsiadów
evaluate_parameter("n_neighbors", [1, 3, 5, 10, 20, 50])

# 2. Waga sąsiadów
evaluate_parameter("weights", ["uniform", "distance"])

# 3. Metryka odległości
evaluate_parameter("metric", ["euclidean", "manhattan", "chebyshev", "minkowski"])

# ============================================================
# 8. ZAPIS WYNIKÓW
# ============================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv("wyniki_knn_final.csv", index=False)
print("\nZapisano wyniki do: wyniki_knn_final.csv")