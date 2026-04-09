# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ============================================================
# 1. WCZYTANIE DANYCH
# ============================================================
df = pd.read_csv("housing.csv")

# ============================================================
# 2. DEFINICJA ZADAN
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
# 4. WALIDACJA KRZYZOWA
# ============================================================
cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 5. PARAMETRY BAZOWE
# ============================================================
BASE_PARAMS_CLF = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale"
}

BASE_PARAMS_REG = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "epsilon": 0.1
}

# ============================================================
# 6. FUNKCJA TESTUJACA
# ============================================================
all_results = []

def evaluate_parameter(param_name, values):
    print(f"\n{'='*20} TEST PARAMETRU: {param_name} {'='*20}")

    for val in values:
        params_clf = BASE_PARAMS_CLF.copy()
        params_reg = BASE_PARAMS_REG.copy()

        if param_name in params_clf:
            params_clf[param_name] = val
        if param_name in params_reg:
            params_reg[param_name] = val

        clf_pipe = Pipeline([
            ("preprocess", preprocessor_clf),
            ("model", SVC(**params_clf))
        ])

        reg_pipe = Pipeline([
            ("preprocess", preprocessor_reg),
            ("model", SVR(**params_reg))
        ])

        clf_scores = cross_validate(
            clf_pipe,
            X_clf,
            y_clf,
            cv=cv_clf,
            scoring="accuracy",
            n_jobs=-1
        )

        reg_scores = cross_validate(
            reg_pipe,
            X_reg,
            y_reg,
            cv=cv_reg,
            scoring="r2",
            n_jobs=-1
        )

        result = {
            "parametr": param_name,
            "wartosc": str(val),
            "clf_acc_mean": round(clf_scores["test_score"].mean(), 4),
            "reg_r2_mean": round(reg_scores["test_score"].mean(), 4)
        }

        all_results.append(result)

        print(
            f"Wartosc={val} | "
            f"clf_acc_mean={result['clf_acc_mean']:.4f} | "
            f"reg_r2_mean={result['reg_r2_mean']:.4f}"
        )

# ============================================================
# 7. EKSPERYMENTY
# ============================================================
evaluate_parameter("C", [0.1, 1, 10, 100])
evaluate_parameter("kernel", ["linear", "rbf", "poly", "sigmoid"])
evaluate_parameter("gamma", ["scale", "auto", 0.01, 0.1])
evaluate_parameter("epsilon", [0.01, 0.1, 0.5, 1.0])

# ============================================================
# 8. ZAPIS WYNIKOW
# ============================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv("wyniki_svm.csv", index=False)
print("\nZapisano wyniki do: wyniki_svm.csv")