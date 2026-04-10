# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
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
# 6. FUNKCJA OCENY
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

        # =====================================================
        # KLASYFIKACJA
        # epsilon nie dotyczy SVC, wiec pomijamy liczenie
        # =====================================================
        if param_name != "epsilon":
            clf_pipe = Pipeline([
                ("preprocess", preprocessor_clf),
                ("model", SVC(**params_clf))
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

            clf_accuracy_mean = round(clf_scores["test_accuracy"].mean(), 4)
            clf_accuracy_std = round(clf_scores["test_accuracy"].std(), 4)
            clf_balanced_accuracy_mean = round(clf_scores["test_balanced_accuracy"].mean(), 4)
            clf_balanced_accuracy_std = round(clf_scores["test_balanced_accuracy"].std(), 4)
            clf_f1_macro_mean = round(clf_scores["test_f1_macro"].mean(), 4)
            clf_f1_macro_std = round(clf_scores["test_f1_macro"].std(), 4)
        else:
            clf_accuracy_mean = "-"
            clf_accuracy_std = "-"
            clf_balanced_accuracy_mean = "-"
            clf_balanced_accuracy_std = "-"
            clf_f1_macro_mean = "-"
            clf_f1_macro_std = "-"

        # =====================================================
        # REGRESJA
        # =====================================================
        reg_pipe = Pipeline([
            ("preprocess", preprocessor_reg),
            ("model", SVR(**params_reg))
        ])

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
            "model": "SVM",
            "parametr": param_name,
            "wartosc": str(val),
            "clf_accuracy_mean": clf_accuracy_mean,
            "clf_accuracy_std": clf_accuracy_std,
            "clf_balanced_accuracy_mean": clf_balanced_accuracy_mean,
            "clf_balanced_accuracy_std": clf_balanced_accuracy_std,
            "clf_f1_macro_mean": clf_f1_macro_mean,
            "clf_f1_macro_std": clf_f1_macro_std,
            "reg_r2_mean": round(reg_scores["test_r2"].mean(), 4),
            "reg_r2_std": round(reg_scores["test_r2"].std(), 4),
            "reg_mae_mean": round(-reg_scores["test_neg_mae"].mean(), 2),
            "reg_mae_std": round(reg_scores["test_neg_mae"].std(), 2),
            "reg_rmse_mean": round(-reg_scores["test_neg_rmse"].mean(), 2),
            "reg_rmse_std": round(reg_scores["test_neg_rmse"].std(), 2)
        }

        all_results.append(result)

        print(
            f"Wartosc={val} | "
            f"Acc={result['clf_accuracy_mean']} | "
            f"BalAcc={result['clf_balanced_accuracy_mean']} | "
            f"F1={result['clf_f1_macro_mean']} | "
            f"R2={result['reg_r2_mean']:.4f} +- {result['reg_r2_std']:.4f} | "
            f"MAE={result['reg_mae_mean']:.2f} | "
            f"RMSE={result['reg_rmse_mean']:.2f}"
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
results_df.to_csv("wyniki_svm_poprawione.csv", index=False)
print("\nZapisano wyniki do: wyniki_svm_poprawione.csv")