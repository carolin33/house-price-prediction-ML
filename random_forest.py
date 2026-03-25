import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ============================================================
# 1. PRZYGOTOWANIE DANYCH
# ============================================================
df = pd.read_csv("housing.csv")

# --- KLASYFIKACJA ---
le = LabelEncoder()
y_class = le.fit_transform(df["ocean_proximity"])
X_class = df.drop(["ocean_proximity", "longitude", "latitude"], axis=1)

# --- REGRESJA ---
df_reg = pd.get_dummies(df, columns=["ocean_proximity"])
y_reg = df_reg["median_house_value"]
X_reg = df_reg.drop(["median_house_value"], axis=1)

# UWAGA: Nie robimy fillna() tutaj! Zrobi to SimpleImputer wewnątrz Pipeline.

# ============================================================
# 2. KONFIGURACJA
# ============================================================
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

BASE_PARAMS_CLF = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'max_features': 'sqrt'}
BASE_PARAMS_REG = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'max_features': 1.0}

# ============================================================
# 3. FUNKCJA EKSPERYMENTALNA (Z Pipeline i Imputerem)
# ============================================================
all_results = []

def analyze_parameter(param_name, values):
    print(f"\nTEST PARAMETRU: {param_name}")
    for val in values:
        p_clf = BASE_PARAMS_CLF.copy()
        p_clf[param_name] = val
        p_reg = BASE_PARAMS_REG.copy()
        p_reg[param_name] = val

        # Pipeline: Najpierw uzupełnij braki (medianą z treningu), potem model
        clf_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('rf', RandomForestClassifier(**p_clf, random_state=42, n_jobs=-1))
        ])

        reg_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('rf', RandomForestRegressor(**p_reg, random_state=42, n_jobs=-1))
        ])

        scores_clf = cross_val_score(clf_pipe, X_class, y_class, cv=kf, scoring='accuracy')
        scores_reg = cross_val_score(reg_pipe, X_reg, y_reg, cv=kf, scoring='r2')

        all_results.append({
            'parametr': param_name, 'wartosc': str(val),
            'clf_acc_mean': round(scores_clf.mean(), 4),
            'reg_r2_mean': round(scores_reg.mean(), 4)
        })
        print(f"Wartość: {val} | Acc: {scores_clf.mean():.4f} | R2: {scores_reg.mean():.4f}")

# ============================================================
# 4. URUCHOMIENIE I ISTOTNOŚĆ
# ============================================================
analyze_parameter('n_estimators', [10, 50, 100, 150, 200, 300])
analyze_parameter('max_depth', [3, 8, 15, 25, None])
analyze_parameter('min_samples_split', [2, 5, 20, 50, 100])
analyze_parameter('max_features', [0.1, 0.3, 0.5, 0.8, 1.0])

pd.DataFrame(all_results).to_csv("wyniki_rf.csv", index=False)

# Istotność cech (na całym zbiorze dla prezentacji)
final_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestRegressor(**BASE_PARAMS_REG, random_state=42, n_jobs=-1))
])
final_pipe.fit(X_reg, y_reg)
importances = pd.Series(final_pipe.named_steps['rf'].feature_importances_, index=X_reg.columns).sort_values(ascending=False)
importances.head(5).to_csv("waznosc_cech_rf.csv", header=True)