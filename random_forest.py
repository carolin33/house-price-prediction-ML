import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. PRZYGOTOWANIE DANYCH (Preprocessing)
# ============================================================
print("Przygotowywanie danych...")
df = pd.read_csv("housing.csv")

# --- DANE DO KLASYFIKACJI (Zmienna celu: ocean_proximity) ---
le = LabelEncoder()
y_class = le.fit_transform(df["ocean_proximity"])
print("Mapowanie klas:", dict(zip(le.classes_, le.transform(le.classes_))))
X_class = df.drop(["ocean_proximity", "longitude", "latitude"], axis=1)

# --- DANE DO REGRESJI (Zmienna celu: median_house_value) ---
df_reg = pd.get_dummies(df, columns=["ocean_proximity"])
y_reg = df_reg["median_house_value"]
X_reg = df_reg.drop(["median_house_value"], axis=1)

# Podział na zbiory Treningowe i Testowe (80/20)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Uzupełnienie braków medianą liczoną tylko na danych treningowych
median_c = X_train_c['total_bedrooms'].median()
X_train_c['total_bedrooms'] = X_train_c['total_bedrooms'].fillna(median_c)
X_test_c['total_bedrooms']  = X_test_c['total_bedrooms'].fillna(median_c)

median_r = X_train_r['total_bedrooms'].median()
X_train_r['total_bedrooms'] = X_train_r['total_bedrooms'].fillna(median_r)
X_test_r['total_bedrooms']  = X_test_r['total_bedrooms'].fillna(median_r)

# ============================================================
# 2. FUNKCJA EKSPERYMENTALNA (Analiza parametrów)
# ============================================================
BASE_PARAMS_CLF = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'max_features': 'sqrt'
}

BASE_PARAMS_REG = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'max_features': 1.0
}

def analyze_parameter(param_name, values):
    print(f"\n{'='*15} TEST PARAMETRU: {param_name} {'='*15}")
    print(f"{'Wartość':>12} | {'Klasyfikacja (Acc)':>18} | {'Regresja (R2)':>15}")
    print("-" * 65)

    for val in values:
        p_clf = BASE_PARAMS_CLF.copy()
        p_clf[param_name] = val

        p_reg = BASE_PARAMS_REG.copy()
        p_reg[param_name] = val

        clf = RandomForestClassifier(**p_clf, random_state=42, n_jobs=-1)
        clf.fit(X_train_c, y_train_c)
        acc = clf.score(X_test_c, y_test_c)

        reg = RandomForestRegressor(**p_reg, random_state=42, n_jobs=-1)
        reg.fit(X_train_r, y_train_r)
        r2 = reg.score(X_test_r, y_test_r)

        print(f"{str(val):>12} | {acc:18.4f} | {r2:15.4f}")

# ============================================================
# 3. URUCHOMIENIE ANALIZY DLA 4 CECH (Zgodnie z wymogami)
# ============================================================
analyze_parameter('n_estimators', [10, 50, 100, 150, 200, 300])
analyze_parameter('max_depth', [3, 8, 15, 25, None])
analyze_parameter('min_samples_split', [2, 5, 20, 50, 100])
analyze_parameter('max_features', [0.1, 0.3, 0.5, 0.8, 1.0])

# ============================================================
# 4. ISTOTNOŚĆ CECH
# ============================================================
print("\n" + "="*20 + " ISTOTNOŚĆ CECH (REGRESJA) " + "="*20)
final_reg = RandomForestRegressor(**BASE_PARAMS_REG, random_state=42, n_jobs=-1)
final_reg.fit(X_train_r, y_train_r)
importances = pd.Series(final_reg.feature_importances_, index=X_reg.columns).sort_values(ascending=False)
print(importances.head(5))