import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
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

# Uzupełnienie braków medianą
X_class['total_bedrooms'] = X_class['total_bedrooms'].fillna(X_class['total_bedrooms'].median())
X_reg['total_bedrooms'] = X_reg['total_bedrooms'].fillna(X_reg['total_bedrooms'].median())

# ============================================================
# 2. KONFIGURACJA
# ============================================================
N_FOLDS = 5

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

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# ============================================================
# 3. FUNKCJA EKSPERYMENTALNA
# ============================================================
all_results = []

def analyze_parameter(param_name, values):
    print(f"\n{'='*15} TEST PARAMETRU: {param_name} {'='*15}")
    print(f"{'Wartość':>12} | {'Clf Acc (mean)':>14} | {'Clf Acc (std)':>13} | {'Reg R2 (mean)':>13} | {'Reg R2 (std)':>12}")
    print("-" * 85)

    for val in values:
        p_clf = BASE_PARAMS_CLF.copy()
        p_clf[param_name] = val

        p_reg = BASE_PARAMS_REG.copy()
        p_reg[param_name] = val

        clf = RandomForestClassifier(**p_clf, random_state=42, n_jobs=-1)
        scores_clf = cross_val_score(clf, X_class, y_class, cv=kf, scoring='accuracy')

        reg = RandomForestRegressor(**p_reg, random_state=42, n_jobs=-1)
        scores_reg = cross_val_score(reg, X_reg, y_reg, cv=kf, scoring='r2')

        acc_mean = scores_clf.mean()
        acc_std  = scores_clf.std()
        r2_mean  = scores_reg.mean()
        r2_std   = scores_reg.std()

        print(f"{str(val):>12} | {acc_mean:14.4f} | {acc_std:13.4f} | {r2_mean:13.4f} | {r2_std:12.4f}")

        all_results.append({
            'parametr':     param_name,
            'wartosc':      str(val),
            'clf_acc_mean': round(acc_mean, 4),
            'clf_acc_std':  round(acc_std, 4),
            'reg_r2_mean':  round(r2_mean, 4),
            'reg_r2_std':   round(r2_std, 4)
        })

# ============================================================
# 4. URUCHOMIENIE ANALIZY
# ============================================================
analyze_parameter('n_estimators',    [10, 50, 100, 150, 200, 300])
analyze_parameter('max_depth',       [3, 8, 15, 25, None])
analyze_parameter('min_samples_split', [2, 5, 20, 50, 100])
analyze_parameter('max_features',    [0.1, 0.3, 0.5, 0.8, 1.0])

# ============================================================
# 5. ZAPIS DO CSV
# ============================================================
df_results = pd.DataFrame(all_results)
df_results.to_csv("wyniki_rf.csv", index=False)
print("\nWyniki zapisane do wyniki_rf.csv")

# ============================================================
# 6. ISTOTNOŚĆ CECH
# ============================================================
print("\n" + "="*20 + " ISTOTNOŚĆ CECH (REGRESJA) " + "="*20)
final_reg = RandomForestRegressor(**BASE_PARAMS_REG, random_state=42, n_jobs=-1)
final_reg.fit(X_reg, y_reg)
importances = pd.Series(final_reg.feature_importances_, index=X_reg.columns).sort_values(ascending=False)
print(importances.head(5))

imp_df = importances.head(5).reset_index()
imp_df.columns = ['cecha', 'importance']
imp_df.to_csv("waznosc_cech_rf.csv", index=False)
print("Istotność cech zapisana do waznosc_cech_rf.csv")