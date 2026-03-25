import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ============================================================
# 1. PRZYGOTOWANIE DANYCH
# ============================================================
print("Przygotowywanie danych...")
df = pd.read_csv("housing.csv")

# --- KLASYFIKACJA ( ocean_proximity ) ---
le = LabelEncoder()
y_class = le.fit_transform(df["ocean_proximity"])
# Usuwamy longitude i latitude, by model nie "zgadywał" położenia z mapy
X_class = df.drop(["ocean_proximity", "longitude", "latitude"], axis=1)

# --- REGRESJA ( median_house_value ) ---
df_reg = pd.get_dummies(df, columns=["ocean_proximity"])
y_reg = df_reg["median_house_value"]
X_reg = df_reg.drop(["median_house_value"], axis=1)

# ============================================================
# 2. KONFIGURACJA
# ============================================================
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Parametry bazowe (p=2 jest domyślne dla euklidesowej, usunięte z bazy dla przejrzystości)
BASE_PARAMS_CLF = {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'euclidean'}
BASE_PARAMS_REG = {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'euclidean'}

# ============================================================
# 3. FUNKCJA EKSPERYMENTALNA
# ============================================================
all_results = []

def analyze_parameter(param_name, values):
    print(f"\n{'='*15} TEST PARAMETRU: {param_name} {'='*15}")
    print(f"{'Wartość':>12} | {'Clf Acc (mean)':>14} | {'Reg R2 (mean)':>13}")
    print("-" * 50)

    for val in values:
        p_clf = BASE_PARAMS_CLF.copy()
        p_clf[param_name] = val
        p_reg = BASE_PARAMS_REG.copy()
        p_reg[param_name] = val

        # Pipeline: Imputacja -> Skalowanie -> KNN
        clf_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(**p_clf))
        ])

        reg_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(**p_reg))
        ])

        scores_clf = cross_val_score(clf_pipe, X_class, y_class, cv=kf, scoring='accuracy')
        scores_reg = cross_val_score(reg_pipe, X_reg, y_reg, cv=kf, scoring='r2')

        acc_mean = scores_clf.mean()
        r2_mean = scores_reg.mean()

        print(f"{str(val):>12} | {acc_mean:14.4f} | {r2_mean:13.4f}")

        all_results.append({
            'parametr': param_name,
            'wartosc': str(val),
            'clf_acc_mean': round(acc_mean, 4),
            'reg_r2_mean': round(r2_mean, 4)
        })

# ============================================================
# 4. URUCHOMIENIE ANALIZY (4 PARAMETRY)
# ============================================================

# 1. Liczba sąsiadów
analyze_parameter('n_neighbors', [1, 3, 5, 10, 20, 50])

# 2. Waga sąsiadów
analyze_parameter('weights', ['uniform', 'distance'])

# 3. Metryka odległości
analyze_parameter('metric', ['euclidean', 'manhattan', 'chebyshev'])

# 4. Rozmiar liścia (leaf_size) - Parametr wymagany przez instrukcję
analyze_parameter('leaf_size', [10, 20, 30, 50, 100])

# ============================================================
# 5. ZAPIS DO CSV
# ============================================================
pd.DataFrame(all_results).to_csv("wyniki_knn.csv", index=False)
print("\nWyniki zapisane do wyniki_knn.csv")