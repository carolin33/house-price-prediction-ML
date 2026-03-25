import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

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
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

BASE_PARAMS_CLF = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'metric': 'euclidean',
    'p': 2
}

BASE_PARAMS_REG = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'metric': 'euclidean',
    'p': 2
}

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

        # KNN wymaga skalowania — używamy Pipeline żeby skalowanie
        # odbywało się osobno w każdym foldzie (brak data leakage)
        clf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(**p_clf))
        ])

        reg_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(**p_reg))
        ])

        scores_clf = cross_val_score(clf_pipeline, X_class, y_class, cv=kf, scoring='accuracy')
        scores_reg = cross_val_score(reg_pipeline, X_reg, y_reg, cv=kf, scoring='r2')

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
# 4. URUCHOMIENIE ANALIZY DLA 4 PARAMETRÓW
# ============================================================

# ============================================================
# 4. URUCHOMIENIE ANALIZY DLA 4 PARAMETRÓW
# ============================================================

# 1. Liczba sąsiadów
analyze_parameter('n_neighbors', [1, 3, 5, 10, 20, 50])

# 2. Waga sąsiadów
analyze_parameter('weights', ['uniform', 'distance'])

# 3. Metryka odległości
analyze_parameter('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])

# 4. Stopień wielomianu Minkowskiego — wymaga metric='minkowski'
BASE_PARAMS_CLF['metric'] = 'minkowski'
BASE_PARAMS_REG['metric'] = 'minkowski'
analyze_parameter('p', [1, 2, 3, 4])
BASE_PARAMS_CLF['metric'] = 'euclidean'
BASE_PARAMS_REG['metric'] = 'euclidean'

# ============================================================
# 5. ZAPIS DO CSV
# ============================================================
df_results = pd.DataFrame(all_results)
df_results.to_csv("wyniki_knn.csv", index=False)
print("\nWyniki zapisane do wyniki_knn.csv")