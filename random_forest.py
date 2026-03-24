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

# Uzupełnienie braków medianą 
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# --- DANE DO KLASYFIKACJI (Zmienna celu: ocean_proximity) ---
le = LabelEncoder()
y_class = le.fit_transform(df["ocean_proximity"])
# Do klasyfikacji używamy cech numerycznych (bez samej lokalizacji tekstowej)
X_class = df.drop(["ocean_proximity"], axis=1)

# --- DANE DO REGRESJI (Zmienna celu: median_house_value) ---
# Ocean_proximity zamieniamy na kolumny 0-1 (One-Hot Encoding)
df_reg = pd.get_dummies(df, columns=["ocean_proximity"])
y_reg = df_reg["median_house_value"]
X_reg = df_reg.drop(["median_house_value"], axis=1)

# Podział na zbiory Treningowe i Testowe (80/20)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# ============================================================
# 2. FUNKCJA EKSPERYMENTALNA (Analiza parametrów)
# ============================================================
# Parametry bazowe 
BASE_PARAMS = {
    'n_estimators': 100, 
    'max_depth': None, 
    'min_samples_split': 2, 
    'max_features': 1.0 # 1.0 oznacza użycie wszystkich cech
}

def analyze_parameter(param_name, values):
    print(f"\n{'='*15} TEST PARAMETRU: {param_name} {'='*15}")
    print(f"{'Wartość':>12} | {'Klasyfikacja (Acc)':>18} | {'Regresja (R2)':>15}")
    print("-" * 65)
    
    for val in values:
        p = BASE_PARAMS.copy()
        p[param_name] = val
        
        # 1. Klasyfikacja - Las Losowy
        clf = RandomForestClassifier(**p, random_state=42, n_jobs=-1)
        clf.fit(X_train_c, y_train_c)
        acc = clf.score(X_test_c, y_test_c)
        
        # 2. Regresja - Las Losowy
        reg = RandomForestRegressor(**p, random_state=42, n_jobs=-1)
        reg.fit(X_train_r, y_train_r)
        r2 = reg.score(X_test_r, y_test_r)
        
        print(f"{str(val):>12} | {acc:18.4f} | {r2:15.4f}")

# ============================================================
# 3. URUCHOMIENIE ANALIZY DLA 4 CECH (Zgodnie z wymogami)
# ============================================================

# Cecha 1: Liczba drzew w lesie
# 1. n_estimators - tutaj warto sprawdzić więcej na początku
analyze_parameter('n_estimators', [10, 50, 100, 150, 200, 300]) 

# 2. max_depth - sprawdzamy skokowo
analyze_parameter('max_depth', [3, 8, 15, 25, None]) 

# 3. min_samples_split - tutaj różnice są małe, więc 4-5 wartości starczy
analyze_parameter('min_samples_split', [2, 5, 20, 50, 100])

# 4. max_features - sprawdzamy kluczowe punkty
analyze_parameter('max_features', [0.1, 0.3, 0.5, 0.8, 1.0])


# ============================================================
# 4.ISTOTNOŚĆ CECH 
# ============================================================
print("\n" + "="*20 + " ISTOTNOŚĆ CECH (REGRESJA) " + "="*20)
final_reg = RandomForestRegressor(**BASE_PARAMS, random_state=42, n_jobs=-1)
final_reg.fit(X_train_r, y_train_r)
importances = pd.Series(final_reg.feature_importances_, index=X_reg.columns).sort_values(ascending=False)
print(importances.head(5))