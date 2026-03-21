"""
play_with_model.py — Skrypt demonstracyjny (Inference)

Trenuje optymalny model regresyjny MLP na zbiorze housing.csv,
a następnie przewiduje cenę wymyślonego domu.
"""

from main import (
    load_housing_data,
    build_regression_dataset,
    train_test_split,
    fit_standardizer,
    transform_standardize,
    fit_target_scaler_regression,
    transform_target_regression,
    inverse_transform_value_regression,
    one_hot,
    MLP,
)

# ============================================================
#  USTAWIENIA
# ============================================================

CSV_FILE = "housing.csv"
SEED = 1001
TEST_RATIO = 0.2

# ============================================================
#  1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# ============================================================

print("=" * 60)
print("INFERENCE — WYCENA NOWEGO DOMU")
print("=" * 60)

rows = load_housing_data(CSV_FILE)
print(f"Wczytano rekordów: {len(rows)}")

X, Y, meta = build_regression_dataset(rows)
categories = meta["categories"]  # posortowane klasy ocean_proximity

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, TEST_RATIO, seed=SEED)

# standaryzacja cech (na podstawie zbioru uczącego)
x_mean, x_std = fit_standardizer(X_train)
X_train_n = transform_standardize(X_train, x_mean, x_std)

# skalowanie targetu
y_mean, y_std = fit_target_scaler_regression(Y_train)
Y_train_n = transform_target_regression(Y_train, y_mean, y_std)

# ============================================================
#  2. TRENING OPTYMALNEGO MODELU
# ============================================================

input_size = len(X_train_n[0])
layer_sizes = [input_size, 64, 32, 1]

print(f"\nArchitektura sieci: {layer_sizes}")
print(f"Aktywacja: relu")
print(f"Trenowanie: epochs=50, lr=0.005, seed={SEED}")
print("Proszę czekać...\n")

model = MLP(
    layer_sizes=layer_sizes,
    task="regression",
    activation="relu",
    seed=SEED,
)

model.fit(
    X_train_n,
    Y_train_n,
    epochs=50,
    lr=0.005,
    seed=SEED,
    verbose=True,
)

# ============================================================
#  3. DEFINICJA NOWEGO DOMU
# ============================================================

custom_house = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}

# ============================================================
#  4. KONWERSJA NA WEKTOR WEJŚCIOWY
# ============================================================

# cechy numeryczne (ta sama kolejność co w build_regression_dataset)
numeric_features = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income",
]

x_raw = [custom_house[col] for col in numeric_features]

# one-hot encoding ocean_proximity
cat_index = categories.index(custom_house["ocean_proximity"])
ocean_vec = one_hot(cat_index, len(categories))
x_raw.extend(ocean_vec)

# standaryzacja (tymi samymi parametrami co zbiór uczący)
x_scaled = [(x_raw[j] - x_mean[j]) / x_std[j] for j in range(len(x_raw))]

# ============================================================
#  5. PREDYKCJA
# ============================================================

pred_scaled = model.predict_one(x_scaled)
predicted_price = inverse_transform_value_regression(pred_scaled[0], y_mean, y_std)

# ============================================================
#  6. WYNIK
# ============================================================

print("\n" + "=" * 60)
print("DANE WEJŚCIOWE NOWEGO DOMU")
print("=" * 60)

for key, val in custom_house.items():
    print(f"  {key:25s} : {val}")

print("\n" + "=" * 60)
print(f"  PRZEWIDYWANA CENA:  ${predicted_price:,.2f}")
print("=" * 60)
