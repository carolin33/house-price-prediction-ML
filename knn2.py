import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# STYL WYKRESÓW
# ============================================================
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})
ACCENT  = "#2563eb"   # niebieski — inny niż RF (zielony) dla łatwego rozróżnienia w raporcie
ACCENT2 = "#c0392b"
NEUTRAL = "#2c3e50"

def thousands(x, pos):
    return f"${x/1000:.0f}k"

# ============================================================
# 1. WCZYTANIE DANYCH
# ============================================================
df = pd.read_csv("housing.csv")

# ============================================================
# 2. DEFINICJA ZADANIA
# ============================================================
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# ============================================================
# 3. TRAIN / TEST SPLIT (20% chowane do szuflady)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,} próbek | Test: {len(X_test):,} próbek")

# ============================================================
# 4. PREPROCESSING
# ============================================================
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe,    num_cols),
        ("cat", categorical_pipe, cat_cols)
    ],
    remainder="drop"
)

# ============================================================
# 5. WALIDACJA KRZYŻOWA (tylko na TRAIN)
# ============================================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

current_params = {
    "n_neighbors": 5,
    "weights":     "uniform",
    "metric":      "minkowski",
    "p":           2,
    "n_jobs":      1
}

all_results = []

# ============================================================
# 6. GREEDY OPTYMALIZACJA
# ============================================================
def evaluate_and_update(param_name, values):
    global current_params

    print(f"\n{'='*60}")
    print(f"  OPTYMALIZACJA: {param_name}")
    print(f"  Aktualne parametry: {current_params}")
    print(f"{'='*60}")

    best_val   = None
    best_score = -np.inf

    for val in values:
        params = current_params.copy()
        params[param_name] = val

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model",      KNeighborsRegressor(**params))
        ])

        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={
                "r2":       "r2",
                "neg_mae":  "neg_mean_absolute_error",
                "neg_rmse": "neg_root_mean_squared_error"
            },
            n_jobs=1
        )

        r2   = scores["test_r2"].mean()
        mae  = -scores["test_neg_mae"].mean()
        rmse = -scores["test_neg_rmse"].mean()

        result = {
            "parametr":        param_name,
            "wartosc":         str(val),
            "r2_mean":         round(r2,   4),
            "r2_std":          round(scores["test_r2"].std(), 4),
            "mae_mean":        round(mae,  2),
            "mae_std":         round(scores["test_neg_mae"].std(), 2),
            "rmse_mean":       round(rmse, 2),
            "rmse_std":        round(scores["test_neg_rmse"].std(), 2),
            "params_snapshot": str(params),
        }
        all_results.append(result)

        marker = ""
        if r2 > best_score:
            best_score = r2
            best_val   = val
            marker     = "  ◄ NOWE NAJLEPSZE"

        print(
            f"  {param_name}={str(val):>12} | "
            f"R2={r2:.4f} ± {scores['test_r2'].std():.4f} | "
            f"MAE={mae:.0f} | RMSE={rmse:.0f}{marker}"
        )

    current_params[param_name] = best_val
    print(f"\n  ✔ Najlepsza: {best_val}  (R2={best_score:.4f})")
    print(f"  Zaktualizowane: {current_params}")


evaluate_and_update("n_neighbors", [1, 3, 5, 10, 20, 50])
evaluate_and_update("weights",     ["uniform", "distance"])
evaluate_and_update("metric",      ["euclidean", "manhattan", "chebyshev", "minkowski"])

results_df = pd.DataFrame(all_results)
results_df.to_csv("wyniki_knn_regresja_greedy.csv", index=False)

print("\n" + "="*60)
print("FINALNE PARAMETRY:")
print(f"  {current_params}")
print("="*60)

# ============================================================
# 7. FINALNY MODEL → FIT NA TRAIN, EWALUACJA NA TEST
# ============================================================
final_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model",      KNeighborsRegressor(**current_params))
])

final_pipe.fit(X_train, y_train)

y_pred_train = final_pipe.predict(X_train)
y_pred_test  = final_pipe.predict(X_test)

metrics = {
    "split": ["Train", "Test"],
    "R2":    [r2_score(y_train, y_pred_train),             r2_score(y_test, y_pred_test)],
    "MAE":   [mean_absolute_error(y_train, y_pred_train),  mean_absolute_error(y_test, y_pred_test)],
    "RMSE":  [np.sqrt(mean_squared_error(y_train, y_pred_train)),
              np.sqrt(mean_squared_error(y_test,  y_pred_test))],
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("metryki_knn_train_test.csv", index=False)
print("\nMetryki Train vs Test:")
print(metrics_df.to_string(index=False))

# ============================================================
# 8. WYKRESY DO RAPORTU
# ============================================================

# -----------------------------------------------------------
# WYKRES 1: Scatter Predicted vs Actual (test set)
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(y_test, y_pred_test,
           alpha=0.25, s=10, color=ACCENT, rasterized=True, label="Test set")

lims = [min(y_test.min(), y_pred_test.min()),
        max(y_test.max(), y_pred_test.max())]
ax.plot(lims, lims, "--", color=NEUTRAL, lw=1.5, label="Idealna predykcja")

r2   = r2_score(y_test, y_pred_test)
mae  = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

ax.text(0.05, 0.93,
        f"R² = {r2:.4f}\nMAE = ${mae:,.0f}\nRMSE = ${rmse:,.0f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

ax.xaxis.set_major_formatter(FuncFormatter(thousands))
ax.yaxis.set_major_formatter(FuncFormatter(thousands))
ax.set_xlabel("Rzeczywista wartość mediany domu")
ax.set_ylabel("Predykcja modelu")
ax.set_title("Predykcja vs Rzeczywistość — KNN (test set)", fontweight="bold")
ax.legend(framealpha=0.7)
plt.tight_layout()
plt.savefig("knn_wykres_01_scatter_pred_vs_actual.png", bbox_inches="tight")
plt.close()
print("Zapisano: knn_wykres_01_scatter_pred_vs_actual.png")

# -----------------------------------------------------------
# WYKRES 2: Residuals vs Predicted
# -----------------------------------------------------------
residuals = y_test.values - y_pred_test

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred_test, residuals,
           alpha=0.25, s=10, color=ACCENT2, rasterized=True)
ax.axhline(0, color=NEUTRAL, lw=1.5, linestyle="--")
ax.axhline( residuals.std(), color=NEUTRAL, lw=0.8, linestyle=":", alpha=0.6)
ax.axhline(-residuals.std(), color=NEUTRAL, lw=0.8, linestyle=":", alpha=0.6)
ax.fill_between([y_pred_test.min(), y_pred_test.max()],
                -residuals.std(), residuals.std(),
                alpha=0.07, color=NEUTRAL)

ax.xaxis.set_major_formatter(FuncFormatter(thousands))
ax.yaxis.set_major_formatter(FuncFormatter(thousands))
ax.set_xlabel("Predykcja modelu")
ax.set_ylabel("Residuum (rzeczywista − predykcja)")
ax.set_title("Residua — KNN (test set)", fontweight="bold")
ax.text(0.02, 0.97, f"±1 std = ±${residuals.std():,.0f}",
        transform=ax.transAxes, fontsize=9, va="top", color="grey")
plt.tight_layout()
plt.savefig("knn_wykres_02_residuals.png", bbox_inches="tight")
plt.close()
print("Zapisano: knn_wykres_02_residuals.png")

# -----------------------------------------------------------
# WYKRES 3: Histogram residuów
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(residuals, bins=60, color=ACCENT, edgecolor="white", linewidth=0.3)
ax.axvline(0, color=ACCENT2, lw=2, linestyle="--", label="Zero")
ax.axvline(residuals.mean(), color=NEUTRAL, lw=1.5,
           linestyle=":", label=f"Średnia = ${residuals.mean():,.0f}")
ax.set_xlabel("Residuum ($)")
ax.set_ylabel("Liczba próbek")
ax.set_title("Rozkład residuów — KNN (test set)", fontweight="bold")
ax.xaxis.set_major_formatter(FuncFormatter(thousands))
ax.legend()
plt.tight_layout()
plt.savefig("knn_wykres_03_histogram_residuals.png", bbox_inches="tight")
plt.close()
print("Zapisano: knn_wykres_03_histogram_residuals.png")

# -----------------------------------------------------------
# WYKRES 4: R² vs n_neighbors (kluczowy dla KNN)
# -----------------------------------------------------------
sub_k = results_df[results_df["parametr"] == "n_neighbors"].copy()

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(sub_k["wartosc"].astype(str), sub_k["r2_mean"], yerr=sub_k["r2_std"],
            fmt="o-", color=ACCENT, capsize=5, lw=2, ms=7)

best_idx = sub_k["r2_mean"].idxmax() - sub_k.index[0]
ax.scatter([best_idx], [sub_k["r2_mean"].iloc[best_idx]],
           color=ACCENT2, zorder=5, s=100, label=f"Najlepsze k={sub_k['wartosc'].iloc[best_idx]}")

ax.set_xlabel("Liczba sąsiadów (k)")
ax.set_ylabel("R² (CV mean ± std)")
ax.set_title("Wpływ liczby sąsiadów na R² — KNN", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("knn_wykres_04_n_neighbors.png", bbox_inches="tight")
plt.close()
print("Zapisano: knn_wykres_04_n_neighbors.png")

# -----------------------------------------------------------
# WYKRES 5: Greedy CV — R² per parametr
# -----------------------------------------------------------
params_order = ["n_neighbors", "weights", "metric"]
fig, axes = plt.subplots(1, 3, figsize=(13, 5))

for ax, param in zip(axes, params_order):
    sub = results_df[results_df["parametr"] == param].copy()
    x   = range(len(sub))
    ax.errorbar(x, sub["r2_mean"], yerr=sub["r2_std"],
                fmt="o-", color=ACCENT, capsize=4, lw=2, ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["wartosc"], fontsize=9, rotation=15)
    ax.set_title(f"Parametr: {param}", fontweight="bold")
    ax.set_ylabel("R² (CV)")
    ax.set_xlabel("Wartość")

    best_idx = sub["r2_mean"].idxmax() - sub.index[0]
    ax.scatter([best_idx], [sub["r2_mean"].iloc[best_idx]],
               color=ACCENT2, zorder=5, s=80, label="Najlepszy")
    ax.legend(fontsize=8)

fig.suptitle("Greedy optymalizacja KNN — R² w zależności od parametru",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("knn_wykres_05_greedy_optymalizacja.png", bbox_inches="tight")
plt.close()
print("Zapisano: knn_wykres_05_greedy_optymalizacja.png")

# -----------------------------------------------------------
# WYKRES 6: Train vs Test metryki
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
metric_names = ["R²", "MAE ($)", "RMSE ($)"]
train_vals   = [metrics["R2"][0], metrics["MAE"][0], metrics["RMSE"][0]]
test_vals    = [metrics["R2"][1], metrics["MAE"][1], metrics["RMSE"][1]]

for ax, name, tv, tev in zip(axes, metric_names, train_vals, test_vals):
    bars = ax.bar(["Train", "Test"], [tv, tev],
                  color=[ACCENT, ACCENT2], edgecolor="white", width=0.5)
    ax.set_title(name, fontweight="bold")
    for bar, val in zip(bars, [tv, tev]):
        label = f"{val:.4f}" if name == "R²" else f"${val:,.0f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                label, ha="center", va="bottom", fontsize=10)
    if name != "R²":
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))

fig.suptitle("Porównanie Train vs Test — finalne metryki KNN",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("knn_wykres_06_train_vs_test_metryki.png", bbox_inches="tight")
plt.close()
print("Zapisano: knn_wykres_06_train_vs_test_metryki.png")

print("\n✅ Wszystkie wykresy zapisane. Gotowe do raportu.")