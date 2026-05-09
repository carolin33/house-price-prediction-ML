import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from sklearn.model_selection import KFold, cross_validate, train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# STYL WYKRESÓW
# ============================================================
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})
ACCENT   = "#1a6b4a"
ACCENT2  = "#c0392b"
NEUTRAL  = "#2c3e50"

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
# 3. TRAIN / TEST SPLIT 
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,} próbek | Test: {len(X_test):,} próbek")

# ============================================================
# 4. PREPROCESSING
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ],
    remainder="drop"
)

# ============================================================
# 5. WALIDACJA KRZYŻOWA 
# ============================================================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

current_params = {
    "n_estimators":    100,
    "max_depth":       None,
    "min_samples_split": 2,
    "max_features":    1.0
}

all_results = []

def evaluate_and_update(param_name, values):
    global current_params

    print(f"\n{'='*60}")
    print(f"  OPTYMALIZACJA: {param_name}")
    print(f"{'='*60}")

    best_val   = None
    best_score = -np.inf

    for val in values:
        params = current_params.copy()
        params[param_name] = val

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(**params, random_state=42, n_jobs=-1))
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
            f"  {param_name}={str(val):>6} | "
            f"R2={r2:.4f} ± {scores['test_r2'].std():.4f} | "
            f"MAE={mae:.0f} | RMSE={rmse:.0f}{marker}"
        )

    current_params[param_name] = best_val
    print(f"\n  ✔ Najlepsza: {best_val}  (R2={best_score:.4f})")
    print(f"  Zaktualizowane: {current_params}")

# ============================================================
# 6. GREEDY OPTYMALIZACJA
# ============================================================
evaluate_and_update("n_estimators",      [50, 100, 200, 300, 500])
evaluate_and_update("max_depth",         [5, 10, 15, 25, None])
evaluate_and_update("min_samples_split", [2, 5, 10, 20, 50])
evaluate_and_update("max_features",      [0.3, 0.5, 0.7, 0.9, 1.0])

results_df = pd.DataFrame(all_results)
results_df.to_csv("wyniki_rf_regresja_greedy.csv", index=False)

print("\n" + "="*60)
print("FINALNE PARAMETRY:")
print(f"  {current_params}")
print("="*60)

# ============================================================
# 7. FINALNY MODEL 
# ============================================================
final_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(**current_params, random_state=42, n_jobs=-1))
])

final_pipe.fit(X_train, y_train)

y_pred_train = final_pipe.predict(X_train)
y_pred_test  = final_pipe.predict(X_test)

metrics = {
    "split":  ["Train", "Test"],
    "R2":     [r2_score(y_train, y_pred_train),  r2_score(y_test, y_pred_test)],
    "MAE":    [mean_absolute_error(y_train, y_pred_train), mean_absolute_error(y_test, y_pred_test)],
    "RMSE":   [np.sqrt(mean_squared_error(y_train, y_pred_train)), np.sqrt(mean_squared_error(y_test, y_pred_test))],
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("metryki_train_test.csv", index=False)
print("\nMetryki Train vs Test:")
print(metrics_df.to_string(index=False))

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================
feature_names = final_pipe.named_steps["preprocess"].get_feature_names_out()
importances   = final_pipe.named_steps["model"].feature_importances_

fi_df = pd.DataFrame({
    "cecha":      feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)
fi_df.to_csv("feature_importance_rf_regresja_greedy.csv", index=False)

# ============================================================
# 9. WYKRESY DO RAPORTU
# ============================================================

def thousands(x, pos):
    return f"${x/1000:.0f}k"



# -----------------------------------------------------------
# WYKRES 4: Top-15 Feature Importance
# -----------------------------------------------------------
top15 = fi_df.head(15).copy()

fig, ax = plt.subplots(figsize=(8, 6))
colors = [ACCENT if i < 3 else "#7fb5a0" for i in range(len(top15))]
bars = ax.barh(range(len(top15)), top15["importance"].values,
               color=colors, edgecolor="white")

ax.set_yticks(range(len(top15)))
ax.set_yticklabels([c.replace("num__", "").replace("cat__", "")
                    for c in top15["cecha"]], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Ważność cechy (Mean Decrease in Impurity)")
ax.set_title("Top 15 najważniejszych cech — Random Forest", fontweight="bold")

for bar, val in zip(bars, top15["importance"].values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("wykres_04_feature_importance.png", bbox_inches="tight")
plt.close()
print("Zapisano: wykres_04_feature_importance.png")

# -----------------------------------------------------------
# WYKRES 5: Greedy CV — metryki R2 per parametr
# -----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
params_order = ["n_estimators", "max_depth", "min_samples_split", "max_features"]

for ax, param in zip(axes, params_order):
    sub = results_df[results_df["parametr"] == param].copy()
    x   = range(len(sub))
    ax.errorbar(x, sub["r2_mean"], yerr=sub["r2_std"],
                fmt="o-", color=ACCENT, capsize=4, lw=2, ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["wartosc"], fontsize=9)
    ax.set_title(f"Parametr: {param}", fontweight="bold")
    ax.set_ylabel("R² (CV mean ± std)")
    ax.set_xlabel("Wartość parametru")

    best_idx = sub["r2_mean"].idxmax() - sub.index[0]
    ax.scatter([best_idx], [sub["r2_mean"].iloc[best_idx]],
               color=ACCENT2, zorder=5, s=80, label="Najlepszy")
    ax.legend(fontsize=8)

fig.suptitle("Greedy optymalizacja — R² w zależności od parametru",
             fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("wykres_05_greedy_optymalizacja.png", bbox_inches="tight")
plt.close()
print("Zapisano: wykres_05_greedy_optymalizacja.png")

# -----------------------------------------------------------
# WYKRES 6: Train vs Test metryki (bar chart)
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

fig.suptitle("Porównanie Train vs Test — finalne metryki modelu",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("wykres_06_train_vs_test_metryki.png", bbox_inches="tight")
plt.close()
print("Zapisano: wykres_06_train_vs_test_metryki.png")

print("\n Wszystkie wykresy zapisane. Gotowe do raportu.")