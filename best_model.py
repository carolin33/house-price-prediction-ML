"""
best_model.py — Faza Wdrożeniowa (Final Model Evaluation)

Skrypt importuje niezbędne funkcje z main.py, uruchamia ostateczne
trenowanie na pełnym zbiorze danych z wyłonionymi optymalnymi
hiperparametrami i zapisuje wyniki do the_best_result.csv.
"""

from main import (
    load_housing_data,
    run_regression_experiment,
    run_classification_experiment,
    export_results_csv,
)

# ============================================================
#  USTAWIENIA
# ============================================================

CSV_FILE = "housing.csv"
MAX_ROWS = None          # pełny zbiór danych
REPEATS = 5
TEST_RATIO = 0.2

# ============================================================
#  OPTYMALNE KONFIGURACJE
# ============================================================

BEST_REGRESSION_CONFIG = {
    "epochs": 50,
    "lr": 0.005,
    "hidden_layers": [64,32],
    "activation": "relu",
    "weight_init_scale": 1.0,
    "seed_base": 999,
    "verbose": False,
}

BEST_CLASSIFICATION_CONFIG = {
    "epochs": 50,
    "lr": 0.01,               
    "hidden_layers": [32,32],
    "activation": "tanh",
    "weight_init_scale": 1.0,
    "seed_base": 999,
    "verbose": False,
}

# ============================================================
#  GŁÓWNY PROGRAM
# ============================================================

def main():
    print("=" * 70)
    print("FAZA WDROŻENIOWA — NAJLEPSZY MODEL")
    print("=" * 70)

    rows = load_housing_data(CSV_FILE, max_rows=MAX_ROWS)
    print(f"Wczytano rekordów: {len(rows)}")

    all_csv_rows = []

    # ----- REGRESJA -----
    print("\n>>> REGRESJA — optymalna konfiguracja")
    reg_summary = run_regression_experiment(
        rows,
        BEST_REGRESSION_CONFIG,
        repeats=REPEATS,
        test_ratio=TEST_RATIO,
    )

    all_csv_rows.append({
        "problem": "regression",
        "config": str(BEST_REGRESSION_CONFIG),
        "avg_train_rmse": f"{reg_summary['avg_train_rmse']:.4f}",
        "avg_test_rmse": f"{reg_summary['avg_test_rmse']:.4f}",
        "avg_train_mae": f"{reg_summary['avg_train_mae']:.4f}",
        "avg_test_mae": f"{reg_summary['avg_test_mae']:.4f}",
        "avg_train_r2": f"{reg_summary['avg_train_r2']:.4f}",
        "avg_test_r2": f"{reg_summary['avg_test_r2']:.4f}",
        "best_test_rmse": f"{reg_summary['best_test_rmse']:.4f}",
        "best_test_r2": f"{reg_summary['best_test_r2']:.4f}",
        "avg_train_acc": "",
        "avg_test_acc": "",
        "best_test_acc": "",
    })

    # ----- KLASYFIKACJA -----
    print("\n>>> KLASYFIKACJA — optymalna konfiguracja")
    cls_summary = run_classification_experiment(
        rows,
        BEST_CLASSIFICATION_CONFIG,
        repeats=REPEATS,
        test_ratio=TEST_RATIO,
    )

    all_csv_rows.append({
        "problem": "classification",
        "config": str(BEST_CLASSIFICATION_CONFIG),
        "avg_train_rmse": "",
        "avg_test_rmse": "",
        "avg_train_mae": "",
        "avg_test_mae": "",
        "avg_train_r2": "",
        "avg_test_r2": "",
        "best_test_rmse": "",
        "best_test_r2": "",
        "avg_train_acc": f"{cls_summary['avg_train_acc']:.4f}",
        "avg_test_acc": f"{cls_summary['avg_test_acc']:.4f}",
        "best_test_acc": f"{cls_summary['best_test_acc']:.4f}",
    })

    # ----- EKSPORT -----
    export_results_csv(all_csv_rows, filename="the_best_result.csv")

    print("\n" + "=" * 70)
    print("GOTOWE — wyniki zapisano do: the_best_result.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
