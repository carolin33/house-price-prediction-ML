import csv
import math
import random
from copy import deepcopy

# ============================================================
#  USTAWIENIA GŁÓWNE
# ============================================================

CSV_FILE = "housing.csv"
TEST_RATIO = 0.2
REPEATS = 3   # ile razy powtórzyć każdy zestaw parametrów
MAX_ROWS = None  # np. 5000 jeśli chcesz szybciej testować; None = cały zbiór

# ============================================================
#  NARZĘDZIA
# ============================================================

def safe_float(x):
    if x is None or x == "":
        return None
    return float(x)

def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)

def std(values):
    if not values:
        return 1.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    s = math.sqrt(var)
    return s if s > 1e-12 else 1.0

def shuffle_in_unison(X, Y, seed):
    idx = list(range(len(X)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    X2 = [X[i] for i in idx]
    Y2 = [Y[i] for i in idx]
    return X2, Y2

def train_test_split(X, Y, test_ratio=0.2, seed=42):
    Xs, Ys = shuffle_in_unison(X, Y, seed)
    split = int(len(Xs) * (1 - test_ratio))
    return Xs[:split], Xs[split:], Ys[:split], Ys[split:]

def one_hot(index, size):
    v = [0.0] * size
    v[index] = 1.0
    return v

# ============================================================
#  WGRYWANIE I PRZYGOTOWANIE DANYCH
# ============================================================

def load_housing_data(csv_file, max_rows=None):
    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(row)
    return rows

def impute_numeric_means(rows, numeric_columns):
    means = {}
    for col in numeric_columns:
        vals = []
        for row in rows:
            v = safe_float(row[col])
            if v is not None:
                vals.append(v)
        means[col] = mean(vals)

    new_rows = []
    for row in rows:
        r = dict(row)
        for col in numeric_columns:
            if r[col] == "" or r[col] is None:
                r[col] = str(means[col])
        new_rows.append(r)

    return new_rows, means

def build_regression_dataset(rows):
    """
    Wejście:
      cechy numeryczne + one-hot z ocean_proximity
    Wyjście:
      median_house_value
    """
    numeric_features = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]

    # imputacja braków
    rows, _ = impute_numeric_means(rows, numeric_features + ["median_house_value"])

    # klasy ocean_proximity
    categories = sorted(list(set(row["ocean_proximity"] for row in rows)))
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    X = []
    Y = []

    for row in rows:
        x = []
        for col in numeric_features:
            x.append(float(row[col]))

        # one-hot ocean_proximity jako cecha wejściowa
        ocean_vec = one_hot(cat_to_idx[row["ocean_proximity"]], len(categories))
        x.extend(ocean_vec)

        y = [float(row["median_house_value"])]
        X.append(x)
        Y.append(y)

    meta = {
        "feature_names": numeric_features + [f"ocean_{c}" for c in categories],
        "target_name": "median_house_value",
        "categories": categories
    }
    return X, Y, meta

def build_classification_dataset(rows):
    """
    Wejście:
      cechy numeryczne + median_house_value
    Wyjście:
      ocean_proximity
    """
    numeric_features = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "median_house_value"
    ]

    rows, _ = impute_numeric_means(rows, numeric_features)

    categories = sorted(list(set(row["ocean_proximity"] for row in rows)))
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    X = []
    Y = []

    for row in rows:
        x = [float(row[col]) for col in numeric_features]
        y = one_hot(cat_to_idx[row["ocean_proximity"]], len(categories))
        X.append(x)
        Y.append(y)

    meta = {
        "feature_names": numeric_features,
        "target_name": "ocean_proximity",
        "categories": categories
    }
    return X, Y, meta

def fit_standardizer(X):
    cols = len(X[0])
    means = []
    stds = []

    for j in range(cols):
        vals = [row[j] for row in X]
        m = mean(vals)
        s = std(vals)
        means.append(m)
        stds.append(s)

    return means, stds

def transform_standardize(X, means, stds):
    Xn = []
    for row in X:
        r = []
        for j, v in enumerate(row):
            r.append((v - means[j]) / stds[j])
        Xn.append(r)
    return Xn

def fit_target_scaler_regression(Y):
    vals = [y[0] for y in Y]
    m = mean(vals)
    s = std(vals)
    return m, s

def transform_target_regression(Y, m, s):
    return [[(y[0] - m) / s] for y in Y]

def inverse_transform_value_regression(v, m, s):
    return v * s + m

# ============================================================
#  AKTYWACJE
# ============================================================

def relu(x):
    return x if x > 0 else 0.0

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def sigmoid(x):
    # zabezpieczenie przed overflow
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative_from_output(out):
    return out * (1.0 - out)

def tanh(x):
    return math.tanh(x)

def tanh_derivative_from_output(out):
    return 1.0 - out * out

def softmax(z):
    max_z = max(z)
    exps = [math.exp(v - max_z) for v in z]
    s = sum(exps)
    return [e / s for e in exps]

# ============================================================
#  SIEĆ NEURONOWA OD ZERA
# ============================================================

class MLP:
    def __init__(self, layer_sizes, task="regression", activation="relu", seed=1):
        """
        layer_sizes np. [13, 16, 8, 1] albo [9, 16, 5]
        """
        self.layer_sizes = layer_sizes
        self.task = task
        self.activation_name = activation
        self.rnd = random.Random(seed)

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            inp = layer_sizes[i]
            out = layer_sizes[i + 1]

            # małe losowe wagi
            limit = math.sqrt(6.0 / (inp + out))
            W = []
            for _ in range(out):
                row = [self.rnd.uniform(-limit, limit) for _ in range(inp)]
                W.append(row)

            b = [0.0 for _ in range(out)]

            self.weights.append(W)
            self.biases.append(b)

    def hidden_activate(self, x):
        if self.activation_name == "relu":
            return relu(x)
        elif self.activation_name == "sigmoid":
            return sigmoid(x)
        elif self.activation_name == "tanh":
            return tanh(x)
        else:
            raise ValueError("Nieznana aktywacja")

    def hidden_derivative_from_output(self, out, pre_act=None):
        if self.activation_name == "relu":
            # dla ReLU potrzebujemy info o wartości przed aktywacją
            return 1.0 if pre_act is not None and pre_act > 0 else 0.0
        elif self.activation_name == "sigmoid":
            return sigmoid_derivative_from_output(out)
        elif self.activation_name == "tanh":
            return tanh_derivative_from_output(out)
        else:
            raise ValueError("Nieznana aktywacja")

    def forward(self, x):
        """
        Zwraca:
          activations - aktywacje kolejnych warstw
          pre_activations - wartości przed aktywacją
        """
        activations = [x[:]]
        pre_activations = []

        a = x[:]

        # warstwy ukryte
        for layer_idx in range(len(self.weights) - 1):
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]

            z = []
            a_next = []
            for i in range(len(W)):
                s = b[i]
                for j in range(len(a)):
                    s += W[i][j] * a[j]
                z.append(s)
                a_next.append(self.hidden_activate(s))

            pre_activations.append(z)
            activations.append(a_next)
            a = a_next

        # warstwa wyjściowa
        W = self.weights[-1]
        b = self.biases[-1]
        z = []
        for i in range(len(W)):
            s = b[i]
            for j in range(len(a)):
                s += W[i][j] * a[j]
            z.append(s)

        pre_activations.append(z)

        if self.task == "regression":
            out = z[:]  # liniowa
        elif self.task == "classification":
            out = softmax(z)
        else:
            raise ValueError("Nieznane zadanie")

        activations.append(out)
        return activations, pre_activations

    def predict_one(self, x):
        activations, _ = self.forward(x)
        return activations[-1]

    def backward(self, x, y, lr):
        activations, pre_activations = self.forward(x)

        # delty dla każdej warstwy
        deltas = [None] * len(self.weights)

        # wyjście
        out = activations[-1]

        if self.task == "regression":
            # MSE: dL/dout = out - y
            delta_out = [(out[i] - y[i]) for i in range(len(out))]
        else:
            # softmax + cross-entropy
            delta_out = [(out[i] - y[i]) for i in range(len(out))]

        deltas[-1] = delta_out

        # warstwy ukryte
        for layer_idx in range(len(self.weights) - 2, -1, -1):
            W_next = self.weights[layer_idx + 1]
            delta_next = deltas[layer_idx + 1]
            a_current = activations[layer_idx + 1]
            z_current = pre_activations[layer_idx]

            delta = [0.0] * len(a_current)

            for i in range(len(a_current)):
                s = 0.0
                for k in range(len(delta_next)):
                    s += W_next[k][i] * delta_next[k]

                deriv = self.hidden_derivative_from_output(a_current[i], z_current[i])
                delta[i] = s * deriv

            deltas[layer_idx] = delta

        # aktualizacja wag
        for layer_idx in range(len(self.weights)):
            a_prev = activations[layer_idx]
            delta = deltas[layer_idx]

            for i in range(len(self.weights[layer_idx])):
                for j in range(len(self.weights[layer_idx][i])):
                    self.weights[layer_idx][i][j] -= lr * delta[i] * a_prev[j]
                self.biases[layer_idx][i] -= lr * delta[i]

    def fit(self, X_train, Y_train, epochs=10, lr=0.001, seed=1, verbose=True):
        rnd = random.Random(seed)

        for epoch in range(1, epochs + 1):
            idx = list(range(len(X_train)))
            rnd.shuffle(idx)

            for i in idx:
                self.backward(X_train[i], Y_train[i], lr)

            if verbose:
                if self.task == "regression":
                    train_pred = self.predict(X_train)
                    mse = mse_regression(Y_train, train_pred)
                    print(f"  epoka {epoch}/{epochs} | train MSE = {mse:.6f}")
                else:
                    train_pred = self.predict(X_train)
                    acc = accuracy_classification(Y_train, train_pred)
                    print(f"  epoka {epoch}/{epochs} | train ACC = {acc:.4f}")

    def predict(self, X):
        return [self.predict_one(x) for x in X]

# ============================================================
#  METRYKI
# ============================================================

def mse_regression(Y_true, Y_pred):
    s = 0.0
    for yt, yp in zip(Y_true, Y_pred):
        s += (yt[0] - yp[0]) ** 2
    return s / len(Y_true)

def rmse_regression(Y_true, Y_pred):
    return math.sqrt(mse_regression(Y_true, Y_pred))

def mae_regression(Y_true, Y_pred):
    s = 0.0
    for yt, yp in zip(Y_true, Y_pred):
        s += abs(yt[0] - yp[0])
    return s / len(Y_true)

def r2_regression(Y_true, Y_pred):
    y_vals = [yt[0] for yt in Y_true]
    y_mean = mean(y_vals)

    ss_res = 0.0
    ss_tot = 0.0

    for yt, yp in zip(Y_true, Y_pred):
        ss_res += (yt[0] - yp[0]) ** 2
        ss_tot += (yt[0] - y_mean) ** 2

    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)

def argmax(v):
    best_i = 0
    best_v = v[0]
    for i in range(1, len(v)):
        if v[i] > best_v:
            best_v = v[i]
            best_i = i
    return best_i

def accuracy_classification(Y_true, Y_pred):
    correct = 0
    for yt, yp in zip(Y_true, Y_pred):
        if argmax(yt) == argmax(yp):
            correct += 1
    return correct / len(Y_true)

# ============================================================
#  EKSPERYMENTY
# ============================================================

def summarize_regression_results(results):
    avg_train_rmse = mean([r["train_rmse"] for r in results])
    avg_test_rmse = mean([r["test_rmse"] for r in results])
    avg_train_mae = mean([r["train_mae"] for r in results])
    avg_test_mae = mean([r["test_mae"] for r in results])
    avg_train_r2 = mean([r["train_r2"] for r in results])
    avg_test_r2 = mean([r["test_r2"] for r in results])

    best = min(results, key=lambda r: r["test_rmse"])

    return {
        "avg_train_rmse": avg_train_rmse,
        "avg_test_rmse": avg_test_rmse,
        "avg_train_mae": avg_train_mae,
        "avg_test_mae": avg_test_mae,
        "avg_train_r2": avg_train_r2,
        "avg_test_r2": avg_test_r2,
        "best_test_rmse": best["test_rmse"],
        "best_test_mae": best["test_mae"],
        "best_test_r2": best["test_r2"],
        "best_seed": best["seed"]
    }

def summarize_classification_results(results):
    avg_train_acc = mean([r["train_acc"] for r in results])
    avg_test_acc = mean([r["test_acc"] for r in results])

    best = max(results, key=lambda r: r["test_acc"])

    return {
        "avg_train_acc": avg_train_acc,
        "avg_test_acc": avg_test_acc,
        "best_test_acc": best["test_acc"],
        "best_seed": best["seed"]
    }

def run_regression_experiment(rows, config, repeats=3):
    print("\n" + "=" * 70)
    print("REGRESJA | konfiguracja:", config)

    X, Y, meta = build_regression_dataset(rows)
    results = []

    for repeat in range(repeats):
        seed = config["seed_base"] + repeat

        # podział
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, TEST_RATIO, seed=seed)

        # standaryzacja X na podstawie treningu
        x_mean, x_std = fit_standardizer(X_train)
        X_train_n = transform_standardize(X_train, x_mean, x_std)
        X_test_n = transform_standardize(X_test, x_mean, x_std)

        # skalowanie Y dla regresji
        y_mean, y_std = fit_target_scaler_regression(Y_train)
        Y_train_n = transform_target_regression(Y_train, y_mean, y_std)
        Y_test_n = transform_target_regression(Y_test, y_mean, y_std)

        layer_sizes = [len(X_train_n[0])] + config["hidden_layers"] + [1]

        model = MLP(
            layer_sizes=layer_sizes,
            task="regression",
            activation=config["activation"],
            seed=seed
        )

        print(f"\nPowtórzenie {repeat + 1}/{repeats}, seed={seed}")
        model.fit(
            X_train_n,
            Y_train_n,
            epochs=config["epochs"],
            lr=config["lr"],
            seed=seed,
            verbose=config["verbose"]
        )

        # predykcja na skali znormalizowanej
        train_pred_n = model.predict(X_train_n)
        test_pred_n = model.predict(X_test_n)

        # odwrócenie skali Y
        train_pred = [[inverse_transform_value_regression(p[0], y_mean, y_std)] for p in train_pred_n]
        test_pred = [[inverse_transform_value_regression(p[0], y_mean, y_std)] for p in test_pred_n]

        train_rmse = rmse_regression(Y_train, train_pred)
        test_rmse = rmse_regression(Y_test, test_pred)
        train_mae = mae_regression(Y_train, train_pred)
        test_mae = mae_regression(Y_test, test_pred)
        train_r2 = r2_regression(Y_train, train_pred)
        test_r2 = r2_regression(Y_test, test_pred)

        results.append({
            "seed": seed,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2
        })

        print(f"Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
        print(f"Train MAE : {train_mae:.2f} | Test MAE : {test_mae:.2f}")
        print(f"Train R^2 : {train_r2:.4f} | Test R^2 : {test_r2:.4f}")

    summary = summarize_regression_results(results)

    print("\n--- PODSUMOWANIE REGRESJI ---")
    print(f"Średni Train RMSE: {summary['avg_train_rmse']:.2f}")
    print(f"Średni Test  RMSE: {summary['avg_test_rmse']:.2f}")
    print(f"Średni Train MAE : {summary['avg_train_mae']:.2f}")
    print(f"Średni Test  MAE : {summary['avg_test_mae']:.2f}")
    print(f"Średni Train R^2 : {summary['avg_train_r2']:.4f}")
    print(f"Średni Test  R^2 : {summary['avg_test_r2']:.4f}")
    print(f"Najlepszy Test RMSE: {summary['best_test_rmse']:.2f} (seed={summary['best_seed']})")
    print(f"Najlepszy Test MAE : {summary['best_test_mae']:.2f}")
    print(f"Najlepszy Test R^2 : {summary['best_test_r2']:.4f}")

    return summary

def run_classification_experiment(rows, config, repeats=3):
    print("\n" + "=" * 70)
    print("KLASYFIKACJA | konfiguracja:", config)

    X, Y, meta = build_classification_dataset(rows)
    results = []

    for repeat in range(repeats):
        seed = config["seed_base"] + repeat

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, TEST_RATIO, seed=seed)

        x_mean, x_std = fit_standardizer(X_train)
        X_train_n = transform_standardize(X_train, x_mean, x_std)
        X_test_n = transform_standardize(X_test, x_mean, x_std)

        layer_sizes = [len(X_train_n[0])] + config["hidden_layers"] + [len(Y_train[0])]

        model = MLP(
            layer_sizes=layer_sizes,
            task="classification",
            activation=config["activation"],
            seed=seed
        )

        print(f"\nPowtórzenie {repeat + 1}/{repeats}, seed={seed}")
        model.fit(
            X_train_n,
            Y_train,
            epochs=config["epochs"],
            lr=config["lr"],
            seed=seed,
            verbose=config["verbose"]
        )

        train_pred = model.predict(X_train_n)
        test_pred = model.predict(X_test_n)

        train_acc = accuracy_classification(Y_train, train_pred)
        test_acc = accuracy_classification(Y_test, test_pred)

        results.append({
            "seed": seed,
            "train_acc": train_acc,
            "test_acc": test_acc
        })

        print(f"Train ACC: {train_acc:.4f} | Test ACC: {test_acc:.4f}")

    summary = summarize_classification_results(results)

    print("\n--- PODSUMOWANIE KLASYFIKACJI ---")
    print(f"Średni Train ACC: {summary['avg_train_acc']:.4f}")
    print(f"Średni Test  ACC: {summary['avg_test_acc']:.4f}")
    print(f"Najlepszy Test ACC: {summary['best_test_acc']:.4f} (seed={summary['best_seed']})")

    return summary

# ============================================================
#  GŁÓWNY PROGRAM
# ============================================================

def main():
    rows = load_housing_data(CSV_FILE, max_rows=MAX_ROWS)
    print(f"Wczytano rekordów: {len(rows)}")

    # --------------------------------------------------------
    # ZESTAWY PARAMETRÓW DO TESTÓW
    # --------------------------------------------------------
    # Możesz dodać więcej konfiguracji, jeśli prowadzący wymaga
    # większej liczby przetestowanych parametrów.
    regression_configs = [
        {
            "hidden_layers": [16],
            "activation": "relu",
            "lr": 0.001,
            "epochs": 10,
            "seed_base": 100,
            "verbose": False
        },
        {
            "hidden_layers": [32, 16],
            "activation": "relu",
            "lr": 0.001,
            "epochs": 12,
            "seed_base": 200,
            "verbose": False
        },
        {
            "hidden_layers": [16, 8],
            "activation": "tanh",
            "lr": 0.0007,
            "epochs": 15,
            "seed_base": 300,
            "verbose": False
        }
    ]

    classification_configs = [
        {
            "hidden_layers": [16],
            "activation": "relu",
            "lr": 0.001,
            "epochs": 10,
            "seed_base": 400,
            "verbose": False
        },
        {
            "hidden_layers": [32, 16],
            "activation": "relu",
            "lr": 0.001,
            "epochs": 12,
            "seed_base": 500,
            "verbose": False
        },
        {
            "hidden_layers": [16, 8],
            "activation": "tanh",
            "lr": 0.0007,
            "epochs": 15,
            "seed_base": 600,
            "verbose": False
        }
    ]

    # --------------------------------------------------------
    # REGRESJA
    # --------------------------------------------------------
    print("\n" + "#" * 70)
    print("BADANIE 1: REGRESJA - przewidywanie median_house_value")
    print("#" * 70)

    regression_summaries = []
    for cfg in regression_configs:
        summary = run_regression_experiment(rows, cfg, repeats=REPEATS)
        regression_summaries.append((cfg, summary))

    # --------------------------------------------------------
    # KLASYFIKACJA
    # --------------------------------------------------------
    print("\n" + "#" * 70)
    print("BADANIE 2: KLASYFIKACJA - przewidywanie ocean_proximity")
    print("#" * 70)

    classification_summaries = []
    for cfg in classification_configs:
        summary = run_classification_experiment(rows, cfg, repeats=REPEATS)
        classification_summaries.append((cfg, summary))

    # --------------------------------------------------------
    # KOŃCOWE PORÓWNANIE
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("KOŃCOWE PORÓWNANIE KONFIGURACJI")
    print("=" * 70)

    print("\nREGRESJA:")
    for i, (cfg, summ) in enumerate(regression_summaries, 1):
        print(f"{i}. {cfg}")
        print(f"   avg test RMSE = {summ['avg_test_rmse']:.2f}, avg test MAE = {summ['avg_test_mae']:.2f}, avg test R^2 = {summ['avg_test_r2']:.4f}")
        print(f"   best test RMSE = {summ['best_test_rmse']:.2f}")

    print("\nKLASYFIKACJA:")
    for i, (cfg, summ) in enumerate(classification_summaries, 1):
        print(f"{i}. {cfg}")
        print(f"   avg test ACC = {summ['avg_test_acc']:.4f}")
        print(f"   best test ACC = {summ['best_test_acc']:.4f}")

if __name__ == "__main__":
    main()