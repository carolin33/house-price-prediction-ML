import csv
import math
import random
from copy import deepcopy
import os
from datetime import datetime

# ============================================================
#  USTAWIENIA GŁÓWNE
# ============================================================

CSV_FILE = "housing.csv"
MAX_ROWS = None  # np. 500 jeśli chcesz szybciej testować; None = cały zbiór

# Baza startowa (Krok 0 – Zgrubna kalibracja)
BASELINE = {
    "test_ratio": 0.2,
    "hidden_layers": [32],
    "activation": "relu",
    "weight_init_scale": 1.0,
    "lr": 0.001,
    "epochs": 100,
    "repeats": 5,
    "seed_base": 42,
    "verbose": False,
}

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

def fit_standardizer(X):
    n = len(X)
    cols = len(X[0])
    means = []
    stds = []

    for j in range(cols):
        # Oblicz srednia i odchylenie dla kolumny j
        vals = [row[j] for row in X]
        m = sum(vals) / n
        var = sum((v - m) ** 2 for v in vals) / n
        s = math.sqrt(var) if var > 1e-24 else 1.0
        means.append(m)
        stds.append(s)

    return means, stds

def transform_standardize(X, means, stds):
    # zip(means, stds) laczy pary (srednia, odchylenie) dla kazdej cechy,
    # co pozwala uniknac indeksowania w wewnetrznej petli
    ms = list(zip(means, stds))
    return [[(v - m) / s for v, (m, s) in zip(row, ms)] for row in X]

def fit_target_scaler_regression(Y):
    vals = [y[0] for y in Y]
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    s = math.sqrt(var) if var > 1e-24 else 1.0
    return m, s

def transform_target_regression(Y, m, s):
    return [[(y[0] - m) / s] for y in Y]

def inverse_transform_value_regression(v, m, s):
    return v * s + m

# ============================================================
#  AKTYWACJE
# ============================================================
#
#  Kazda aktywacja ma dwie funkcje:
#    - act(x)    -> wartosc po aktywacji
#    - deriv(out, pre_act) -> pochodna (potrzebna w backpropagation)
#
#  Uzywamy par (act_fn, deriv_fn) zamiast lancucha if/elif,
#  dzieki czemu Python wywoluje funkcje bezposrednio
#  bez sprawdzania nazwy przy kazdym neuronie.

_math_exp = math.exp    # lokalna referencja – szybszy dostep niz math.exp
_math_tanh = math.tanh

def _relu(x):
    return x if x > 0.0 else 0.0

def _relu_deriv(out, pre_act):
    return 1.0 if pre_act > 0.0 else 0.0

def _sigmoid(x):
    # zabezpieczenie przed overflow exp()
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + _math_exp(-x))

def _sigmoid_deriv(out, pre_act):
    # pochodna sigmoidy wyrazona przez wartosc wyjsciowa: sig * (1 - sig)
    return out * (1.0 - out)

def _tanh(x):
    return _math_tanh(x)

def _tanh_deriv(out, pre_act):
    # pochodna tanh wyrazona przez wyjscie: 1 - tanh^2
    return 1.0 - out * out

def _leaky_relu(x):
    return x if x > 0.0 else 0.01 * x

def _leaky_relu_deriv(out, pre_act):
    return 1.0 if pre_act > 0.0 else 0.01

# Slownik mapujacy nazwe aktywacji na pare (funkcja, pochodna).
# Dzieki temu w __init__ wystarczy jedno wyszukanie zamiast
# powtarzanych if/elif przy kazdym neuronie.
_ACTIVATION_REGISTRY = {
    "relu":       (_relu, _relu_deriv),
    "sigmoid":    (_sigmoid, _sigmoid_deriv),
    "tanh":       (_tanh, _tanh_deriv),
    "leaky_relu": (_leaky_relu, _leaky_relu_deriv),
}

# ============================================================
#  SIEC NEURONOWA OD ZERA (TYLKO REGRESJA)
# ============================================================

class MLP:
    def __init__(self, layer_sizes, activation="relu",
                 seed=1, weight_init_scale=1.0):
        """
        layer_sizes np. [13, 16, 8, 1]
        weight_init_scale - mnoznik limitu Xavier (domyslnie 1.0)
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.rnd = random.Random(seed)

        # Pobierz funkcje aktywacji raz, zamiast sprawdzac nazwe
        # przy kazdym wywolaniu neuronu (eliminuje ~500k if/elif na eksperyment)
        if activation not in _ACTIVATION_REGISTRY:
            raise ValueError(f"Nieznana aktywacja: {activation}")
        self._act_fn, self._deriv_fn = _ACTIVATION_REGISTRY[activation]

        self.weights = []
        self.biases = []

        _sqrt = math.sqrt
        _uniform = self.rnd.uniform
        for i in range(len(layer_sizes) - 1):
            inp = layer_sizes[i]
            out = layer_sizes[i + 1]

            # male losowe wagi - limit Xavier z mnoznikiem
            limit = _sqrt(6.0 / (inp + out)) * weight_init_scale
            W = [[_uniform(-limit, limit) for _ in range(inp)]
                 for _ in range(out)]

            b = [0.0] * out

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        """
        Propagacja w przod (forward pass).
        Zwraca:
          activations     - lista aktywacji kolejnych warstw
          pre_activations - lista wartosci PRZED aktywacja (potrzebne do backprop)
        """
        # Zapisujemy referencje do czesto uzywanych obiektow jako zmienne lokalne.
        # W Pythonie dostep do zmiennej lokalnej jest ~2x szybszy niz self.atrybut,
        # bo omija mechanizm wyszukiwania atrybutow obiektu.
        weights = self.weights
        biases = self.biases
        act_fn = self._act_fn
        n_hidden = len(weights) - 1   # liczba warstw ukrytych

        activations = [x]
        pre_activations = []
        a = x

        # --- warstwy ukryte ---
        for layer_idx in range(n_hidden):
            W = weights[layer_idx]
            b = biases[layer_idx]

            z = []
            a_next = []
            for w_row, bi in zip(W, b):
                # Iloczyn skalarny wiersza wag i wektora aktywacji:
                # sum(w*a for w,a in zip(...)) uzywa wewnetrznej petli C Pythona,
                # co jest ~2-3x szybsze niz reczna petla for j in range(...)
                s = bi + sum(w * aj for w, aj in zip(w_row, a))
                z.append(s)
                a_next.append(act_fn(s))

            pre_activations.append(z)
            activations.append(a_next)
            a = a_next

        # --- warstwa wyjsciowa (liniowa aktywacja – regresja) ---
        W = weights[-1]
        b = biases[-1]
        z = [bi + sum(w * aj for w, aj in zip(w_row, a))
             for w_row, bi in zip(W, b)]

        pre_activations.append(z)
        activations.append(z[:])  # kopia – warstwa wyjsciowa jest liniowa
        return activations, pre_activations

    def predict_one(self, x):
        """
        Szybka predykcja jednej probki (bez zapamietywania warstw posrednich).
        Uzywana w predict() – nie potrzebujemy pre_activations.
        """
        weights = self.weights
        biases = self.biases
        act_fn = self._act_fn
        n_hidden = len(weights) - 1
        a = x

        # warstwy ukryte
        for layer_idx in range(n_hidden):
            W = weights[layer_idx]
            b = biases[layer_idx]
            a = [act_fn(bi + sum(w * aj for w, aj in zip(w_row, a)))
                 for w_row, bi in zip(W, b)]

        # warstwa wyjsciowa (liniowa)
        W = weights[-1]
        b = biases[-1]
        return [bi + sum(w * aj for w, aj in zip(w_row, a))
                for w_row, bi in zip(W, b)]

    def backward(self, x, y, lr):
        """
        Propagacja wsteczna (backpropagation) z aktualizacja wag.
        """
        activations, pre_activations = self.forward(x)

        weights = self.weights
        biases = self.biases
        deriv_fn = self._deriv_fn
        n_layers = len(weights)

        # --- Delty warstwy wyjsciowej ---
        # Dla MSE: dL/dout_i = out_i - y_i
        out = activations[-1]
        deltas = [None] * n_layers
        deltas[-1] = [oi - yi for oi, yi in zip(out, y)]

        # --- Propagacja delt wstecz przez warstwy ukryte ---
        for layer_idx in range(n_layers - 2, -1, -1):
            W_next = weights[layer_idx + 1]
            delta_next = deltas[layer_idx + 1]
            a_current = activations[layer_idx + 1]
            z_current = pre_activations[layer_idx]

            # zip(*W_next) "transponuje" macierz wag:
            # jesli W_next ma wiersze [w0, w1, ...] to zip(*W_next)
            # daje kolumny – i-ta kolumna zawiera wagi prowadzace
            # od i-tego neuronu biezacej warstwy do wszystkich neuronow nastepnej.
            # Dzieki temu obliczamy sume delta_k * W_next[k][i] jako
            # iloczyn skalarny zamiast petli po k.
            delta = []
            for col, ai, zi in zip(zip(*W_next), a_current, z_current):
                grad_sum = sum(wk * dk for wk, dk in zip(col, delta_next))
                delta.append(grad_sum * deriv_fn(ai, zi))

            deltas[layer_idx] = delta

        # --- Aktualizacja wag i biasow ---
        for layer_idx in range(n_layers):
            a_prev = activations[layer_idx]
            delta = deltas[layer_idx]
            W = weights[layer_idx]
            b = biases[layer_idx]

            for i, di in enumerate(delta):
                # Mnozenie lr * delta_i obliczamy RAZ i uzywamy wielokrotnie
                # dla wszystkich wag w wierszu (oszczedza N mnozen na neuron)
                lr_di = lr * di
                w_row = W[i]
                for j, aj in enumerate(a_prev):
                    w_row[j] -= lr_di * aj
                b[i] -= lr_di

    def fit(self, X_train, Y_train, epochs=10, lr=0.001, seed=1, verbose=True):
        rnd = random.Random(seed)
        # Referencja lokalna do metody backward – eliminuje
        # wyszukiwanie atrybutu self.backward w kazdej iteracji
        _backward = self.backward
        n = len(X_train)

        for epoch in range(1, epochs + 1):
            idx = list(range(n))
            rnd.shuffle(idx)

            for i in idx:
                _backward(X_train[i], Y_train[i], lr)

            if verbose:
                train_pred = self.predict(X_train)
                mse = mse_regression(Y_train, train_pred)
                print(f"  epoka {epoch}/{epochs} | train MSE = {mse:.6f}")

    def predict(self, X):
        # Referencja lokalna – szybszy dostep w petli
        _predict_one = self.predict_one
        return [_predict_one(x) for x in X]

# ============================================================
#  METRYKI (TYLKO REGRESJA)
# ============================================================

def mse_regression(Y_true, Y_pred):
    # sum() z generatorem jest szybsze niz reczna akumulacja w petli
    n = len(Y_true)
    return sum((yt[0] - yp[0]) ** 2 for yt, yp in zip(Y_true, Y_pred)) / n

def rmse_regression(Y_true, Y_pred):
    return math.sqrt(mse_regression(Y_true, Y_pred))

def mae_regression(Y_true, Y_pred):
    n = len(Y_true)
    return sum(abs(yt[0] - yp[0]) for yt, yp in zip(Y_true, Y_pred)) / n

def r2_regression(Y_true, Y_pred):
    y_vals = [yt[0] for yt in Y_true]
    y_mean = sum(y_vals) / len(y_vals)

    ss_res = 0.0
    ss_tot = 0.0

    for yt, yp in zip(Y_true, Y_pred):
        diff = yt[0] - yp[0]
        ss_res += diff * diff          # diff*diff szybsze niz diff**2
        dev = yt[0] - y_mean
        ss_tot += dev * dev

    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)

# ============================================================
#  EKSPERYMENTY – REGRESJA
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

def run_regression_experiment(rows, config):
    """
    Uruchamia eksperyment regresji z polnym configiem
    (config zawiera: epochs, lr, hidden_layers, activation,
     test_ratio, weight_init_scale, repeats, seed_base, verbose).
    """
    X, Y, meta = build_regression_dataset(rows)
    results = []
    repeats = config["repeats"]
    test_ratio = config["test_ratio"]

    for repeat in range(repeats):
        seed = config["seed_base"] + repeat

        # podział
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_ratio, seed=seed)

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
            activation=config["activation"],
            seed=seed,
            weight_init_scale=config.get("weight_init_scale", 1.0)
        )

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

    summary = summarize_regression_results(results)
    return summary

# ============================================================
#  SELEKCJA MODELU Z OCHRONA PRZED OVERFITTINGIEM
# ============================================================

def wybierz_najlepszy_model(wyniki, prog_r2=0.015):
    """
    Wybiera najlepszy model z listy wynikow, odporny na overfitting.

    Parametry:
        wyniki  - lista slownikow, kazdy zawiera klucze:
                  'param_value', 'test_rmse', 'train_r2', 'test_r2'
        prog_r2 - maksymalna dopuszczalna roznica (train_r2 - test_r2)
                  powyzej ktorej model uznajemy za przeuczony (domyslnie 0.03)

    Logika:
        1. Filtruj modele "stabilne":  (train_r2 - test_r2) <= prog_r2
        2. Sposrod stabilnych wybierz ten z najnizszym test_rmse.
        3. Jesli zaden model nie jest stabilny (fallback),
           wybierz model z najmniejsza roznica (train_r2 - test_r2)
           – "najmniejsze zlo".

    Zwraca:
        Slownik zwycieskiego modelu (jeden element z listy 'wyniki').
    """
    # Oblicz roznice R^2 dla kazdego modelu
    for w in wyniki:
        w["_r2_gap"] = w["train_r2"] - w["test_r2"]

    # Filtruj stabilne modele (nieprzeuczone)
    stabilne = [w for w in wyniki if w["_r2_gap"] <= prog_r2]

    if stabilne:
        # Sposrod stabilnych -> najnizsze test_rmse
        zwyciezca = min(stabilne, key=lambda w: w["test_rmse"])
    else:
        # Fallback: zaden nie spelnia progu -> najmniejsza roznica R^2
        zwyciezca = min(wyniki, key=lambda w: w["_r2_gap"])

    # Posprzataj tymczasowy klucz
    for w in wyniki:
        w.pop("_r2_gap", None)

    return zwyciezca

# ============================================================
#  EKSPORT WYNIKÓW DO CSV
# ============================================================

def export_step_csv(step_results, filepath):
    """
    Zapisuje wyniki jednego kroku optymalizacji do pliku CSV.
    step_results to lista słowników z wynikami.
    """
    if not step_results:
        return

    fieldnames = list(step_results[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in step_results:
            writer.writerow(row)

    print(f"  >> Zapisano: {filepath}")

def export_master_csv(all_results, filepath):
    """
    Zapisuje zbiorczy plik CSV ze wszystkimi krokami optymalizacji.
    """
    if not all_results:
        return

    fieldnames = list(all_results[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\n>>> Zbiorczy plik zapisano do: {filepath}")

# ============================================================
#  GENEROWANIE TOPOLOGII DLA KROKU 3
# ============================================================

def generate_topologies(winning_depth):
    """
    Na podstawie wygranej głębokości (liczba warstw ukrytych)
    generuje 5 różnych topologii (szerokości) do przetestowania.
    """
    n_layers = len(winning_depth)

    if n_layers == 0:
        # Perceptron – brak warstw ukrytych, nie da się zmieniać szerokości
        return [[]]

    if n_layers == 1:
        return [[4], [8], [16], [32], [64]]

    if n_layers == 2:
        return [[8, 4], [16, 8], [32, 16], [64, 32], [32, 32]]

    if n_layers == 3:
        return [[8, 4, 2], [16, 8, 4], [32, 16, 8], [64, 32, 16], [32, 32, 32]]

    if n_layers == 4:
        return [[8, 4, 4, 2], [16, 8, 4, 2], [32, 16, 8, 4], [64, 32, 16, 8], [32, 32, 32, 32]]

    # fallback dla n_layers > 4
    topologies = []
    for base in [8, 16, 32, 64, 48]:
        topo = []
        for i in range(n_layers):
            width = max(2, base // (2 ** i))
            topo.append(width)
        topologies.append(topo)
    return topologies

# ============================================================
#  TWORZENIE FOLDERU Z WYNIKAMI
# ============================================================

def create_results_folder():
    """
    Tworzy unikalny folder wynikowy: wyniki_YYYYMMDD_HHMMSS/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"wyniki_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f"  FOLDER WYNIKÓW: {folder_name}/")
    print(f"{'=' * 70}")
    return folder_name

# ============================================================
#  GŁÓWNY PROGRAM – OPTYMALIZACJA SEKWENCYJNA
# ============================================================

def main():
    rows = load_housing_data(CSV_FILE, max_rows=MAX_ROWS)
    print(f"Wczytano rekordów: {len(rows)}")

    # Tworzenie folderu wynikowego
    results_folder = create_results_folder()

    # Kopia robocza baseline – będzie aktualizowana po każdym kroku
    baseline = deepcopy(BASELINE)

    # Zbiorczy log wszystkich eksperymentów
    master_log = []

    # =========================================================
    #  DEFINICJA KROKÓW OPTYMALIZACJI (8 kroków)
    # =========================================================
    optimization_steps = [
        {
            "step_num": 1,
            "name": "test_ratio",
            "display": "Wielkość próby testowej (test_ratio)",
            "param_key": "test_ratio",
            "values": [0.1, 0.15, 0.2, 0.3, 0.4],
        },
        {
            "step_num": 2,
            "name": "depth",
            "display": "Głębokość sieci (liczba warstw ukrytych)",
            "param_key": "hidden_layers",
            "values": [[], [16], [16, 16], [16, 16, 16], [16, 16, 16, 16]],
        },
        {
            "step_num": 3,
            "name": "topology",
            "display": "Szerokość / Kształt sieci (topology)",
            "param_key": "hidden_layers",
            "values": None,  # zostanie wygenerowane dynamicznie
        },
        {
            "step_num": 4,
            "name": "activation",
            "display": "Funkcja aktywacji (activation)",
            "param_key": "activation",
            "values": ["relu", "leaky_relu", "tanh", "sigmoid"],
        },
        {
            "step_num": 5,
            "name": "weight_init_scale",
            "display": "Skala inicjalizacji wag (weight_init_scale)",
            "param_key": "weight_init_scale",
            "values": [0.01, 0.1, 1.0, 2.0, 5.0],
        },
        {
            "step_num": 6,
            "name": "lr",
            "display": "Tempo uczenia – fine tuning (lr)",
            "param_key": "lr",
            "values": [0.01, 0.005, 0.001, 0.0005, 0.0001],
        },
        {
            "step_num": 7,
            "name": "epochs",
            "display": "Moment zatrzymania (epochs)",
            "param_key": "epochs",
            "values": [10, 30, 50, 100, 200],
        },
        {
            "step_num": 8,
            "name": "repeats",
            "display": "Stabilność Super-Modelu (repeats)",
            "param_key": "repeats",
            "values": [1, 3, 5, 10],
        },
    ]

    total_steps = len(optimization_steps)

    # =========================================================
    #  PĘTLA OPTYMALIZACJI SEKWENCYJNEJ
    # =========================================================
    for step in optimization_steps:
        step_num = step["step_num"]
        step_name = step["name"]
        step_display = step["display"]
        param_key = step["param_key"]

        # Krok 3 – dynamiczne generowanie topologii
        if step_name == "topology":
            winning_depth = baseline["hidden_layers"]
            step["values"] = generate_topologies(winning_depth)
            print(f"\n  [Krok 3] Wygrana głębokość z kroku 2: {winning_depth}")
            print(f"  [Krok 3] Generuję topologie do testowania: {step['values']}")

        values_to_test = step["values"]

        print("\n" + "#" * 70)
        print(f"  KROK {step_num}/{total_steps}: {step_display}")
        print(f"  Aktualny BASELINE: {_baseline_summary(baseline)}")
        print(f"  Wartości do testowania: {values_to_test}")
        print("#" * 70)

        step_csv_rows = []
        candidates = []  # lista kandydatow do wybierz_najlepszy_model

        for val_idx, param_value in enumerate(values_to_test):
            # Zbuduj config z aktualnym baseline + testowana wartoscia
            config = deepcopy(baseline)
            config[param_key] = param_value

            display_value = str(param_value)

            print(f"\n>>> Krok {step_num}, wariant {val_idx + 1}/{len(values_to_test)}: "
                  f"{step_name} = {display_value}")

            summary = run_regression_experiment(rows, config)

            r2_gap = summary["avg_train_r2"] - summary["avg_test_r2"]

            # Wyswietl wyniki
            print(f"    Sr. Train RMSE: {summary['avg_train_rmse']:.2f} | "
                  f"Sr. Test RMSE: {summary['avg_test_rmse']:.2f}")
            print(f"    Sr. Train MAE:  {summary['avg_train_mae']:.2f} | "
                  f"Sr. Test MAE:  {summary['avg_test_mae']:.2f}")
            print(f"    Sr. Train R^2:  {summary['avg_train_r2']:.4f} | "
                  f"Sr. Test R^2:  {summary['avg_test_r2']:.4f} | "
                  f"Gap R^2: {r2_gap:.4f}")

            # Zbierz wiersz CSV
            csv_row = {
                "krok": step_num,
                "nazwa_kroku": step_name,
                "wartosc_parametru": display_value,
                "avg_train_rmse": f"{summary['avg_train_rmse']:.4f}",
                "avg_test_rmse": f"{summary['avg_test_rmse']:.4f}",
                "avg_train_mae": f"{summary['avg_train_mae']:.4f}",
                "avg_test_mae": f"{summary['avg_test_mae']:.4f}",
                "avg_train_r2": f"{summary['avg_train_r2']:.4f}",
                "avg_test_r2": f"{summary['avg_test_r2']:.4f}",
                "r2_gap": f"{r2_gap:.4f}",
                "best_test_rmse": f"{summary['best_test_rmse']:.4f}",
                "best_test_r2": f"{summary['best_test_r2']:.4f}",
                "best_seed": summary["best_seed"],
            }
            step_csv_rows.append(csv_row)
            master_log.append(csv_row)

            # Dodaj kandydata do selekcji anty-overfittingowej
            candidates.append({
                "param_value": param_value,
                "test_rmse": summary["avg_test_rmse"],
                "train_r2": summary["avg_train_r2"],
                "test_r2": summary["avg_test_r2"],
            })

        # ---- SELEKCJA Z OCHRONA PRZED OVERFITTINGIEM ----
        winner = wybierz_najlepszy_model(candidates, prog_r2=0.03)
        best_value = winner["param_value"]
        best_rmse = winner["test_rmse"]
        best_gap = winner["train_r2"] - winner["test_r2"]

        # Sprawdz czy uzyta byla sciezka fallback
        stabilne_count = sum(
            1 for c in candidates
            if (c["train_r2"] - c["test_r2"]) <= 0.03
        )
        if stabilne_count == 0:
            selection_mode = "FALLBACK (najmniejszy gap R^2)"
        else:
            selection_mode = f"STABILNY ({stabilne_count}/{len(candidates)} ponizej progu)"

        # ---- AKTUALIZACJA BASELINE ----
        print(f"\n{'-' * 60}")
        print(f"  KROK {step_num} ZAKONCZONY")
        print(f"  Tryb selekcji: {selection_mode}")
        print(f"  Najlepsza wartosc: {step_name} = {best_value}")
        print(f"  Test RMSE: {best_rmse:.2f} | Gap R^2: {best_gap:.4f}")
        print(f"  >>> Nadpisuje BASELINE: {param_key} = {best_value}")
        print(f"{'-' * 60}")

        baseline[param_key] = best_value

        # ---- ZAPIS CSV DLA TEGO KROKU ----
        step_filename = f"krok_{step_num:02d}_{step_name}.csv"
        step_filepath = os.path.join(results_folder, step_filename)
        export_step_csv(step_csv_rows, step_filepath)

    # =========================================================
    #  ZAPIS ZBIORCZEGO CSV
    # =========================================================
    master_filepath = os.path.join(results_folder, "wyniki_zbiorcze.csv")
    export_master_csv(master_log, master_filepath)

    # =========================================================
    #  ZAPIS FINALNEGO BASELINE DO CSV
    # =========================================================
    final_baseline_path = os.path.join(results_folder, "supermodel_config.csv")
    with open(final_baseline_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parametr", "wartosc"])
        for key, val in baseline.items():
            writer.writerow([key, str(val)])
    print(f"\n>>> Konfiguracje Super-Modelu zapisano do: {final_baseline_path}")

    # =========================================================
    #  PODSUMOWANIE KONCOWE
    # =========================================================
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + "  OSTATECZNA ARCHITEKTURA SUPER-MODELU".center(68) + "|")
    print("+" + "=" * 68 + "+")

    param_labels = {
        "test_ratio":        "Wielkosc proby testowej",
        "hidden_layers":     "Architektura warstw ukrytych",
        "activation":        "Funkcja aktywacji",
        "weight_init_scale": "Skala inicjalizacji wag",
        "lr":                "Tempo uczenia (learning rate)",
        "epochs":            "Liczba epok",
        "repeats":           "Liczba powtorzen",
        "seed_base":         "Bazowy seed",
        "verbose":           "Tryb verbose",
    }

    for key, val in baseline.items():
        label = param_labels.get(key, key)
        line = f"  {label:<40s} {str(val):>24s}"
        print(f"|{line}|")

    print("+" + "=" * 68 + "+")
    print(f"|{'  Folder wynikow: ' + results_folder:<68s}|")
    print("+" + "=" * 68 + "+")

    print("\n" + "=" * 70)
    print("WSZYSTKIE EKSPERYMENTY ZAKONCZONE")
    print("=" * 70)


def _baseline_summary(baseline):
    """Zwraca czytelny string z kluczowymi parametrami baseline."""
    return (
        f"lr={baseline['lr']}, "
        f"epochs={baseline['epochs']}, "
        f"layers={baseline['hidden_layers']}, "
        f"act={baseline['activation']}, "
        f"test_ratio={baseline['test_ratio']}, "
        f"w_init={baseline['weight_init_scale']}, "
        f"repeats={baseline['repeats']}"
    )


if __name__ == "__main__":
    main()