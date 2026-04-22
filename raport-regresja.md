# Projekt Elementy sztucznej inteligencji

## Wstęp i opis badanych problemów

W niniejszym projekcie porównano skuteczność wybranych metod uczenia maszynowego w zadaniu regresji. Do analizy wykorzystano zbiór danych California Housing, który jest często używany do testowania modeli predykcyjnych. Celem projektu było sprawdzenie, jak zmiany wybranych hiperparametrów wpływają na jakość działania modeli oraz ich zdolność do przewidywania wartości na nowych danych. Chodziło więc nie tylko o uzyskanie jak najlepszego wyniku, ale też o lepsze zrozumienie, jak zachowują się różne metody przy zmianie ustawień.

W projekcie porównano kilka metod:

- sztuczne sieci neuronowe (MLP),
- lasy losowe (Random Forest),
- algorytm k najbliższych sąsiadów (k-NN),
- Gradient Boosting,
- SVM.

Wszystkie te metody wykorzystano do rozwiązania problemu regresji, czyli przewidywania wartości zmiennej median_house_value. Zadanie polegało na estymacji mediany cen nieruchomości na podstawie dostępnych cech opisujących dany obszar. Nie jest to łatwy problem, bo ceny nieruchomości zależą od wielu różnych czynników, a zależności między nimi często są nieliniowe.

Do oceny jakości modeli wykorzystano miary MAE,RMSE oraz R². Dzięki nim można było sprawdzić, jak duży jest błąd przewidywań i jak dobrze model dopasowuje się do danych. 



### 1.1. Czym jest Random Forest 

Random Forest to model, który składa się z wielu drzew decyzyjnych. Zamiast opierać wynik na jednym drzewie, bierze się dużo drzew i łączy ich odpowiedzi. W regresji po prostu liczy się średnią z ich przewidywań. To działa głównie dlatego, że pojedyncze drzewo decyzyjne łatwo przesadza z dopasowaniem do danych treningowych. Może dobrze zapamiętać przykłady, ale potem gorzej radzi sobie na nowych danych. A kiedy takich drzew jest dużo, to błędy jednych często są niwelowane przez inne. Jedno drzewo może się pomylić mocniej, ale pozostałe trochę to wyrównają. Ważne jest też to, że te drzewa nie są budowane identycznie. Każde uczy się na trochę innym fragmencie danych i przy podziałach bierze pod uwagę tylko część cech. Dzięki temu nie powstaje sto kopii tego samego modelu, tylko wiele podobnych, ale jednak różnych drzew. No i właśnie przez to Random Forest zwykle działa lepiej niż pojedyncze drzewo. Jest bardziej odporny na przypadkowe błędy w danych, mniej się przeucza i daje bardziej stabilne wyniki. Nie dlatego, że każde drzewo jest idealne. 

## 1.2. Strojenie hiperparametrów modelu Random Forest

Strojenie modelu Random Forest wykonano metodą greedy. W praktyce oznaczało to, że w każdym kroku zmieniano tylko jeden hiperparametr, a pozostałe pozostawały bez zmian. Dzięki temu można było sprawdzić, jak dana zmiana wpływa na jakość modelu.

Do oceny wykorzystano trzy metryki:

- R² – pokazuje, jak dobrze model wyjaśnia zmienność danych,
- MAE – średni błąd bezwzględny,
- RMSE – pierwiastek z błędu średniokwadratowego.

### 1.2.1. Strojenie parametru `n_estimators`

Na początku sprawdzono wpływ liczby drzew w lesie. Testowano wartości od 50 do 500.

| `n_estimators` | R² | Odchylenie R² | MAE | RMSE | Uwagi |
|---|---:|---:|---:|---:|---|
| 50  | 0.8223 | 0.0068 | 31607 | 48621 | wynik bazowy |
| 100 | 0.8239 | 0.0078 | 31402 | 48407 | lepszy od poprzedniego |
| 200 | 0.8243 | 0.0076 | 31345 | 48354 | lekka poprawa |
| 300 | 0.8247 | 0.0079 | 31308 | 48296 | kolejna niewielka poprawa |
| 500 | 0.8248 | 0.0079 | 31298 | 48277 | najlepszy wynik |

Na podstawie tych wyników wybrano `n_estimators = 500`. Widać, że większa liczba drzew poprawiała wyniki, ale poprawa była już raczej niewielka.

### 1.2.2. Strojenie parametru `max_depth`

W kolejnym kroku sprawdzono maksymalną głębokość drzew.

| `max_depth` | R² | Odchylenie R² | MAE | RMSE | Uwagi |
|---|---:|---:|---:|---:|---|
| 5    | 0.6550 | 0.0131 | 48273 | 67757 | wynik wyraźnie słaby |
| 10   | 0.7896 | 0.0095 | 35881 | 52909 | duża poprawa |
| 15   | 0.8203 | 0.0079 | 31994 | 48902 | wynik już dużo lepszy |
| 25   | 0.8248 | 0.0079 | 31308 | 48285 | bardzo dobry wynik |
| None | 0.8248 | 0.0079 | 31298 | 48277 | najlepszy wynik |

Tutaj było widać bardzo wyraźnie, że zbyt mała głębokość pogarszała działanie modelu. Najlepszy wynik uzyskano dla `max_depth = None`, czyli bez ograniczenia głębokości.

### 1.2.3. Strojenie parametru `min_samples_split`

Następnie testowano minimalną liczbę próbek potrzebną do wykonania podziału.

| `min_samples_split` | R² | Odchylenie R² | MAE | RMSE | Uwagi |
|---|---:|---:|---:|---:|---|
| 2  | 0.8248 | 0.0079 | 31298 | 48277 | najlepszy wynik |
| 5  | 0.8246 | 0.0079 | 31366 | 48313 | prawie taki sam wynik |
| 10 | 0.8222 | 0.0084 | 31687 | 48631 | lekki spadek |
| 20 | 0.8160 | 0.0089 | 32483 | 49480 | wyraźnie gorzej |
| 50 | 0.7977 | 0.0091 | 34610 | 51883 | najsłabszy wynik |

Najlepsza okazała się wartość `min_samples_split = 2`. Wraz ze wzrostem tego parametru jakość modelu stopniowo spadała.

### 1.2.4. Strojenie parametru `max_features`

Na końcu sprawdzono, jaka liczba cech używanych przy podziale daje najlepsze wyniki.

| `max_features` | R² | Odchylenie R² | MAE | RMSE | Uwagi |
|---|---:|---:|---:|---:|---|
| 0.3 | 0.8208 | 0.0081 | 32562 | 48825 | słabszy wynik |
| 0.5 | 0.8285 | 0.0078 | 31324 | 47770 | najlepszy wynik |
| 0.7 | 0.8283 | 0.0079 | 31213 | 47794 | bardzo podobny wynik |
| 0.9 | 0.8270 | 0.0080 | 31200 | 47972 | lekki spadek |
| 1.0 | 0.8248 | 0.0079 | 31298 | 48277 | słabiej niż dla 0.5 i 0.7 |

Najlepszy wynik końcowy uzyskano dla `max_features = 0.5`. Oznacza to, że model działał lepiej, gdy przy każdym podziale analizował tylko część cech, a nie wszystkie.

### 1.2.5. Końcowa konfiguracja modelu

Po przeprowadzeniu całego strojenia jako najlepszy zestaw parametrów wybrano:

| Parametr | Najlepsza wartość |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | None |
| `min_samples_split` | 2 |
| `max_features` | 0.5 |

Dla tej konfiguracji model osiągnął najlepszy średni wynik R² = 0.8285, przy MAE = 31324 oraz RMSE = 47770.

Można więc uznać, że model Random Forest dobrze poradził sobie z zadaniem regresji. Największy wpływ na wynik miały tutaj parametry `max_depth` oraz `max_features`, natomiast zwiększanie liczby drzew poprawiało wynik bardziej stopniowo.

## 2. Model K-Nearest Neighbors (KNN)

### 2.1. Opis KNN

KNN to algorytm, który przewiduje wartość na podstawie najbardziej podobnych przykładów ze zbioru treningowego. Nie buduje klasycznego modelu podczas uczenia, tylko przechowuje dane i korzysta z nich dopiero przy predykcji.

Dla nowej obserwacji algorytm liczy odległość do wszystkich punktów w zbiorze treningowym. Następnie wybiera `k` najbliższych sąsiadów. W zadaniu regresji przewidywana wartość jest wyznaczana na podstawie ich wartości, najczęściej jako średnia albo średnia ważona.


### 2.2. Strojenie hiperparametrów modelu K-Nearest Neighbors

Najpierw sprawdzono wpływ parametru `n_neighbors`, czyli liczby najbliższych sąsiadów branych pod uwagę przy predykcji. Testowano wartości od 1 do 50.

| `n_neighbors` | R² | Odchylenie R² | MAE | RMSE | Uwagi |
|---|---:|---:|---:|---:|---|
| 1  | 0.5852 | 0.0137 | 48311 | 74295 | wynik wyraźnie słaby |
| 3  | 0.6992 | 0.0119 | 42086 | 63272 | duża poprawa |
| 5  | 0.7194 | 0.0120 | 40950 | 61106 | kolejna poprawa |
| 10 | 0.7307 | 0.0120 | 40406 | 59857 | najlepszy wynik |
| 20 | 0.7262 | 0.0104 | 41023 | 60355 | lekki spadek |
| 50 | 0.7086 | 0.0094 | 42934 | 62272 | wyraźnie gorzej |

Na podstawie tych wyników wybrano `n_neighbors = 10`. Widać, że zbyt mała liczba sąsiadów dawała słabsze wyniki, a zbyt duża powodowała zbyt mocne uśrednianie.

Następnie sprawdzono parametr `weights`, czyli sposób ważenia sąsiadów.

| `weights` | R² | Odchylenie R² | MAE | RMSE | Uwagi |
|---|---:|---:|---:|---:|---|
| `uniform`  | 0.7307 | 0.0120 | 40406 | 59857 | wynik bazowy |
| `distance` | 0.7356 | 0.0115 | 39874 | 59313 | lepszy wynik |

Lepsze wyniki uzyskano dla `weights = distance`, więc większy wpływ miały bliższe obserwacje.

Na końcu sprawdzono metrykę odległości.

| `metric` | R² | Odchylenie R² | MAE | RMSE | Uwagi |
|---|---:|---:|---:|---:|---|
| `euclidean` | 0.7356 | 0.0115 | 39874 | 59313 | wynik dobry |
| `manhattan` | 0.7480 | 0.0104 | 38936 | 57904 | najlepszy wynik |
| `chebyshev` | 0.7198 | 0.0127 | 41234 | 61060 | wynik słabszy |
| `minkowski` | 0.7356 | 0.0115 | 39874 | 59313 | taki sam jak euclidean |

Najlepszy wynik uzyskano dla `metric = manhattan`. Metryka `chebyshev` wypadła wyraźnie słabiej. 
Końcowo wybrano następującą konfigurację modelu:

| Parametr | Najlepsza wartość |
|---|---|
| `n_neighbors` | 10 |
| `weights` | `distance` |
| `metric` | `manhattan` |

Dla tej konfiguracji model osiągnął:

- R² = 0.7480
- MAE = 38936
- RMSE = 57904