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

### 1.1. Czym są Sztuczne Sieci Neuronowe

!!! Dodać

### 1.2 Analiza wpływu parametrów na skuteczność działania sieci

Model bazowy:
- Proporcja zbioru testowego = 0.2
- Architektura warstw ukrytych = [32]
- Funkcja aktywacji = relu
- Skala inicjalizacji wag = 1.0
- Współczynnik uczenia = 0.001
- Liczba epok = 100
- Liczba powtórzeń = 3

Badanie przeprowadzono, testując kolejno jeden parametr w każdym kroku, na ten który wykazał się najmniejszym RMSE. Po znalezieniu najlepszego wyniku, skrypt nadpisywał model bazowy optymalną wartością. Każdy eksperyment powtarzano trzykrotnie, co było absolutnym minimum, w celu uśrednienia wyników i wyeliminowania wpływu losowej inicjalizacji wag oraz podziału danych. Taki sposób oparty jedynie na minimalzacji błędu, miał jeden zasadniczy problem. Skrypt, działający automatycznie, mógł wybrać model przeuczony, co dostrzegliśmy na początku analizy. Z tego powodu wprowadzliśmy do kodu dodatkowy bezpiecznik. Zanim skrypt wybrał parametr z najniższym RMSE, sprawdzał różnicę między wynikiem R^2 na zbiorze treningowym a testowym. Ustaliliśmy próg na poziomie 3%. Jeśli różnica przekrzaczała tę wartość, model automatycznie był odrzucany.

#### 1.2.1 Wielkość próby testowej (test_ratio)

W pierwszym kroku zbadaliśmy wpływ proporcji podziału danych na zbiór uczący i testowy. Testowano wartości: 0.1, 0.15, 0.2, 0.3, 0.4.

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| 0.1 | 54512.81 | 55426.59 | 0.7763 | 0.7724 | 0.7704 |
| 0.15 | 54178.14 | 54361.03 | 0.7795 | 0.7779 | 0.7778 |
| 0.2 | 54478.91 | 55568.48 | 0.7769 | 0.7681 | 0.7810 |
| 0.3 | 54625.10 | 55421.69 | 0.7755 | 0.7699 | 0.7701 |
| 0.4 | 55518.36 | 56628.78 | 0.7672 | 0.7607 | 0.7695 |

Najlepszym wynikiem okazała się proporcja 0.15, osiągając najniższy błąd na zbiorze testowym RMSE = 54361.03. Choć w uczeniu maszynowym najczęściej stosuje się podział 80/20, który u nas zdobył niewiele gorszy wynik RMSE = 55568.48, to w tym eksperymencie zmniejszenie zbioru testowego, prawdopodobnie pozwoliło na zasilenie sieci dodatkową pulą danych treningowych. Powiększenie zbioru testowego do wartości 0.3, 0.4 powodowało spadek jakości modelu. Ze względu na te wyniki, to właśnie wartość 0.15 została wybrana do dalszych eksperymentów.

#### 1.2.2 Głębokość sieci liczba wartw ukrytych

W drugim etapie badaliśmy wpływ głebokości sieci. Testowaliśmy od 0 do 4 wartw ukrytych o stałej liczbie neuronów 16.

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| [] (0 warstw) | 69046.58 | 68877.29 | 0.6418 | 0.6434 | 0.6493 |
| [16] | 55998.80 | 55825.74 | 0.7644 | 0.7657 | 0.7613 |
| [16, 16] | 52655.37 | 53031.54 | 0.7917 | 0.7886 | 0.7900 |
| [16, 16, 16] | 51461.20 | 52513.99 | 0.8010 | 0.7927 | 0.7933 |
| [16, 16, 16, 16] | 50824.52 | 52168.90 | 0.8059 | 0.7954 | 0.7962 |

Przy braku wartsw ukrytych widać, że model osiąga najsłabszy wynik i wyjaśnia jedynie 64% zmienności ceny.
Wprowadzenie już pierwszej warstwy ukrytej [16] znacząco poprawia wyniki, skok 12 punktów procentowych R^2. Kolejne warstwy poprawiają wyniki, ale nie tak drastycznie jak pierwszy przeskok. Zwycięzcą okazał się model z 4 warstwami ukrytymi, który osiągnął najlepszy wynik 52168.90 RMSE, dlatego też ta architektura 4-warstwowa została wybrana do dalszej optymalizacji.

#### 1.2.3 Topologia sieci 

W trzecim etapie badania testowaliśmy rozkład neuronów w naszej 4-warstwowej sieci. Testowaliśmy głównie strukturę, w której liczba neuronów maleje w głąb sieci, ale również taką gdzie liczba neuronów była stała.

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| [8, 4, 4, 2] | 76718.92 | 76621.35 | 0.5013 | 0.5040 | 0.7533 |
| [16, 8, 4, 2] | 52779.60 | 53684.18 | 0.7907 | 0.7834 | 0.7866 |
| [32, 16, 8, 4] | 50008.41 | 51922.69 | 0.8121 | 0.7974 | 0.7969 |
| [64, 32, 16, 8] | 47788.03 | 51787.94 | 0.8284 | 0.7984 | 0.7978 |
| [32, 32, 32, 32] | 48375.93 | 51682.46 | 0.8242 | 0.7992 | 0.8002 |

Wąska topologia sieci [8, 4, 4, 2] okazała się zdecydowanie zbyt wąska, nie była w stanie przetworzyć naszych danych, osiągając wynik 50% R^2. Minimalne zwiększeni do [16, 8, 4, 2] podniosło wynik o 28 punktów procentowych R^2. Architektura [32, 32, 32, 32] osiągnęła najlepszy wynik R^2 = 0.7992, przekraczając w pojedynczej próbie barierę 0.8. Pozostałe warianty, jak [32,16,8,4] i [64,32,16,8] uzyskały zbliżone, ale ostatecznie słabsze wyniki pod względem RMSE. Ostatenie algorytm wybrał układ o stałej szerokości [32,32,32,32], ponieważ miał on najlepszą celność RMSE = 51682.46.

#### 1.2.4 Funkcja aktywacji

W czwartym kroku sprawdziliśmy, funkcję aktywacji. Testowaliśmy funkcje: ReLU, Leaky ReLU,, Tanh i Sigmoid.

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| relu | 48375.93 | 51682.46 | 0.8242 | 0.7992 | 0.8002 |
| leaky_relu | 47822.17 | 51219.45 | 0.8282 | 0.8028 | 0.8035 |
| tanh | 49221.07 | 50906.53 | 0.8180 | 0.8052 | 0.8084 |
| sigmoid | 59534.01 | 58378.58 | 0.7337 | 0.7438 | 0.7437 |

Najsłabszy wynik uzyskała funkcja Sigmoid śrendi test R^2 = 0.7438. Najciekawszym wynikiem tego etapu była rywalizacja między ReLU a Tanh. Funkcja ReLU jest standardem w głębokim uczeniu, jednak w naszym eksperymencie funkcja Tanh osiągneła nieznacznie lepszy wynik  R^2 = 0.8052. Różnica RMSE wyniosła między funkcjami 775,93 na przewagę Tanh. Z tego względu to funkcja Tanh została wybrana do dalszej optymalizacji.

#### 1.2.5 Skala inicjalizacji wag

W tym etapie sprawdziliśmy, jak zmiana skali (mnożnika) początkowych wag wpływa na naukę sieci. Testowane wartości: 0.01, 0.1, 1.0, 2.0, 5.0

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| 0.01 | 115418.29 | 115439.63 | -0.0008 | -0.0016 | -0.0043 |
| 0.1 | 56776.69 | 55967.42 | 0.7578 | 0.7645 | 0.7749 |
| 1.0 | 49221.07 | 50906.53 | 0.8180 | 0.8052 | 0.8084 |
| 2.0 | 46981.72 | 52262.53 | 0.8342 | 0.7947 | 0.7927 |
| 5.0 | 52829.00 | 63793.70 | 0.7903 | 0.6938 | 0.7189 |

Najgorszy wyniki uzyskaliśym dla skali 0.01. Przy tak małych wagach sieć praktycznie się nie uczy, co widać po ujemnym R^2, oznaczającym błąd większy niż przy zwykłym zgadywaniu średniej ceny. Z kolei przy skali 5.0 wagi były zbyt duże, co widać na bardzo wysokim błędzie na zbiorze testowym. Najlepszym wartością okazała się skala 1.0, czyli standardowa. Uzyskała ona RMSE = 50906.53. Warto zauważyć, że skala 2.0 dała lepszy wynik na treningu 0.8342, jednak gorszy na testowym i różnica była spora bo aż 4 punkty procentowe R^2, może to świadczyć o tym że model zaczął się przeuczać. Dlatego do końcowego modelu wybraliśmy skalę 1.0.

#### 1.2.6 Współczynnik uczenia

W szóstym kroku zbadaliśmy współczynnik uczenia. Testowane wartości: 0.01, 0.005, 0.001, 0.0005, 0.0001

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| 0.01 | 44950.77 | 51748.24 | 0.8482 | 0.7987 | 0.7966 |
| 0.005 | 45918.59 | 51007.88 | 0.8415 | 0.8043 | 0.8122 |
| 0.001 | 49221.07 | 50906.53 | 0.8180 | 0.8052 | 0.8084 |
| 0.0005 | 50940.22 | 51526.85 | 0.8050 | 0.8004 | 0.8012 |
| 0.0001 | 56166.90 | 55475.32 | 0.7630 | 0.7687 | 0.7670 |

Testy wykazały klasyczną zależność, zbyt wysoki współczynnik 0.01 prowadził do przeuczenia, różnica między R^2 na zbiorze treningowym a testowym wyniosła około 5 punktów procentowych. Natomiast zbyt niski współczynnik 0.0001 powodował niedouczenie wynik R^2 wyniósł 0.7687. Najlepszą wartością okazało się 0.001 z najniższym RMSE = 50906.53. Okazało się lepsze minimalnie od naszej wartości początkowej 0.005. 


#### 1.2.7 Liczba epok 

Przedostatnim badanym parametrem była liczba epok. Testowane liczby epok: 10,30,50,100,200

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| 10 | 57275.63 | 56457.85 | 0.7535 | 0.7604 | 0.7594 |
| 30 | 53974.35 | 53802.17 | 0.7811 | 0.7825 | 0.7879 |
| 50 | 52317.66 | 52720.65 | 0.7944 | 0.7910 | 0.7853 |
| 100 | 49221.07 | 50906.53 | 0.8180 | 0.8052 | 0.8084 |
| 200 | 46208.40 | 50458.34 | 0.8396 | 0.8086 | 0.8086 |

Wzrast ze wzrostem liczby epok błąd na zbiorze testowym jak również na zbiorze treningowym malał. Najlepszy wynik na zbiorze testowym osiągneliśmy dla 200 epok R^2 = 0.8086. Ta wartość przekroczyła narzucony przez nas próg 0.03 różnicy w R^2 między zbiorem testowym, a treningowym. Widać, że w porównaniu do 100 epok, wzrost na zbiorze testowym jest 0.003, a na treningowym 0.02, są to pierwsze sygnały przeucznia. Dlatego 100 epok wygrało.

!!! poprawić wyżej 

#### 1.2.8 Liczba powtórzeń

W otatnim etapie przeanalizowaliśmy wpływ uśredniania wyników na ocene jakości modelu. Testowana liczba powtórzeń: 1,3,5,10.

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| 1 | 48871.61 | 49672.24 | 0.8216 | 0.8084 | 0.8084 |
| 3 | 49221.07 | 50906.53 | 0.8180 | 0.8052 | 0.8084 |
| 5 | 49400.75 | 50715.63 | 0.8168 | 0.8054 | 0.8126 |
| 10 | 49438.36 | 51486.73 | 0.8162 | 0.8014 | 0.8126 |

Wynik dla zalednie 1 powtórzenia może się wydawać najlepszy RMSE = 49672.24, ale w praktyce jest to szcześliwy wylosowanie inicjalizacji wag lub/i podziału danych. Zwiększenie liczby powtórzeń do 3, 5 i 10 urealnia wyniki, niwelując wpływ przypadkowości. Różnice między wariantami są już znaczenie miejsze. Początkowe założenie o wykonaniu minumum 3 powtórzeń w każdym kroku było słuszne, jest to liczba, która pozwoliła zachować balans pomiędzy istotnością statycznyną, a czasem wykonywania skryptu.

### 1.3 Podsumowanie i Wnioski

Ostateczna, optymalna konfigracja sieci: 
- Proporcja zbioru testowego: 0.15
- Głębokość i topologia: 4 warstwy ukryte [32,32,32,32]
- Funckja aktywacji: Tanh
- Skala inicjalizacji wag: 1.0
- Współczynnik uczenia: 0.001
- Liczba epok: 100
- Liczba powtórzeń: 3 

Osiągneliśmy przy tym R^2 = 0.8052, RMSE = 50906.53

!!! do dokończenia


### 2.1. Czym jest Random Forest 

Random Forest to model, który składa się z wielu drzew decyzyjnych. Zamiast opierać wynik na jednym drzewie, bierze się dużo drzew i łączy ich odpowiedzi. W regresji po prostu liczy się średnią z ich przewidywań. To działa głównie dlatego, że pojedyncze drzewo decyzyjne łatwo przesadza z dopasowaniem do danych treningowych. Może dobrze zapamiętać przykłady, ale potem gorzej radzi sobie na nowych danych. A kiedy takich drzew jest dużo, to błędy jednych często są niwelowane przez inne. Jedno drzewo może się pomylić mocniej, ale pozostałe trochę to wyrównają. Ważne jest też to, że te drzewa nie są budowane identycznie. Każde uczy się na trochę innym fragmencie danych i przy podziałach bierze pod uwagę tylko część cech. Dzięki temu nie powstaje sto kopii tego samego modelu, tylko wiele podobnych, ale jednak różnych drzew. No i właśnie przez to Random Forest zwykle działa lepiej niż pojedyncze drzewo. Jest bardziej odporny na przypadkowe błędy w danych, mniej się przeucza i daje bardziej stabilne wyniki. Nie dlatego, że każde drzewo jest idealne. 

### 2.2. Strojenie hiperparametrów modelu Random Forest

Strojenie modelu Random Forest wykonano metodą greedy. W praktyce oznaczało to, że w każdym kroku zmieniano tylko jeden hiperparametr, a pozostałe pozostawały bez zmian. Dzięki temu można było sprawdzić, jak dana zmiana wpływa na jakość modelu.

Do oceny wykorzystano trzy metryki:

- R² – pokazuje, jak dobrze model wyjaśnia zmienność danych,
- MAE – średni błąd bezwzględny,
- RMSE – pierwiastek z błędu średniokwadratowego.

### 2.2.1. Strojenie parametru `n_estimators`

Na początku sprawdzono wpływ liczby drzew w lesie. Testowano wartości od 50 do 500.

| `n_estimators` | R² | Odchylenie R² | MAE | RMSE | Uwagi |
|---|---:|---:|---:|---:|---|
| 50  | 0.8223 | 0.0068 | 31607 | 48621 | wynik bazowy |
| 100 | 0.8239 | 0.0078 | 31402 | 48407 | lepszy od poprzedniego |
| 200 | 0.8243 | 0.0076 | 31345 | 48354 | lekka poprawa |
| 300 | 0.8247 | 0.0079 | 31308 | 48296 | kolejna niewielka poprawa |
| 500 | 0.8248 | 0.0079 | 31298 | 48277 | najlepszy wynik |

Na podstawie tych wyników wybrano `n_estimators = 500`. Widać, że większa liczba drzew poprawiała wyniki, ale poprawa była już raczej niewielka.

### 2.2.2. Strojenie parametru `max_depth`

W kolejnym kroku sprawdzono maksymalną głębokość drzew.

| `max_depth` | R² | Odchylenie R² | MAE | RMSE | Uwagi |
|---|---:|---:|---:|---:|---|
| 5    | 0.6550 | 0.0131 | 48273 | 67757 | wynik wyraźnie słaby |
| 10   | 0.7896 | 0.0095 | 35881 | 52909 | duża poprawa |
| 15   | 0.8203 | 0.0079 | 31994 | 48902 | wynik już dużo lepszy |
| 25   | 0.8248 | 0.0079 | 31308 | 48285 | bardzo dobry wynik |
| None | 0.8248 | 0.0079 | 31298 | 48277 | najlepszy wynik |

Tutaj było widać bardzo wyraźnie, że zbyt mała głębokość pogarszała działanie modelu. Najlepszy wynik uzyskano dla `max_depth = None`, czyli bez ograniczenia głębokości.

### 2.2.3. Strojenie parametru `min_samples_split`

Następnie testowano minimalną liczbę próbek potrzebną do wykonania podziału.

| `min_samples_split` | R² | Odchylenie R² | MAE | RMSE | Uwagi |
|---|---:|---:|---:|---:|---|
| 2  | 0.8248 | 0.0079 | 31298 | 48277 | najlepszy wynik |
| 5  | 0.8246 | 0.0079 | 31366 | 48313 | prawie taki sam wynik |
| 10 | 0.8222 | 0.0084 | 31687 | 48631 | lekki spadek |
| 20 | 0.8160 | 0.0089 | 32483 | 49480 | wyraźnie gorzej |
| 50 | 0.7977 | 0.0091 | 34610 | 51883 | najsłabszy wynik |

Najlepsza okazała się wartość `min_samples_split = 2`. Wraz ze wzrostem tego parametru jakość modelu stopniowo spadała.

### 2.2.4. Strojenie parametru `max_features`

Na końcu sprawdzono, jaka liczba cech używanych przy podziale daje najlepsze wyniki.

| `max_features` | R² | Odchylenie R² | MAE | RMSE | Uwagi |
|---|---:|---:|---:|---:|---|
| 0.3 | 0.8208 | 0.0081 | 32562 | 48825 | słabszy wynik |
| 0.5 | 0.8285 | 0.0078 | 31324 | 47770 | najlepszy wynik |
| 0.7 | 0.8283 | 0.0079 | 31213 | 47794 | bardzo podobny wynik |
| 0.9 | 0.8270 | 0.0080 | 31200 | 47972 | lekki spadek |
| 1.0 | 0.8248 | 0.0079 | 31298 | 48277 | słabiej niż dla 0.5 i 0.7 |

Najlepszy wynik końcowy uzyskano dla `max_features = 0.5`. Oznacza to, że model działał lepiej, gdy przy każdym podziale analizował tylko część cech, a nie wszystkie.

### 2.2.5. Końcowa konfiguracja modelu

Po przeprowadzeniu całego strojenia jako najlepszy zestaw parametrów wybrano:

| Parametr | Najlepsza wartość |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | None |
| `min_samples_split` | 2 |
| `max_features` | 0.5 |

Dla tej konfiguracji model osiągnął najlepszy średni wynik R² = 0.8285, przy MAE = 31324 oraz RMSE = 47770.

Można więc uznać, że model Random Forest dobrze poradził sobie z zadaniem regresji. Największy wpływ na wynik miały tutaj parametry `max_depth` oraz `max_features`, natomiast zwiększanie liczby drzew poprawiało wynik bardziej stopniowo.

## 3. Model K-Nearest Neighbors (KNN)

### 3.1. Opis KNN

KNN to algorytm, który przewiduje wartość na podstawie najbardziej podobnych przykładów ze zbioru treningowego. Nie buduje klasycznego modelu podczas uczenia, tylko przechowuje dane i korzysta z nich dopiero przy predykcji.

Dla nowej obserwacji algorytm liczy odległość do wszystkich punktów w zbiorze treningowym. Następnie wybiera `k` najbliższych sąsiadów. W zadaniu regresji przewidywana wartość jest wyznaczana na podstawie ich wartości, najczęściej jako średnia albo średnia ważona.


### 3.2. Strojenie hiperparametrów modelu K-Nearest Neighbors

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