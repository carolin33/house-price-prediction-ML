# Projekt Elementy sztucznej inteligencji

## Wstęp i opis badanych problemów

Niniejsze opracowanie stanowi analizę porównawczą skuteczności wybranych metod uczenia maszynowego w rozwiązywaniu dwóch fundamentalnych problemów analitycznych: regresji wartości ciągłych oraz klasyfikacji wieloklasowej. Badania oparto na zbiorze danych California Housing, który od lat służy jako benchmark w dziedzinie modelowania predykcyjnego, oferując złożone zależności nieliniowe oraz wyzwania związane z interpretacją cech przestrzennych i demograficznych.

Głównym celem projektu jest zbadanie, jak zmiany kluczowych hiperparametrów wpływają na zdolność generalizacji modeli. W pracy skupiono się na trzech odmiennych architekturach:

- Sztucznych Sieciach Neuronowych (SSN) typu MLP, zaimplementowanych niskopoziomowo, co pozwoliło na precyzyjną kontrolę procesu optymalizacji.
- Lasach Losowych (Random Forest), reprezentujących podejście zespołowe oparte na drzewach decyzyjnych.
- Algorytmie k-Najbliższych Sąsiadów (k-NN), służącym jako punkt odniesienia dla metod opartych na podobieństwie lokalnym.

W ramach projektu zdefiniowano dwa równoległe zadania:

- Problem Regresji (Estymacja cen): Próba przewidzenia mediany wartości nieruchomości (median_house_value). Jest to zadanie o wysokiej złożoności ze względu na silną nieliniowość rynku nieruchomości. Sukces modelu mierzony jest zdolnością do minimalizacji błędu średniokwadratowego (RMSE) oraz maksymalizacją współczynnika determinacji (R2), który w najlepszych konfiguracjach sieci neuronowych osiągnął poziom powyżej 0.78.
- Problem Klasyfikacji (Lokalizacja względem oceanu): Przewidywanie zmiennej kategorycznej ocean_proximity (5 klas). Wyjątkowość tego zadania polega na celowym usunięciu współrzędnych geograficznych. Decyzja ta wymusza na modelach poszukiwanie ukrytych wzorców w strukturze demograficznej i standardzie budynków, zamiast prostego przypisania klasy na podstawie mapy. Jest to problem trudniejszy, gdzie naturalna bariera dokładności (Accuracy) oscyluje wokół 69%, co wynika z dużego nakładania się cech różnych lokalizacji.

Zastosowanie podejścia ceteris paribus (zmiana jednego parametru przy stałych pozostałych) miało na celu nie tylko znalezienie "najlepszego wyniku", ale przede wszystkim zrozumienie mechaniki uczenia

## Przegląd literatury

  W ramach prac nad projektem przeprowadzono szczegółowy przegląd literatury przedmiotu oraz analizę rozwiązań dostępnych na platformach analitycznych takich jak Kaggle oraz GitHub. Analiza ta pozwoliła na identyfikację dominujących trendów metodologicznych oraz ustalenie punktów odniesienia dla skuteczności budowanych modeli predykcyjnych. 
  
  Powszechnym podejściem w problematyce estymacji cen nieruchomości w Kalifornii jest wykorzystanie zaawansowanych algorytmów uczenia maszynowego, w szczególności metod zespołowych, takich jak Random Forest oraz XGBoost. Opracowania te wykazują, że modele oparte na drzewach decyzyjnych osiągają stabilną skuteczność w przedziale od 0.80 do 0.84. W przypadku alternatywnych rozwiązań wykorzystujących sztuczne sieci neuronowe (ANN), badacze wskazują na konieczność stosowania głębokich architektur typu wielowarstwowy perceptron oraz rygorystycznego skalowania danych wejściowych, co jest niezbędne dla poprawnego procesu uczenia. Publikacje naukowe potwierdzają, że sieci neuronowe (szczególnie wielowarstwowe perceptrony) wykazują znacznie wyższą zdolność adaptacyjną do nieliniowych trendów rynkowych niż klasyczna regresja liniowa. 
  
  Kluczowym wnioskiem płynącym z analizy porównawczej jest dominująca rola zmiennej dochodu mieszkańca (median_income), która w większości modeli posiada największą moc predykcyjną, przewyższając parametry czysto fizyczne budynku, takie jak liczba pokoi. Istotnym aspektem poruszanym w literaturze jest również problem „szumu informacyjnego” wynikający ze specyfiki zbioru danych. Badania wskazują, że sztuczne ograniczenie wartości mediany ceny do 500 000 USD stanowi wyzwanie metodologiczne – wielu autorów decyduje się na eliminację tych rekordów, aby uniknąć błędów systematycznych (tzw. bias) i uzyskać bardziej realistyczne prognozy. 
  
  Wiele analiz dowodzi, że przejście do zaawansowanych modeli nieliniowych wymaga zastosowania celowej inżynierii cech. Za najbardziej efektywne uznaje się tworzenie wskaźników relatywnych, takich jak liczba pokoi na gospodarstwo domowe (rooms_per_household) czy zagęszczenie populacji. Jak zauważa E. Clark, kluczem do sukcesu jest inżynieria cech specyficzna dla branży nieruchomości (ang. Domain-Specific Feature Engineering), np. analiza proporcji sypialni do ogólnej liczby pokoi, co pozwala modelowi lepiej odróżnić domy rodzinne od obiektów inwestycyjnych. 
  
  Ostatecznie, badacze są zgodni, że lokalizacja geograficzna w ścisłym powiązaniu z bliskością oceanu (ocean_proximity) stanowi drugi najważniejszy czynnik determinujący wartość nieruchomości. Modele klasy boostingowej, w szczególności XGBoost, najlepiej radzą sobie z wychwytywaniem nieliniowych zależności przestrzennych. Nowoczesne podejścia testują również wykorzystanie struktur piramidalnych sieci neuronowych (np. o architekturze 8-24-12-6-1), które poprzez stopniową kompresję informacji pozwalają na precyzyjną ekstrakcję kluczowych danych. 
  
  Eksperci wskazują także, że najwyższym standardem optymalizacji (pozwalającym osiągnąć błąd MAE poniżej 26 000 USD) jest tzw. geokodowanie odwrotne, czyli zamiana współrzędnych na nazwy konkretnych miast. Dzięki temu model „rozumie” lokalną specyfikę rynków (np. prestiż danej dzielnicy), co w połączeniu z cechami wielomianowymi pozwala na najdokładniejsze odwzorowanie cen na tym zbiorze danych. 
 
 

## Badanie działania Sztucznych Sieci Neuronowych SSN

### Metodyka i parametry badawcze

W tej sekcji przeprowadziliśmy analizę wpływu poszczególnych parametrów na skuteczność i stabilność procesu uczenia sztucznej sieci neuronowej typu MLP (Multilayer Perceptron). Sieć zaimplementowano od podstaw w języku Python, bez korzystania z frameworków takich jak TensorFlow czy PyTorch.

Badania przeprowadzono zgodnie z metodologią **ceteris paribus** - zmienialiśmy tylko jeden parametr naraz, utrzymując pozostałe na wartościach bazowych. Konfiguracja bazowa to:

- Liczba epok: **30**
- Tempo uczenia się (lr): **0.001**
- Architektura warstw ukrytych: **[16]** (jedna warstwa ukryta z 16 neuronami)
- Funkcja aktywacji: **ReLU**
- Proporcja zbioru testowego: **0.2**
- Skala inicjalizacji wag: **1.0** (He initialization)
- Liczba powtórzeń eksperymentu: **3** (wyniki uśredniane)

Sieć trenowana była metodą **mini-batch gradient descent** z propagacją wsteczną (backpropagation). W zadaniu regresji zastosowano liniową warstwę wyjściową z funkcją kosztu MSE, natomiast w klasyfikacji - warstwę Softmax z kategoryczną entropią krzyżową.

Metryki oceny:
- **Regresja:** RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), R² (współczynnik determinacji)
- **Klasyfikacja:** Accuracy (dokładność)

---

### Wyniki dla problemu regresyjnego

Zadanie regresyjne polegało na przewidywaniu mediany wartości nieruchomości (`median_house_value`) na podstawie wszystkich pozostałych cech zbioru California Housing. Zmienna kategoryczna `ocean_proximity` została zakodowana metodą One-Hot Encoding. Wszystkie cechy numeryczne oraz zmienna docelowa zostały poddane standaryzacji (`StandardScaler`), co jest kluczowe dla stabilności procesu uczenia sieci neuronowej.

#### 1. Wpływ liczby epok

Liczba epok określa, ile razy sieć przechodzi przez cały zbiór treningowy. Zbyt mała liczba epok prowadzi do niedouczenia - sieć nie zdąży wystarczająco dopasować wag do danych. Zbyt duża może z kolei prowadzić do przeuczenia, choć w przypadku prostych architektur MLP efekt ten jest mniej wyraźny niż np. w głębokich sieciach konwolucyjnych.

| Epoki | Śr. RMSE (trening) | Śr. RMSE (test) | Śr. MAE (trening) | Śr. MAE (test) | Śr. R² (trening) | Śr. R² (test) | Najl. RMSE (test) | Najl. R² (test) |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 61 328 | 61 023 | 43 664 | 44 075 | 0.7173 | 0.7204 | 60 668 | 0.7182 |
| 30 | 58 309 | 58 045 | 40 607 | 40 762 | 0.7445 | 0.7471 | 57 002 | 0.7512 |
| 50 | 57 532 | 57 513 | 39 847 | 40 164 | 0.7513 | 0.7516 | 56 488 | 0.7618 |
| 100 | 56 305 | 56 754 | 39 228 | 39 838 | 0.7618 | 0.7581 | 55 829 | 0.7674 |
| 200 | 54 953 | 55 578 | 37 985 | 38 530 | 0.7731 | 0.7681 | 54 861 | 0.7695 |

**Analiza:** Widać wyraźny, monotoniczny spadek błędu RMSE i wzrost R² wraz ze wzrostem liczby epok. Przyrosty jednak maleją - różnica między 10 a 30 epokami (\~3 000 RMSE) jest znacznie większa niż między 100 a 200 (~1 200 RMSE). To typowe zachowanie algorytmu gradientowego: na początku optymalizacja przebiega szybko w kierunku minimum, a potem zwalnia w okolicach punktu zbieżności.

Warto zwrócić uwagę, że przy 200 epokach pojawia się niewielka rozbieżność między RMSE treningowym (54 953) a testowym (55 578). To sygnał początkowego przeuczenia - sieć zaczyna dopasowywać się do szumu w danych treningowych. W praktyce optymalnym wyborem jest **50–100 epok**, które dają bardzo dobry kompromis między jakością a ryzykiem overfittingu.

---

#### 2. Wpływ tempa uczenia (Learning Rate)

Tempo uczenia (LR) to jeden z najważniejszych hiperparametrów sieci neuronowej. Określa wielkość kroku przy aktualizacji wag w kierunku malejącego gradientu. Zbyt wysoki LR powoduje „przeskakiwanie" minimum i niestabilność, a zbyt niski - ekstremalnie wolną zbieżność.

| LR | Śr. RMSE (trening) | Śr. RMSE (test) | Śr. MAE (test) | Śr. R² (trening) | Śr. R² (test) | Najl. RMSE (test) | Najl. R² (test) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 63 781 | 63 400 | 45 715 | 0.6943 | 0.6982 | 62 558 | 0.7003 |
| 0.001 | 58 309 | 58 045 | 40 762 | 0.7445 | 0.7471 | 57 002 | 0.7512 |
| 0.005 | 57 602 | 57 206 | 39 660 | 0.7506 | 0.7543 | 55 883 | 0.7609 |
| 0.01 | 57 550 | 57 398 | 39 632 | 0.7510 | 0.7527 | 55 836 | 0.7613 |

**Analiza:** Widać tu klasyczny efekt wpływu tempa uczenia. Przy LR = 0.0001 sieć uczy się zbyt wolno - przy 30 epokach nie zdąży w wystarczającym stopniu zminimalizować funkcji kosztu, stąd RMSE pozostaje wysoki (63 400), a R² najniższy (0.6982).

Zwiększenie LR do 0.001 daje skokową poprawę (~5 000 RMSE mniej). Dalsze zwiększanie do 0.005 i 0.01 przynosi dodatkowy, choć mniejszy zysk. Wartości **0.005–0.01** okazują się optymalne - osiągają najlepsze R² na teście (0.754 i 0.753). Wyższe wartości LR (np. 0.1 czy 0.5) nie zostały uwzględnione, ponieważ we wcześniejszych próbach powodowały eksplozję gradientów (wartości NaN), co jest typowym objawem zbyt agresywnej optymalizacji w przypadku funkcji kosztu MSE.

---

#### 3. Wpływ architektury warstw ukrytych

Architektura sieci - liczba warstw i neuronów w każdej z nich - bezpośrednio determinuje zdolność modelu do modelowania nieliniowych zależności. Głębsza sieć z większą liczbą neuronów potrafi uchwycić bardziej złożone wzorce, ale kosztem większego ryzyka przeuczenia i dłuższego czasu treningu.

| Warstwy ukryte | Śr. RMSE (trening) | Śr. RMSE (test) | Śr. MAE (test) | Śr. R² (trening) | Śr. R² (test) | Najl. RMSE (test) | Najl. R² (test) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| [] (brak) | 68 952 | 68 875 | 50 324 | 0.6427 | 0.6439 | 67 896 | 0.6470 |
| [16] | 58 309 | 58 045 | 40 762 | 0.7445 | 0.7471 | 57 002 | 0.7512 |
| [16, 8] | 56 589 | 56 578 | 39 517 | 0.7592 | 0.7595 | 54 988 | 0.7743 |
| [16, 8, 4] | 56 023 | 55 935 | 38 675 | 0.7641 | 0.7651 | 54 851 | 0.7696 |
| [16, 8, 4, 2] | 55 596 | 55 646 | 38 368 | 0.7677 | 0.7675 | 54 790 | 0.7759 |
| [32, 16] | 54 394 | 54 675 | 37 709 | 0.7777 | 0.7756 | 54 068 | 0.7818 |
| [32, 32] | 54 465 | 54 614 | 37 566 | 0.7770 | 0.7760 | 53 767 | 0.7786 |
| [64, 32] | 52 961 | 53 665 | 36 870 | 0.7892 | 0.7838 | 52 825 | 0.7863 |

**Analiza:** Wyniki jednoznacznie potwierdzają fundamentalną zasadę sieci neuronowych - **głębsza i szersza sieć lepiej modeluje złożone zależności**. Model bez warstw ukrytych (regresja liniowa) osiąga R² = 0.64, co jest wynikiem dramatycznie gorszym od nawet najprostszej sieci z jedną warstwą ukrytą (R² = 0.75).

Najciekawsza jest tutaj konfrontacja dwóch podejść architektonicznych:
- **Piramida malejąca** [16, 8, 4, 2] - stopniowe zwężanie warstw wymusza hierarchiczną kompresję informacji. R² = 0.7675.
- **Szersza, płytsza sieć** [64, 32] - mniej warstw, ale znacznie więcej neuronów. R² = 0.7838.

Szersza architektura wygrywa, co sugeruje, że w tym zbiorze danych ważniejsza jest **pojemność pojedynczych warstw** niż głębokość sieci. Wynika to z faktu, że zbiór California Housing zawiera stosunkowo proste, tabelaryczne zależności - nie potrzebuje hierarchii abstrakcji typowej dla np. rozpoznawania obrazów.

Architektura **[64, 32]** osiągnęła najlepsze wyniki: najniższy RMSE (53 665), najniższy MAE (36 870) i najwyższe R² (0.7838). Warto jednak zauważyć, że pojawia się tu rozbieżność train/test (RMSE: 52 961 vs 53 665), sygnalizująca początkowy overfitting.

---

#### 4. Wpływ liczby neuronów (pojedyncza warstwa ukryta)

Test ten izoluje wpływ szerokości warstwy, utrzymując architekturę na jednej warstwie ukrytej.

| Neurony | Śr. RMSE (trening) | Śr. RMSE (test) | Śr. MAE (test) | Śr. R² (trening) | Śr. R² (test) | Najl. RMSE (test) | Najl. R² (test) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| [4] | 63 801 | 63 044 | 45 106 | 0.6939 | 0.7012 | 60 424 | 0.7275 |
| [8] | 60 471 | 60 320 | 42 510 | 0.7252 | 0.7267 | 59 005 | 0.7401 |
| [16] | 58 309 | 58 045 | 40 762 | 0.7445 | 0.7471 | 57 002 | 0.7512 |
| [32] | 56 813 | 56 712 | 39 740 | 0.7574 | 0.7585 | 55 490 | 0.7702 |
| [64] | 55 939 | 56 115 | 39 054 | 0.7648 | 0.7635 | 54 991 | 0.7743 |

**Analiza:** Zależność jest czytelna i monotonia - każde podwojenie liczby neuronów przynosi wyraźną poprawę. Różnica między 4 a 64 neuronami to aż **7 000 punktów RMSE** i **0.06 R²**. Sieć z 4 neuronami jest po prostu zbyt „ciasna", by zamodelować wielowymiarowy problem z 13 cechami wejściowymi (8 numerycznych + 5 kategorii OHE).

Przy 64 neuronach pojawia się już lekka rozbieżność train/test (55 939 vs 56 115), co sugeruje, że jeszcze większe warstwy mogłyby prowadzić do przeuczenia bez zastosowania regularyzacji (dropout, L2).

---

#### 5. Wpływ funkcji aktywacji

Funkcja aktywacji definiuje nieliniowość warstw ukrytych. Bez niej sieć wielowarstwowa sprowadzałaby się do zwykłej regresji liniowej.

| Aktywacja | Śr. RMSE (trening) | Śr. RMSE (test) | Śr. MAE (test) | Śr. R² (trening) | Śr. R² (test) | Najl. RMSE (test) | Najl. R² (test) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| ReLU | 58 309 | 58 045 | 40 762 | 0.7445 | 0.7471 | 57 002 | 0.7512 |
| Leaky ReLU | 58 362 | 58 189 | 40 867 | 0.7440 | 0.7458 | 57 109 | 0.7503 |
| Tanh | 58 554 | 58 162 | 40 886 | 0.7423 | 0.7460 | 57 414 | 0.7476 |
| Sigmoid | 62 120 | 61 508 | 44 029 | 0.7100 | 0.7159 | 60 319 | 0.7284 |

**Analiza:** Wyniki potwierdzają wiedzę z literatury - **ReLU** to zdecydowanie najlepsza funkcja aktywacji dla warstw ukrytych w tym zadaniu. Leaky ReLU i Tanh dają praktycznie identyczne wyniki, co sugeruje, że na tej głębokości sieci (1 warstwa) problem „martwych neuronów" ReLU nie jest istotny.

**Sigmoid** wyraźnie odstaje - RMSE wyższe o ~3 500, R² niższe o 0.03. To klasyczny efekt **zanikającego gradientu** (*vanishing gradient*): sigmoid kompresuje sygnał do zakresu (0, 1), a jego pochodna ma maksimum równe 0.25. Oznacza to, że gradienty podczas propagacji wstecznej są systematycznie tłumione, co dramatycznie spowalnia uczenie. Efekt jest szczególnie dotkliwy w głębszych sieciach, ale nawet tu, przy jednej warstwie ukrytej, jest już wyraźnie widoczny.

---

#### 6. Wpływ proporcji podziału train/test

Proporcja podziału danych wpływa na dwa aspekty: im więcej danych treningowych, tym lepiej model się uczy, ale tym mniej danych pozostaje do rzetelnej walidacji.

| Test ratio | Śr. RMSE (trening) | Śr. RMSE (test) | Śr. MAE (test) | Śr. R² (trening) | Śr. R² (test) | Najl. RMSE (test) | Najl. R² (test) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 58 206 | 59 063 | 41 133 | 0.7450 | 0.7417 | 57 895 | 0.7430 |
| 0.2 | 58 309 | 58 045 | 40 762 | 0.7445 | 0.7471 | 57 002 | 0.7512 |
| 0.3 | 58 518 | 58 526 | 41 107 | 0.7424 | 0.7434 | 57 785 | 0.7477 |
| 0.4 | 59 119 | 59 343 | 41 989 | 0.7361 | 0.7373 | 58 646 | 0.7419 |
| 0.8 | 62 703 | 62 324 | 45 223 | 0.7027 | 0.7086 | 61 647 | 0.7160 |

**Analiza:** Podział **80/20** (test_ratio = 0.2) okazuje się optymalny, co jest zgodne z ogólnie przyjętą zasadą w uczeniu maszynowym. Przy tym podziale model ma wystarczająco dużo danych do nauki i jednocześnie osiąga najlepszy R² na zbiorze testowym (0.7471).

Skrajne podziały ujawniają dwa zjawiska:
- **Test ratio = 0.1**: model ma więcej danych do nauki, ale zbiór testowy jest bardzo mały, więc estymacja błędu jest mniej stabilna (wyższy RMSE testowy niż treningowy).
- **Test ratio = 0.8**: model ma do dyspozycji zaledwie 20% danych do treningu. R² spada do 0.7086, czyli o ~4 punkty procentowe. To oznacza, że sieć neuronowa jest „głodna danych" - potrzebuje wystarczająco dużego zbioru treningowego, aby dobrze ustawić wagi.

---

#### 7. Wpływ skali inicjalizacji wag

Inicjalizacja wag to krok, który decyduje o punkcie startowym optymalizacji. Złe wartości początkowe mogą powodować zanikanie lub eksplozję gradientów już od pierwszej iteracji.

| Skala init. | Śr. RMSE (trening) | Śr. RMSE (test) | Śr. MAE (test) | Śr. R² (trening) | Śr. R² (test) | Najl. RMSE (test) | Najl. R² (test) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 59 082 | 58 636 | 41 388 | 0.7377 | 0.7418 | 57 624 | 0.7522 |
| 0.1 | 58 657 | 58 449 | 41 221 | 0.7414 | 0.7435 | 57 213 | 0.7493 |
| 1.0 | 58 309 | 58 045 | 40 762 | 0.7445 | 0.7471 | 57 002 | 0.7512 |
| 2.0 | 59 476 | 59 503 | 41 749 | 0.7342 | 0.7342 | 58 275 | 0.7400 |
| 5.0 | 64 432 | 64 010 | 45 613 | 0.6879 | 0.6924 | 61 913 | 0.7065 |

**Analiza:** Skala inicjalizacji 1.0 (odpowiadająca standardowej inicjalizacji He dla ReLU) daje najlepsze wyniki. Stało się tak, ponieważ inicjalizacja He jest specjalnie zaprojektowana, by utrzymać stałą wariancję aktywacji w kolejnych warstwach - zapewnia to stabilny przepływ gradientu.

Zbyt mała skala (0.01) powoduje, że aktywacje są bliskie zeru i sieć potrzebuje więcej epok, by „ruszyć" z uczeniem. Zbyt duża skala (5.0) prowadzi do eksplozji wartości na wyjściu - RMSE rośnie o ponad 5 000, a R² spada do 0.69. To potwierdza, że **prawidłowa inicjalizacja wag jest krytyczna** dla efektywnego uczenia sieci neuronowej.

---

#### 8. Wpływ liczby powtórzeń eksperymentu

Ze względu na losową inicjalizację wag, kolejność mieszania danych i stochastyczność mini-batch, każde uruchomienie sieci daje nieco inne wyniki. Powtórzenia pozwalają ocenić stabilność modelu.

| Powtórzenia | Śr. RMSE (trening) | Śr. RMSE (test) | Śr. MAE (test) | Śr. R² (trening) | Śr. R² (test) | Najl. RMSE (test) | Najl. R² (test) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 58 040 | 57 002 | 39 875 | 0.7481 | 0.7512 | 57 002 | 0.7512 |
| 3 | 58 309 | 58 045 | 40 762 | 0.7445 | 0.7471 | 57 002 | 0.7512 |
| 5 | 58 413 | 58 307 | 40 886 | 0.7438 | 0.7439 | 57 002 | 0.7512 |
| 10 | 58 455 | 58 794 | 41 150 | 0.7431 | 0.7407 | 56 856 | 0.7545 |

**Analiza:** Wyniki średnie są bardzo stabilne niezależnie od liczby powtórzeń - rozrzut R² wynosi zaledwie 0.01 (od 0.74 do 0.75). To świadczy o tym, że konfiguracja bazowa jest dość odporna na losowość inicjalizacji.

Co ciekawe, najlepszy pojedynczy przebieg (Best RMSE) jest praktycznie taki sam przy 1, 3 i 5 powtórzeniach (~57 002), ale przy 10 powtórzeniach spada do 56 856, co potwierdza, że więcej prób zwiększa szansę na trafienie szczególnie dobrego startu. W praktyce **3 powtórzenia** to rozsądne minimum pozwalające oszacować stabilność wyników.

---

### Wyniki dla problemu klasyfikacji

Zadanie klasyfikacyjne polegało na przypisaniu nieruchomości do jednej z 5 klas zmiennej `ocean_proximity`: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`. Celowo usunięto cechy `longitude` i `latitude`, aby model nie mógł trywialnie rozwiązać zadania na podstawie współrzędnych geograficznych, lecz musiał uczyć się bardziej subtelnych zależności wynikających z cech demograficznych i mieszkaniowych.

Warstwa wyjściowa korzystała z funkcji **Softmax**, a funkcją kosztu była **kategoryczna entropia krzyżowa** (Categorical Cross-Entropy).

#### 1. Wpływ liczby epok

| Epoki | Śr. Accuracy (trening) | Śr. Accuracy (test) | Najl. Accuracy (test) |
|:---:|---:|---:|---:|
| 10 | 0.6735 | 0.6715 | 0.6751 |
| 30 | 0.6791 | 0.6798 | 0.6828 |
| 50 | 0.6799 | 0.6768 | 0.6797 |
| 100 | 0.6828 | 0.6813 | 0.6845 |
| 200 | 0.6854 | 0.6833 | 0.6930 |

**Analiza:** Widoczny jest powolny, ale konsekwentny wzrost dokładności ze wzrostem liczby epok. Poprawa jest jednak mniej dramatyczna niż w regresji - różnica między 10 a 200 epokami to zaledwie ~1.2 punktu procentowego. To wynika z natury problemu: klasy `ocean_proximity` nakładają się na siebie w przestrzeni cech (po usunięciu współrzędnych geograficznych), co tworzy naturalną barierę dokładności.

Interesujący jest wynik dla 50 epok: Accuracy testowe (0.6768) jest niższe niż dla 30 epok (0.6798). To prawdopodobnie efekt losowości - przy uśrednieniu 3 powtórzeń takie wahania w zakresie ~0.003 są normalne. Najlepszy pojedynczy wynik przy 200 epokach (0.6930) potwierdza, że dłuższy trening pozwala sieci lepiej dopasować granice decyzji.

---

#### 2. Wpływ tempa uczenia (Learning Rate)

| LR | Śr. Accuracy (trening) | Śr. Accuracy (test) | Najl. Accuracy (test) |
|:---:|---:|---:|---:|
| 0.0001 | 0.6644 | 0.6632 | 0.6683 |
| 0.001 | 0.6791 | 0.6798 | 0.6828 |
| 0.005 | 0.6805 | 0.6773 | 0.6833 |
| 0.01 | 0.6771 | 0.6761 | 0.6816 |

**Analiza:** W przeciwieństwie do regresji, klasyfikacja toleruje dosyć wąskie spektrum LR. Najlepsze średnie accuracy na teście uzyskano przy **LR = 0.001** (0.6798), a najlepszy pojedynczy wynik - przy **LR = 0.005** (0.6833).

Wartość 0.0001 jest za mała - w ciągu 30 epok sieć nie zdąży dostatecznie dobrze ustawić granic decyzyjnych. Z kolei wyższy LR = 0.01 nie poprawia wyników, a wręcz je lekko pogarsza. To może wynikać z faktu, że na granicy stabilności optymalizacji gradienty Softmax + Cross-Entropy są bardziej „łagodne" niż w MSE (ograniczone naturalnie w zakresie [0,1]), ale mimo to zbyt agresywna aktualizacja wag może prowadzić do oscylacji wokół minimum.

---

#### 3. Wpływ architektury warstw ukrytych

| Warstwy ukryte | Śr. Accuracy (trening) | Śr. Accuracy (test) | Najl. Accuracy (test) |
|:---:|---:|---:|---:|
| [] (brak) | 0.6584 | 0.6620 | 0.6668 |
| [16] | 0.6791 | 0.6798 | 0.6828 |
| [16, 8] | 0.6832 | 0.6794 | 0.6874 |
| [16, 8, 4] | 0.6811 | 0.6777 | 0.6806 |
| [16, 8, 4, 2] | 0.6758 | 0.6733 | 0.6838 |
| [32, 16] | 0.6866 | 0.6854 | 0.6911 |
| [32, 32] | 0.6884 | 0.6871 | 0.6935 |
| [64, 32] | 0.6900 | 0.6861 | 0.6932 |

**Analiza:** Wyniki ujawniają bardzo interesujący wzorzec. Model bez warstw ukrytych osiąga 0.6620 - to właściwie wieloklasowa regresja logistyczna. Dodanie jednej warstwy ukrytej podnosi wynik o ~1.8 pkt. procentowego.

Głęboka piramida [16, 8, 4, 2] **pogarsza** wynik w porównaniu z prostszymi architekturami! Warstwa z zaledwie 2 neuronami tworzy wąskie gardło informacyjne, które kompresuje reprezentację zbyt agresywnie dla 5-klasowego problemu. Optymalna architektura to **[32, 32]** - dwie równoległe warstwy o identycznej szerokości, oferujące najlepszą średnią Accuracy na teście (0.6871).

---

#### 4. Wpływ liczby neuronów (pojedyncza warstwa ukryta)

| Neurony | Śr. Accuracy (trening) | Śr. Accuracy (test) | Najl. Accuracy (test) |
|:---:|---:|---:|---:|
| [4] | 0.6638 | 0.6637 | 0.6700 |
| [8] | 0.6720 | 0.6720 | 0.6753 |
| [16] | 0.6791 | 0.6798 | 0.6828 |
| [32] | 0.6815 | 0.6770 | 0.6821 |
| [64] | 0.6841 | 0.6837 | 0.6894 |

**Analiza:** Trend jest taki sam jak w regresji - więcej neuronów = lepsze wyniki. Sieć z 64 neuronami osiąga najwyższy wynik testowy (0.6837). Warto jednak zauważyć, że przyrost jest znacznie mniejszy niż w regresji: 4→64 neuronów daje poprawę ~2 pkt. procentowych (vs ~6 pkt. R² w regresji). To potwierdza, że barierą w klasyfikacji nie jest zdolność modelu, lecz sama natura cech wejściowych - bez lokalizacji geograficznej przewidywanie bliskości oceanu jest fundamentalnie trudne.

---

#### 5. Wpływ funkcji aktywacji

| Aktywacja | Śr. Accuracy (trening) | Śr. Accuracy (test) | Najl. Accuracy (test) |
|:---:|---:|---:|---:|
| ReLU | 0.6791 | 0.6798 | 0.6828 |
| Leaky ReLU | 0.6793 | 0.6794 | 0.6823 |
| Tanh | 0.6776 | 0.6764 | 0.6787 |
| Sigmoid | 0.6655 | 0.6659 | 0.6702 |

**Analiza:** Ranking funkcji aktywacji jest identyczny jak w regresji: **ReLU ≈ Leaky ReLU > Tanh > Sigmoid**. Różnica między ReLU a Sigmoid (~1.4 pkt. proc.) jest mniejsza niż w regresji, ale nadal wyraźna.

ReLU i Leaky ReLU dają niemal identyczne wyniki, co potwierdza obserwację z regresji - przy jednej warstwie ukrytej problem „martwych neuronów" nie jest znaczący. Tanh wypada nieco słabiej, choć jest to funkcja antysymetryczna (w przeciwieństwie do sigmoid), co teoretycznie powinno pomagać w optymalizacji. Prawdopodobnie zbieżność Tanh jest po prostu wolniejsza przy 30 epokach niż ReLU.

---

#### 6. Wpływ proporcji podziału train/test

| Test ratio | Śr. Accuracy (trening) | Śr. Accuracy (test) | Najl. Accuracy (test) |
|:---:|---:|---:|---:|
| 0.1 | 0.6819 | 0.6743 | 0.6778 |
| 0.2 | 0.6791 | 0.6798 | 0.6828 |
| 0.3 | 0.6796 | 0.6735 | 0.6779 |
| 0.4 | 0.6773 | 0.6762 | 0.6775 |
| 0.8 | 0.6663 | 0.6661 | 0.6667 |

**Analiza:** Podział **80/20** ponownie okazuje się optymalny. Przy test_ratio = 0.1 Accuracy treningowe jest najwyższe (0.6819), ale testowe spada do 0.6743 - mały zbiór testowy daje mniej wiarygodną estymację. 

Przy test_ratio = 0.8 (zaledwie 20% danych do nauki) model wyraźnie cierpi, osiągając najsłabsze wyniki (0.6661). Zauważmy jednak, że nawet w takiej ekstremalnej sytuacji model nadal bije baseline bezdatowy - to świadczy o tym, że sieć potrafi wyciągnąć użyteczne wzorce nawet z bardzo ograniczonej liczby przykładów.

---

#### 7. Wpływ skali inicjalizacji wag

| Skala init. | Śr. Accuracy (trening) | Śr. Accuracy (test) | Najl. Accuracy (test) |
|:---:|---:|---:|---:|
| 0.01 | 0.6724 | 0.6693 | 0.6753 |
| 0.1 | 0.6752 | 0.6722 | 0.6775 |
| 1.0 | 0.6791 | 0.6798 | 0.6828 |
| 2.0 | 0.6787 | 0.6761 | 0.6765 |
| 5.0 | 0.6720 | 0.6703 | 0.6763 |

**Analiza:** Klasyfikacja jest nieco bardziej odporna na złą inicjalizację niż regresja. Różnica między najlepszą (1.0) a najgorszą (0.01) skalą to ~1 pkt. procentowy, podczas gdy w regresji różnica R² wynosiła ~0.05. Wynika to z „łagodniejszej" natury funkcji kosztu Cross-Entropy - gradienty Softmax są bardziej stabilne niż gradienty MSE w obecności dużych wartości.

Mimo to optymalna skala to nadal **1.0** (He initialization), co potwierdza jej uniwersalność.

---

### Podsumowanie wyników SSN

#### Najlepsza konfiguracja - Regresja

| Parametr | Optymalna wartość | R² (test) |
|:---|:---|---:|
| Architektura | [64, 32] | **0.7838** |
| Epoki | 200 | 0.7681 |
| Learning Rate | 0.005–0.01 | 0.7543 |
| Aktywacja | ReLU | 0.7471 |
| Inicjalizacja wag | 1.0 (He) | 0.7471 |
| Podział danych | 80/20 | 0.7471 |

Najlepszy pojedynczy wynik regresji: **RMSE = 52 825, R² = 0.7863** (architektura [64, 32]).

#### Najlepsza konfiguracja - Klasyfikacja

| Parametr | Optymalna wartość | Accuracy (test) |
|:---|:---|---:|
| Architektura | [32, 32] | **0.6871** |
| Epoki | 200 | 0.6833 |
| Learning Rate | 0.001 | 0.6798 |
| Aktywacja | ReLU | 0.6798 |
| Inicjalizacja wag | 1.0 (He) | 0.6798 |
| Podział danych | 80/20 | 0.6798 |

Najlepszy pojedynczy wynik klasyfikacji: **Accuracy = 0.6935** (architektura [32, 32]).

#### Kluczowe wnioski z badań SSN

1. **Architektura jest najważniejszym parametrem.** To wybór liczby warstw i neuronów ma największy wpływ na wynik końcowy - zarówno w regresji (rozpiętość R²: 0.64–0.78), jak i w klasyfikacji (Accuracy: 0.66–0.69).

2. **ReLU zdominowała.** We wszystkich eksperymentach ReLU dawała najlepsze lub praktycznie najlepsze wyniki. Sigmoid konsekwentnie traciła ze względu na zanikający gradient.

3. **Inicjalizacja He jest optymalna.** Zarówno zbyt mała (0.01), jak i zbyt duża (5.0) skala inicjalizacji wag prowadziły do gorszych wyników, potwierdzając teoretyczne uzasadnienie metody He.

4. **Podział 80/20 to złoty standard.** Zarówno dla regresji, jak i klasyfikacji, podział danych z 20% na zbiór testowy dawał najlepsze wyniki.

5. **Klasyfikacja bez współrzędnych geograficznych jest fundamentalnie trudna.** Accuracy na poziomie ~69% to granica wynikająca z natury danych - klasy `ocean_proximity` nie są dobrze separowalne wyłącznie na podstawie cech demograficznych.

---

## Badanie działania metod Uczenia Maszynowego UN

### Metodologia i przygotowanie danych

Wszystkie klasyczne algorytmy uczenia maszynowego (Random Forest, KNN, SVM, Gradient Boosting) zostały przetestowane przy użyciu jednolitej metodologii, co pozwala na rzetelne porównanie ich wyników.

**Zakres badań i wybór cech:**

Eksperymenty przeprowadzono na zbiorze **California Housing**, zawierającym informacje o nieruchomościach z Kalifornii. Zbiór liczy około 20 000 obserwacji i obejmuje cechy takie jak m.in. mediana dochodu mieszkańców (`median_income`), liczba pokoi, wiek budynku, populacja obszaru czy bliskość oceanu (`ocean_proximity`)

Dla każdego modelu zrealizowano dwa oddzielne zadania:
* **Klasyfikację** zmiennej `ocean_proximity` (5 klas: `NEAR BAY`, `NEAR OCEAN`, `INLAND`, `ISLAND`, `<1H OCEAN`),
* **Regresję** zmiennej `median_house_value` (mediana wartości nieruchomości).

W zadaniu klasyfikacji, dla wszystkich modeli, celowo **usunięto cechy `longitude` i `latitude`**. Zapobiegło to sytuacji, w której algorytmy mogłyby podjąć decyzję wyłącznie na podstawie dokładnych współrzędnych, co wymusiło naukę zależności z cech demograficznych i mieszkaniowych.

**Preprocessing:**
Proces przygotowania danych obejmował:
* Uzupełnianie brakujących wartości medianą (cechy numeryczne) lub wartością najczęstszą (cechy kategoryczne).
* Kodowanie zmiennych kategorycznych metodą **One-Hot Encoding**.
* **Standaryzację cech (`StandardScaler`):** Stosowaną w modelach KNN i SVM.  Skalowanie realizowane było wewnątrz **Pipeline** w procesie walidacji krzyżowej, co zapobiegło wyciekowi danych (*data leakage*). 

**Walidacja:**
Do oceny jakości każdego modelu wykorzystano **5-krotną walidację krzyżową** (5-fold Cross-Validation). W zadaniu klasyfikacji zastosowano wariant `StratifiedKFold`, aby zachować identyczny rozkład klas w każdym z pięciu folderów, natomiast w regresji standardowy `KFold`. Dzięki temu uzyskane wyniki (Accuracy, R², RMSE) są uśrednione i bardziej odporne na przypadkowy podział danych niż w przypadku pojedynczego testu.

**Procedura testowa:** Zastosowano metodę izolacji parametrów, każdy parametr badano osobno, przy stałych wartościach bazowych dla pozostałych ustawień.

---
## 1. Model Random Forest

### 1.1. Czym jest Random Forest i dlaczego działa?

Random Forest (las losowy) to metoda uczenia zespołowego (*ensemble learning*), która buduje wiele drzew decyzyjnych i łączy ich odpowiedzi w finalną predykcję. Idea kluczowa to zasada tłumu: wiele modeli popełniających różne błędy, połączonych razem, daje lepszy wynik niż pojedynczy model. Właśnie to zapewnia losowość wbudowana w algorytm.

Każde drzewo w lesie jest trenowane na innym, losowo wylosowanym podzbiorze danych treningowych (*bootstrap sampling*). Dodatkowo przy każdym podziale węzła drzewo wybiera najlepszą cechę nie spośród wszystkich cech, ale spośród losowo wybranego ich podzbioru. Ten mechanizm, kontrolowany przez parametr `max_features`, wymusza różnorodność drzew: każde z nich uczy się nieco innych wzorców.

W zadaniu klasyfikacji wynik końcowy to głosowanie większościowe — klasa wskazana przez największą liczbę drzew. W zadaniu regresji wynikiem jest średnia z predykcji wszystkich drzew. To uśrednianie jest kluczowe, ponieważ skutecznie redukuje wariancję modelu i ogranicza ryzyko przeuczenia, które stanowi główną słabość pojedynczych drzew decyzyjnych.

**Konfiguracja bazowa modelu:**
- **klasyfikacja:** `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `max_features="sqrt"`
- **regresja:** `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `max_features=1.0`

---

### 1.2. Wpływ liczby drzew (`n_estimators`)

Parametr `n_estimators` kontroluje liczbę drzew tworzących las. Jest to jeden z nielicznych parametrów, dla których zwiększanie wartości zazwyczaj nie pogarsza wyników, lecz z czasem przynosi coraz mniejsze korzyści. Wynika to z prawa malejących przyrostów (*diminishing returns*): pierwsze dodatkowe drzewa istotnie stabilizują model, ale kolejne poprawiają wynik już tylko nieznacznie.

| n_estimators | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 50  | 0.6778 | 0.0036 | 0.5012 | 0.5022 | 0.8223 | 0.0068 | 31 606.94 | 48 621.39 |
| 100 | 0.6810 | 0.0037 | 0.5032 | 0.5044 | 0.8239 | 0.0078 | 31 402.30 | 48 406.60 |
| 200 | 0.6815 | 0.0039 | 0.5029 | 0.5037 | 0.8243 | 0.0076 | 31 345.26 | 48 354.46 |
| 300 | 0.6811 | 0.0038 | 0.5028 | 0.5035 | 0.8247 | 0.0079 | 31 308.49 | 48 296.04 |
| 500 | 0.6821 | 0.0039 | 0.5037 | 0.5045 | 0.8248 | 0.0079 | 31 297.65 | 48 277.18 |

Wyniki dobrze pokazują typowe zachowanie Random Forest. Największy wzrost jakości następuje między 50 a 100 drzewami. Dalsze zwiększanie liczby drzew nadal poprawia wyniki, ale skala poprawy jest już niewielka. W praktyce oznacza to, że zakres **100–200 drzew** jest rozsądnym kompromisem między jakością a kosztem obliczeniowym. Wartość 500 daje najlepsze wyniki, ale przewaga nad 200 jest już bardzo mała.

---

### 1.3. Wpływ głębokości drzew (`max_depth`)

Parametr `max_depth` ogranicza maksymalną głębokość pojedynczego drzewa. Bezpośrednio kontroluje więc złożoność modelu i wpływa na kompromis **bias–variance**.

- mała głębokość → model zbyt prosty, niedouczony (*underfitting*),
- bardzo duża głębokość → pojedyncze drzewo może się przeuczyć,
- w Random Forest ryzyko to jest mniejsze niż w pojedynczym drzewie, ponieważ wyniki są uśredniane.

| max_depth | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 5    | 0.6467 | 0.0051 | 0.4402 | 0.4161 | 0.6553 | 0.0126 | 48 269.18 | 67 727.53 |
| 10   | 0.6714 | 0.0051 | 0.4757 | 0.4597 | 0.7895 | 0.0096 | 35 888.42 | 52 926.90 |
| 15   | 0.6800 | 0.0025 | 0.4902 | 0.4827 | 0.8195 | 0.0077 | 32 072.06 | 49 001.52 |
| 25   | 0.6815 | 0.0022 | 0.5036 | 0.5055 | 0.8239 | 0.0077 | 31 429.73 | 48 407.04 |
| None | 0.6810 | 0.0037 | 0.5032 | 0.5044 | 0.8239 | 0.0078 | 31 402.30 | 48 406.60 |

Przy `max_depth=5` model jest wyraźnie niedouczony. Wyniki są słabe zarówno w klasyfikacji, jak i w regresji, co oznacza, że płytkie drzewa nie potrafią uchwycić złożonych zależności obecnych w zbiorze. Wraz ze wzrostem głębokości jakość systematycznie rośnie.

Najlepsze wyniki uzyskano dla `max_depth=25` oraz `max_depth=None`, a różnice między tymi wariantami są minimalne. Sugeruje to, że drzewa i tak naturalnie zatrzymują się na sensownej głębokości wynikającej z danych. To ważna obserwacja: w Random Forest brak ograniczenia głębokości nie musi oznaczać problemu z przeuczeniem, ponieważ uśrednianie wielu drzew skutecznie redukuje wariancję.

---

### 1.4. Wpływ minimalnej liczby próbek do podziału (`min_samples_split`)

Parametr `min_samples_split` określa minimalną liczbę próbek w węźle potrzebną do wykonania kolejnego podziału. Im wyższa wartość tego parametru, tym trudniej drzewu tworzyć bardzo szczegółowe reguły decyzyjne.

| min_samples_split | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2  | 0.6810 | 0.0037 | 0.5032 | 0.5044 | 0.8239 | 0.0078 | 31 402.30 | 48 406.60 |
| 5  | 0.6801 | 0.0043 | 0.5006 | 0.5003 | 0.8242 | 0.0079 | 31 437.42 | 48 366.97 |
| 10 | 0.6813 | 0.0012 | 0.4970 | 0.4942 | 0.8218 | 0.0084 | 31 742.84 | 48 693.84 |
| 20 | 0.6805 | 0.0038 | 0.4914 | 0.4844 | 0.8157 | 0.0090 | 32 525.82 | 49 521.42 |
| 50 | 0.6740 | 0.0033 | 0.4800 | 0.4651 | 0.7975 | 0.0088 | 34 623.55 | 51 901.27 |

Wyniki pokazują dość czytelną zależność: **małe wartości (`2`, `5`) są najlepsze**, a większe prowadzą do pogorszenia jakości. Dla `min_samples_split=50` model staje się zbyt zachowawczy — drzewa nie mogą rozgałęziać się wystarczająco głęboko, więc tracą zdolność do uchwycenia lokalnych wzorców i nieregularności w danych.

Ciekawa jest obserwacja, że przy `min_samples_split=10` odchylenie standardowe Accuracy jest wyjątkowo niskie. Oznacza to większą stabilność między foldami, ale kosztem trochę słabszych średnich wyników. W praktyce jest to klasyczny kompromis: nieco bardziej konserwatywny model może być stabilniejszy, ale niekoniecznie najlepszy jakościowo.

---

### 1.5. Wpływ liczby rozważanych cech przy podziale (`max_features`)

Parametr `max_features` to jeden z kluczowych elementów, który odróżnia Random Forest od zwykłego zbioru podobnych drzew. Przy każdym podziale model losuje tylko część cech i szuka najlepszego splitu wyłącznie wśród nich.

Mniejsza wartość `max_features`:

- zwiększa losowość,
- zmniejsza korelację między drzewami,
- poprawia efekt uśredniania.

Zbyt mała wartość może jednak ograniczyć zdolność modelu do znajdowania dobrych podziałów.

| max_features | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.3 | 0.6810 | 0.0037 | 0.5032 | 0.5044 | 0.8187 | 0.0087 | 32 836.55 | 49 112.95 |
| 0.5 | 0.6781 | 0.0018 | 0.5039 | 0.5057 | 0.8262 | 0.0076 | 31 613.48 | 48 081.86 |
| 0.7 | 0.6775 | 0.0025 | 0.5065 | 0.5101 | 0.8250 | 0.0080 | 31 551.78 | 48 259.90 |
| 0.9 | 0.6734 | 0.0033 | 0.5061 | 0.5103 | 0.8261 | 0.0082 | 31 352.83 | 48 101.34 |
| 1.0 | 0.6711 | 0.0038 | 0.5055 | 0.5100 | 0.8239 | 0.0078 | 31 402.30 | 48 406.60 |

Wyniki pokazują interesującą różnicę między klasyfikacją a regresją. W klasyfikacji najwyższe Accuracy uzyskano dla `max_features=0.3`, ale metryki lepiej uwzględniające klasy mniejszościowe (`Balanced Accuracy`, `F1 Macro`) są najlepsze dla wartości **0.7–0.9**. W regresji najlepsze wyniki osiągnięto dla `max_features=0.5` i `0.9`.

To bardzo dobrze pokazuje logikę działania Random Forest. Użycie wszystkich cech (`1.0`) nie daje najlepszych rezultatów, ponieważ wtedy drzewa stają się do siebie zbyt podobne. Z punktu widzenia ensemble nie chodzi o to, żeby każde drzewo było maksymalnie podobnym „ekspertem”, tylko żeby wiele drzew wnosiło częściowo różne spojrzenia na dane. Dopiero wtedy uśrednianie daje realny zysk.

---

### 1.6. Podsumowanie analizy Random Forest

Random Forest osiągnął bardzo dobre wyniki w obu zadaniach, szczególnie w regresji, gdzie uzyskano **R² ≈ 0.82–0.83**. Oznacza to, że model wyjaśnia ponad 82% wariancji cen nieruchomości, co jest wynikiem bardzo dobrym.

W klasyfikacji uzyskano Accuracy na poziomie około **0.68**, przy Balanced Accuracy około **0.50**. Ten wynik należy interpretować ostrożnie: zadanie jest trudniejsze, ponieważ:

- klasy `ocean_proximity` są nierównoliczne,
- usunięto współrzędne geograficzne,
- pozostałe cechy opisują relację do oceanu tylko pośrednio.

Najważniejsze wnioski z analizy hiperparametrów RF:

- `n_estimators`: największe korzyści pojawiają się do około 100–200 drzew,
- `max_depth`: głębsze drzewa działają lepiej, a `None` nie prowadzi tu do problematycznego przeuczenia,
- `min_samples_split`: małe wartości są najlepsze, duże zbyt upraszczają model,
- `max_features`: najlepsze wyniki daje umiarkowane ograniczenie liczby cech.

---

## 2. Model K-Nearest Neighbors (KNN)

### 2.1. Czym jest KNN i jak działa?

K-Nearest Neighbors (K najbliższych sąsiadów) to algorytm typu *lazy learning* oraz *instance-based learning*. Oznacza to, że podczas treningu nie buduje jawnego modelu parametrycznego, lecz zapamiętuje zbiór treningowy. Cała praca obliczeniowa odbywa się dopiero w momencie predykcji.

Dla nowej obserwacji algorytm:

1. oblicza odległość do wszystkich obserwacji treningowych,
2. wybiera `k` najbliższych sąsiadów,
3. podejmuje decyzję na podstawie ich etykiet lub wartości.

W klasyfikacji wynik to najczęściej głosowanie większościowe, a w regresji — średnia lub średnia ważona wartości sąsiadów.

Najważniejszą cechą KNN jest silna zależność od geometrii przestrzeni cech. Algorytm nie uczy się reguł, tylko porównuje punkty między sobą. Z tego powodu standaryzacja danych jest absolutnie kluczowa: bez niej cechy o dużej skali zdominowałyby obliczanie odległości.

---

### 2.2. Wpływ liczby sąsiadów (`n_neighbors`)

Parametr `n_neighbors` (`k`) jest najważniejszym hiperparametrem KNN, ponieważ bezpośrednio kontroluje kompromis między wariancją a biasem.

- małe `k` → model bardzo lokalny, podatny na szum i przeuczenie,
- duże `k` → model bardziej wygładzony, ale mniej czuły na lokalne struktury.

| n_neighbors | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1  | 0.5791 | 0.0052 | 0.4762 | 0.4761 | 0.5852 | 0.0137 | 48 310.63 | 74 295.21 |
| 3  | 0.6355 | 0.0070 | 0.4810 | 0.4873 | 0.6992 | 0.0119 | 42 085.78 | 63 272.21 |
| 5  | 0.6495 | 0.0046 | 0.4831 | 0.4853 | 0.7194 | 0.0120 | 40 950.42 | 61 105.90 |
| 10 | 0.6658 | 0.0013 | 0.4918 | 0.4932 | 0.7307 | 0.0120 | 40 405.72 | 59 856.84 |
| 20 | 0.6698 | 0.0055 | 0.4873 | 0.4822 | 0.7262 | 0.0104 | 41 022.85 | 60 354.60 |
| 50 | 0.6663 | 0.0041 | 0.4762 | 0.4615 | 0.7086 | 0.0094 | 42 933.98 | 62 272.04 |

Wyniki bardzo dobrze potwierdzają teorię. Dla `k=1` model jest wyraźnie przeuczony: osiąga słabe wyniki i ma wysoką niestabilność między foldami. Taki model właściwie „zapamiętuje” dane treningowe i jest bardzo czuły na przypadkowe lokalne zakłócenia.

Najlepszy kompromis pojawia się w okolicach `k=10` dla regresji i `k=10–20` dla klasyfikacji. Dalsze zwiększanie liczby sąsiadów prowadzi do nadmiernego wygładzenia granicy decyzyjnej i pogorszenia wyników, zwłaszcza dla mniejszych klas.

---

### 2.3. Wpływ sposobu ważenia sąsiadów (`weights`)

Parametr `weights` określa, czy każdy sąsiad ma taki sam wpływ na decyzję, czy też bliżsi sąsiedzi powinni ważyć więcej.

Dostępne warianty:

- `uniform` — każdy z `k` sąsiadów ma taki sam głos,
- `distance` — bliżsi sąsiedzi mają większy wpływ na predykcję.

| weights | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| uniform  | 0.6495 | 0.0046 | 0.4831 | 0.4853 | 0.7194 | 0.0120 | 40 950.42 | 61 105.90 |
| distance | 0.6409 | 0.0067 | 0.4972 | 0.5018 | 0.7230 | 0.0119 | 40 506.55 | 60 707.95 |

Na pierwszy rzut oka wyniki klasyfikacji mogą wydawać się sprzeczne: `uniform` daje wyższe Accuracy, ale `distance` poprawia Balanced Accuracy i F1 Macro. Nie jest to błąd — po prostu różne metryki opisują inne aspekty jakości.

Accuracy jest silnie zależne od klas dominujących. Z kolei Balanced Accuracy i F1 Macro lepiej pokazują, czy model radzi sobie także z klasami mniej licznymi. To sugeruje, że ważenie odległością pozwala modelowi bardziej precyzyjnie korzystać z lokalnej informacji i lepiej obsługiwać klasy trudniejsze lub rzadsze.

W regresji wariant `distance` jest wyraźnie lepszy: poprawia wszystkie metryki. Jest to bardzo intuicyjne — bardziej podobne nieruchomości powinny mieć większy wpływ na prognozę niż te bardziej odległe w przestrzeni cech.

---

### 2.4. Wpływ metryki odległości (`metric`)

Metryka odległości definiuje, co algorytm rozumie przez „bliskość” dwóch obserwacji. To decyzja fundamentalna, bo cały mechanizm KNN opiera się właśnie na lokalnym sąsiedztwie.

Przetestowano cztery warianty:

- **euclidean** — klasyczna odległość euklidesowa,
- **manhattan** — suma bezwzględnych różnic po wymiarach,
- **chebyshev** — maksymalna różnica w jednym wymiarze,
- **minkowski** — uogólnienie, które przy `p=2` odpowiada euklidesowej.

| metric | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| euclidean | 0.6495 | 0.0046 | 0.4831 | 0.4853 | 0.7194 | 0.0120 | 40 950.42 | 61 105.90 |
| manhattan | 0.6523 | 0.0052 | 0.4856 | 0.4872 | 0.7350 | 0.0088 | 39 883.54 | 59 388.63 |
| chebyshev | 0.6528 | 0.0040 | 0.4868 | 0.4899 | 0.7036 | 0.0124 | 42 245.86 | 62 795.87 |
| minkowski | 0.6495 | 0.0046 | 0.4831 | 0.4853 | 0.7194 | 0.0120 | 40 950.42 | 61 105.90 |

Wyniki dla `euclidean` i `minkowski` są identyczne, co jest całkowicie zgodne z teorią, ponieważ w kodzie metryka Minkowskiego używa domyślnego `p=2`, czyli dokładnie tej samej geometrii co odległość euklidesowa.

Najlepszy wynik regresyjny uzyskała metryka **Manhattan**. Osiągnęła najwyższe `R²`, najniższe `MAE` i najniższe odchylenie standardowe `R²`, co oznacza zarówno wysoką jakość, jak i dobrą stabilność. To sugeruje, że w tym zbiorze danych suma odchyleń po wymiarach lepiej opisuje podobieństwo nieruchomości niż klasyczna odległość euklidesowa. Możliwa interpretacja jest taka, że Manhattan jest mniej czuła na pojedyncze silnie odstające wartości.

Metryka **Chebysheva** daje dobre wyniki klasyfikacyjne dla metryk uwzględniających wszystkie klasy, ale słabo wypada w regresji. Wynika to z jej natury: bierze pod uwagę tylko największą różnicę między cechami, a ignoruje resztę informacji rozproszonej po innych wymiarach.

---

### 2.5. Podsumowanie analizy KNN

KNN osiągnął poprawne, ale wyraźnie słabsze wyniki niż Random Forest, szczególnie w regresji. Przyczyną nie jest pojedynczy źle dobrany parametr, lecz sama natura algorytmu.

Najważniejsze ograniczenia KNN w tym zadaniu to:

- brak jawnego modelowania zależności między cechami,
- silna wrażliwość na wybór `k`,
- duża zależność od definicji odległości,
- podatność na **przekleństwo wymiarowości**.

W przestrzeni o większej liczbie cech odległości między punktami stają się mniej rozróżnialne, przez co pojęcie „najbliższego sąsiada” przestaje być tak użyteczne jak w małych, prostych problemach.

Najlepsza konfiguracja KNN w tej analizie to:

- `n_neighbors = 10`,
- `weights = 'distance'`,
- `metric = 'manhattan'`.

---

## 3. Model Support Vector Machine (SVM)

### 3.1. Czym jest SVM i dlaczego działa?

Support Vector Machine to metoda uczenia maszynowego, która próbuje znaleźć możliwie najlepszą granicę oddzielającą obserwacje należące do różnych klas. W najprostszym przypadku jest to hiperpłaszczyzna maksymalizująca margines, czyli odległość od najbliższych punktów obu klas. Punkty te nazywane są wektorami nośnymi (support vectors), ponieważ to one w największym stopniu decydują o położeniu granicy decyzyjnej.

W wersji klasyfikacyjnej model występuje jako SVC, natomiast w wersji regresyjnej jako SVR. W regresji celem nie jest dokładne dopasowanie każdej obserwacji, lecz znalezienie funkcji, która dobrze opisuje zależność między cechami a zmienną objaśnianą przy zachowaniu odpowiedniej tolerancji błędu.

Dużą zaletą SVM jest możliwość modelowania zależności nieliniowych dzięki zastosowaniu funkcji jądra (kernel). Pozwala to przenieść dane do przestrzeni o wyższym wymiarze, w której łatwiej znaleźć dobrą granicę decyzyjną lub funkcję regresyjną. Jednocześnie metoda ta jest stosunkowo wrażliwa na dobór hiperparametrów, dlatego analiza parametrów ma tu szczególnie duże znaczenie. Ponadto, ponieważ algorytm opiera się na geometrycznym obliczaniu odległości, wymaga on bezwzględnie ujednolicenia skali cech, co w naszym projekcie zrealizowano za pomocą narzędzia StandardScaler.

Analizie poddano cztery kluczowe hiperparametry:
* `C` (parametr regularyzacji),
* `kernel` (rodzaj funkcji jądra),
* `gamma` (współczynnik jądra),
* `epsilon` (margines tolerancji błędu dla regresji).

**Wartości referencyjne (bazowe):**
* **klasyfikacja:** `C=1.0`, `kernel='rbf'`, `gamma='scale'`.
* **regresja:** `C=1.0`, `kernel='rbf'`, `gamma='scale'`, `epsilon=0.1`.

---

### 3.2. Wpływ parametru kary (`C`)

Parametr `C` to hiperparametr, który decyduje o tym, jak bardzo zależy nam na bezbłędnym dopasowaniu modelu do danych treningowych. 
* **Małe wartości C:** Pozwalają na szerszy margines i większą tolerancję na błędy, co tworzy "miękką" granicę decyzyjną.
* **Duże wartości C:** Wymuszają bardzo dokładne dopasowanie do każdego punktu, zawężając margines.

| C | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.1  | 0.6656 | 0.0021 | 0.4511 | 0.4189 | -0.0546 | 0.0026 | 88 294.11 | 118 470.70 |
| 1.0  | 0.6803 | 0.0035 | 0.4807 | 0.4608 | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |
| 10.0 | 0.6836 | 0.0033 | 0.4919 | 0.4753 |  0.0072 | 0.0024 | 84 819.51 | 114 946.44 |
| 100.0| 0.6863 | 0.0025 | 0.5016 | 0.4958 |  0.3312 | 0.0043 | 65 615.82 | 94 349.82 |

Wyniki pokazują, że wraz ze wzrostem `C` jakość modelu wyraźnie się poprawiała, zwłaszcza w zadaniu regresji. Oznacza to, że dla naszego zbioru danych opłacalne było bardziej restrykcyjne dopasowanie, co pozwoliło algorytmowi lepiej uchwycić ukryte zależności.

---

### 3.3. Wpływ funkcji jądra (`kernel`)

Parametr `kernel` określa sposób, w jaki algorytm szuka powiązań między danymi. Przetestowano cztery warianty transformacji przestrzeni: `linear`, `rbf`, `poly` oraz `sigmoid`.

| Kernel | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| linear  | 0.6528 | 0.0038 | 0.4266 | 0.3740 |  0.0685 | 0.0032 | 82 189.00 | 111 345.29 |
| rbf     | 0.6803 | 0.0035 | 0.4807 | 0.4608 | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |
| poly    | 0.6407 | 0.0020 | 0.4442 | 0.4316 | -0.0417 | 0.0018 | 87 745.25 | 117 748.14 |
| sigmoid | 0.5330 | 0.0133 | 0.3717 | 0.3547 | -0.0422 | 0.0027 | 87 699.59 | 117 775.90 |


Zdecydowanie najlepsze rezultaty w zadaniu klasyfikacji dało jądro `rbf`, co sugeruje nieliniowy charakter zależności odpowiedzialnych za rozróżnianie klas `ocean_proximity`. W zadaniu regresji lepszy wynik uzyskano natomiast dla jądra `linear`, co pokazuje, że optymalny wybór funkcji jądra może zależeć od rodzaju rozwiązywanego problemu.

---

### 3.4. Wpływ parametru `gamma`

Parametr `gamma` (używany m.in. z jądrem `rbf`) decyduje o tym, jak daleko sięga wpływ pojedynczej obserwacji treningowej na kształt granicy decyzyjnej. 
* **Małe wartości:** Prowadzą do bardziej globalnego spojrzenia na dane.
* **Duże wartości:** Skupiają się na silnym dopasowaniu lokalnym.

| Gamma | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| scale | 0.6803 | 0.0035 | 0.4807 | 0.4608 | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |
| auto  | 0.6803 | 0.0035 | 0.4807 | 0.4608 | -0.0486 | 0.0027 | 87 973.78 | 118 132.90 |
| 0.01  | 0.6578 | 0.0036 | 0.4299 | 0.3771 | -0.0530 | 0.0026 | 88 221.23 | 118 381.15 |
| 0.1   | 0.6776 | 0.0037 | 0.4726 | 0.4502 | -0.0486 | 0.0026 | 87 967.10 | 118 132.48 |

W przeprowadzonych eksperymentach parametry bazujące na danych (`scale` oraz `auto`) okazały się stabilne i zapewniły najlepszy kompromis między jakością predykcji a ryzykiem nadmiernego dopasowania (overfittingiem), delikatnie wyprzedzając sztywne wartości liczbowe.

---

### 3.5. Wpływ parametru `epsilon` (tylko regresja)

Parametr `epsilon` dotyczy wyłącznie modelu SVR. Definiuje on szerokość specjalnej strefy tolerancji wokół przewidywanej wartości, wewnątrz której drobne błędy nie są w ogóle karane przez algorytm.

| Epsilon | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- |
| 0.01 | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |
| 0.1  | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |
| 0.5  | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |
| 1.0  | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |

Analiza wykazała, że zmiany parametru `epsilon` miały całkowicie marginalny wpływ na jakość regresji przy tych ustawieniach bazowych (wyniki pozostały bez zmian). Oznacza to, że w badanym przypadku szerokość tej strefy nie była kluczowa, a o sukcesie modelu decydowały głównie odpowiednio dobrane parametry `C` oraz `kernel`.

---

### 3.6. Podsumowanie analizy SVM

Algorytm SVM uzyskał w zadaniu regresji wynik **R² ≈ 0.33**, a w klasyfikacji Accuracy na poziomie **0.69**. Model ten wykazał ekstremalną wrażliwość na dobór parametrów – przy niewłaściwych ustawieniach (niskie $C$, jądro liniowe) wyniki regresji były niższe od modelu bazowego (wartości ujemne).

**Kluczowe wnioski z testów:**

* **Najlepsza konfiguracja:** Wyraźnie najlepsze rezultaty osiągnięto dla parametrów: `C=100`, `kernel='rbf'`, `gamma='scale'`.
* **Rola regularyzacji:** Wzrost parametru C z 0.1 do 100 spowodował skok współczynnika R² z wartości ujemnych do **0.3312**. Pokazuje to, że model SVM wymaga silnej regularyzacji, aby skutecznie wychwycić zależności w zbiorze California Housing.
* **Wybór jądra:** Jądro nieliniowe **`rbf` (0.6803)** okazało się skuteczniejsze od liniowego (0.6528) oraz wielomianowego (0.6407). Najsłabiej wypadło jądro `sigmoid` (0.5330).
* **Margines `epsilon`:** Zgodnie z wynikami, zmiana parametru `epsilon` w zakresie od 0.01 do 1.0 nie wpłynęła w żaden sposób na wynik R² (stała wartość -0.0486 przy bazowym C=1), co czyni go najmniej istotnym parametrem w tej analizie.

Model SVM wykazał się solidną skutecznością w klasyfikacji, jednak w zadaniu regresji wyraźnie ustępuje modelom drzewiastym (Random Forest), osiągając znacznie niższe wartości R² przy większym koszcie obliczeniowym.

---

## 4. Gradient Boosting

### 4.1. Czym jest Gradient Boosting i jak działa?

Gradient Boosting to zaawansowana metoda uczenia zespołowego, polegająca na sekwencyjnym budowaniu serii modeli (najczęściej płytkich drzew decyzyjnych). W przeciwieństwie do algorytmu Random Forest, w którym drzewa powstają niezależnie od siebie, w metodzie Gradient Boosting każde kolejne drzewo jest tworzone w celu skorygowania błędów popełnionych przez modele zbudowane wcześniej. 

Proces ten można porównać do wyciągania wniosków z poprzednich pomyłek. Algorytm analizuje, w których miejscach dotychczasowe przewidywania były niedokładne, a następnie dodaje nowe drzewo, którego zadaniem jest naprawienie tych konkretnych błędów (tzw. reziduów). Dzięki takiemu etapowemu podejściu, końcowy model staje się coraz precyzyjniejszy i potrafi skutecznie wykrywać złożone, nieliniowe zależności w danych.

Istotną cechą Gradient Boosting jest jego wysoka skuteczność, która jednak zależy od odpowiedniego doboru tzw. **hiperparametrów** – czyli ustawień konfiguracyjnych wybieranych przez badacza przed rozpoczęciem procesu uczenia. Ponieważ algorytm opiera się na strukturze drzew decyzyjnych, jest on niewrażliwy na różnice w skali cech numerycznych. W związku z tym, w procesie przygotowania danych zrezygnowano z użycia narzędzia `StandardScaler`, co pozwoliło na uproszczenie obliczeń przy zachowaniu wysokiej jakości predykcji.

W analizie rozpatrzono cztery kluczowe hiperparametry, przyjmując dla nich następujące **wartości referencyjne (bazowe)**:
* **`n_estimators`** (100): liczba drzew tworzących model.
* **`learning_rate`** (0.1): tempo uczenia, określające wkład każdego kolejnego drzewa w końcowy wynik.
* **`max_depth`** (3): maksymalna głębokość pojedynczego drzewa (w GB drzewa są celowo płytkie).
* **`subsample`** (1.0): ułamek próbek danych używanych do budowy każdego z drzew.

---

### 4.2. Wpływ liczby estymatorów (`n_estimators`)

Parametr `n_estimators` określa liczbę kolejnych drzew decyzyjnych budujących model. Zwiększanie tej wartości zazwyczaj poprawia jakość predykcji, jednak jednocześnie wydłuża czas obliczeń i może prowadzić do nadmiernego dopasowania (overfittingu), jeśli nie jest równoważone odpowiednio małym współczynnikiem uczenia (`learning_rate`).

| n_estimators | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 50   | 0.6681 | 0.0061 | 0.4804 | 0.4655 | 0.7207 | 0.0126 | 43 226.27 | 60 952.45 |
| 100  | 0.6750 | 0.0061 | 0.4881 | 0.4764 | 0.7706 | 0.0113 | 38 417.48 | 55 236.87 |
| 200  | 0.6776 | 0.0050 | 0.4937 | 0.4865 | 0.8017 | 0.0100 | 35 187.75 | 51 362.18 |
| 300  | 0.6781 | 0.0042 | 0.4980 | 0.4943 | 0.8127 | 0.0090 | 33 926.84 | 49 912.06 |

Wyniki eksperymentu wykazały, że optymalną liczbą drzew dla badanego zbioru jest **300**. Zwiększanie liczby estymatorów konsekwentnie poprawiało zdolności predykcyjne modelu (szczególnie w zadaniu regresji, gdzie błąd RMSE systematycznie malał), bez widocznych oznak drastycznego przeuczenia.

---

### 4.3. Wpływ współczynnika uczenia (`learning_rate`)

Parametr `learning_rate` kontroluje siłę wpływu (wagę) każdego kolejnego drzewa na model końcowy. 
* **Małe wartości:** Prowadzą do wolniejszego, ale zazwyczaj bardziej stabilnego uczenia (wymagają jednak większej liczby drzew).
* **Większe wartości:** Mogą poprawiać wyniki znacznie szybciej, ale drastycznie zwiększają ryzyko przeuczenia.

| Learning Rate | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.01 | 0.6407 | 0.0038 | 0.4316 | 0.4087 | 0.5157 | 0.0085 | 61 273.43 | 80 282.69 |
| 0.05 | 0.6684 | 0.0064 | 0.4805 | 0.4654 | 0.7182 | 0.0119 | 43 402.98 | 61 226.50 |
| 0.1  | 0.6750 | 0.0061 | 0.4881 | 0.4764 | 0.7706 | 0.0113 | 38 417.48 | 55 236.87 |
| 0.2  | 0.6779 | 0.0045 | 0.4942 | 0.4881 | 0.8009 | 0.0094 | 35 335.45 | 51 464.76 |

Zaobserwowano, że wartość `learning_rate` na poziomie **0.2** zapewniła najlepsze rezultaty. Model uczył się na tyle dynamicznie, że przy bazowej liczbie drzew był w stanie wychwycić złożone zależności szybciej i skuteczniej niż przy zachowawczej wartości 0.01.

---

### 4.4. Wpływ głębokości drzew (`max_depth`)

Parametr `max_depth` kontroluje maksymalną złożoność pojedynczych drzew składowych. 
* **Małe wartości:** Generują prostsze modele, które mogą nie wychwytywać wszystkich ukrytych zależności (niedouczenie).
* **Większe wartości:** Pozwalają algorytmowi lepiej i dokładniej dopasować się do danych treningowych, jednak ryzykują zapamiętywaniem szumu.

| Max Depth | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2  | 0.6662 | 0.0066 | 0.4802 | 0.4650 | 0.7282 | 0.0129 | 42 447.98 | 60 135.58 |
| 3  | 0.6750 | 0.0061 | 0.4881 | 0.4764 | 0.7706 | 0.0113 | 38 417.48 | 55 236.87 |
| 4  | 0.6792 | 0.0050 | 0.4961 | 0.4898 | 0.8021 | 0.0097 | 35 076.17 | 51 315.89 |
| 5  | 0.6797 | 0.0054 | 0.4998 | 0.4970 | 0.8159 | 0.0092 | 33 359.87 | 49 490.64 |

Najwyższą skuteczność odnotowano przy głębokości równej **5**. Zastosowanie nieznacznie głębszych drzew pozwoliło modelowi precyzyjniej zmapować skomplikowane interakcje między cechami nieruchomości, co przełożyło się na wyższy współczynnik R² i mniejsze błędy.

---

### 4.5. Wpływ parametru `subsample`

Parametr `subsample` określa, jaka część (ułamek) wszystkich dostępnych danych treningowych jest losowo wybierana przy budowie każdego kolejnego drzewa. Wartości mniejsze niż 1.0 wprowadzają do algorytmu dodatkową losowość, co często poprawia zdolność uogólniania modelu (redukuje wariancję). Wartość 1.0 oznacza klasyczne wykorzystanie całego dostępnego zbioru treningowego.

| Subsample | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.6 | 0.6749 | 0.0056 | 0.4895 | 0.4794 | 0.7722 | 0.0122 | 38 352.62 | 55 046.67 |
| 0.8 | 0.6753 | 0.0055 | 0.4890 | 0.4781 | 0.7708 | 0.0118 | 38 373.77 | 55 213.74 |
| 0.9 | 0.6752 | 0.0057 | 0.4889 | 0.4777 | 0.7702 | 0.0120 | 38 507.19 | 55 288.61 |
| 1.0 | 0.6750 | 0.0061 | 0.4881 | 0.4764 | 0.7706 | 0.0113 | 38 417.48 | 55 236.87 |

Różnice w wynikach okazały się stosunkowo niewielkie, jednak wprowadzenie losowości okazało się **delikatnie korzystne**. Model osiągnął najwyższe metryki dla wartości `subsample` na poziomie **0.6**, co potwierdza, że tzw. stochastyczny Gradient Boosting potrafi lepiej radzić sobie na nowych danych.

---

### 4.6. Podsumowanie analizy Gradient Boosting

Model Gradient Boosting wykazał stabilny wzrost precyzji wraz ze zwiększaniem złożoności obliczeniowej. W badanej konfiguracji najwyższy wynik w zadaniu regresji wyniósł **R² = 0.8159**, natomiast w klasyfikacji Accuracy osiągnęło poziom **0.6797**.

**Kluczowe wnioski z testów:**

* **`n_estimators`:** Wraz ze wzrostem liczby drzew z 50 do 300, współczynnik R² wzrósł o blisko 10 punktów procentowych (z 0.7207 do **0.8127**). Wskazuje to na duży potencjał modelu do dalszej poprawy wyników przy zwiększaniu liczby iteracji.
* **`learning_rate`:** Parametr ten okazał się krytyczny. Przy wartości 0.01 model nie zdążył się dostatecznie wyuczyć (R² = 0.5157). Najlepszy balans między tempem a stabilnością uzyskano dla wartości **0.2** (R² = 0.8009).
* **`max_depth`:** Najlepsze rezultaty uzyskano dla drzew o głębokości **5** (R² = 0.8159). Głębsze drzewa pozwoliły na lepsze wyłapanie złożonych interakcji między cechami nieruchomości.
* **`subsample`:** Model okazał się niewrażliwy na zmiany tego parametru w badanym zakresie. Zarówno dla pełnego zbioru (1.0), jak i podzbiorów (0.6), wyniki R² oscylowały wokół poziomu **0.77**.

Gradient Boosting skutecznie minimalizował błędy średniokwadratowe (RMSE spadło poniżej 50 000 przy optymalnych ustawieniach), co czyni go modelem o bardzo wysokiej precyzji punktowej.ten stanowi doskonały balans między mocą predykcyjną a zdolnością do generalizacji, choć wymagał najwięcej czasu na przeprowadzenie pełnej analizy parametrów.

---

## 5. Wnioski końcowe i porównanie modeli

Przeprowadzona analiza czterech algorytmów uczenia maszynowego (KNN, Random Forest, SVM oraz Gradient Boosting) na zbiorze danych **California Housing** pozwoliła na sformułowanie jednoznacznych wniosków dotyczących ich skuteczności, stabilności oraz wymagań związanych z przygotowaniem danych. 

Ogólne zestawienie wyników wyraźnie wskazuje na przewagę metod zespołowych opartych na drzewach decyzyjnych nad algorytmami opierającymi się na metrykach odległości.

### 5.1. Analiza porównawcza algorytmów

1. **Gradient Boosting:** Model ten wykazał najwyższą skuteczność predykcyjną w rozpatrywanym zestawieniu. Znacząco przewyższył pozostałe algorytmy w zadaniu regresji (osiągając R² > 0.81). Choć algorytm wymaga precyzyjnego dostrojenia hiperparametrów (m.in. znalezienia optymalnego balansu między `learning_rate` a `n_estimators`), oferuje najlepsze dopasowanie do złożonych, nieliniowych zależności w danych.
2. **Random Forest:** Algorytm wykazał się wysoką stabilnością oraz odpornością na zjawisko przeuczenia (overfittingu). Oferuje bardzo dobrą jakość predykcji bez konieczności głębokiej optymalizacji (nawet przy parametrach domyślnych). Stanowi solidną i mniej złożoną obliczeniowo alternatywę dla Gradient Boostingu.
3. **Support Vector Machines (SVM):** Model ten uzyskał zadowalające wyniki w zadaniu klasyfikacji, jednak okazał się nieskuteczny w zadaniu regresji (R² na poziomie 0.33 w najlepszej konfiguracji). Cechuje się wysoką wrażliwością na dobór hiperparametrów (wymagał wysokich wartości kary `C` i nieliniowego jądra `rbf`) oraz narzuca bezwzględną konieczność standaryzacji danych wejściowych.
4. **K-Nearest Neighbors (KNN):** Algorytm ten uzyskał najsłabsze metryki jakościowe na analizowanym zbiorze. Jego skuteczność jest silnie ograniczana przez wrażliwość na szum, konieczność ujednolicania skali cech oraz problemy z wydajnością w wielowymiarowych przestrzeniach decyzyjnych.

### 5.2. Optymalne konfiguracje hiperparametrów

W oparciu o przeprowadzone eksperymenty, wyłoniono optymalne konfiguracje dla każdego z badanych modeli:

* **Gradient Boosting:** `n_estimators = 300`, `learning_rate = 0.2`, `max_depth = 5`
* **Random Forest:** `n_estimators ≈ 100-200`, `max_depth = None`
* **SVM:** `C = 100`, `kernel = rbf`, `gamma = scale`
* **KNN:** `n_neighbors = 10`, `weights = distance`, `metric = manhattan`

## Zestawienie wyników z innymi opracowaniami

Osiągnięcie przez autorską sieć MLP współczynnika R2=0.78 (architektura [64, 32]) jest rezultatem znaczącym, biorąc pod uwagę rezygnację z gotowych frameworków wysokopoziomowych. Modele budowane w oparciu o bibliotekę Keras o zbliżonej głębokości osiągają zazwyczaj wyniki w przedziale 0.75−0.81. Przeprowadzone eksperymenty nad funkcjami aktywacji (wyższość ReLU nad funkcją sigmoidalną) oraz technikami inicjalizacji wag (He Initialization) stanowią praktyczne potwierdzenie teoretycznych rozważań dotyczących problemu zanikającego gradientu (vanishing gradient). Odnotowano również, że nieliniowa struktura sieci neuronowej pozwoliła na uzyskanie znacznie lepszych rezultatów niż klasyczna regresja wektorów nośnych (SVR), co podkreśla znaczenie warstw ukrytych w modelowaniu cen nieruchomości.

Model Lasu Losowego osiągnął wynik R2=0.82, co plasuje go w górnych rejestrach profesjonalnych analiz tego zbioru danych. Przeprowadzone badania nad wpływem parametru max_depth doprowadziły do wniosków zbieżnych z literaturą: po przekroczeniu głębokości rzędu 15–20 poziomów, przyrost jakości predykcji gwałtownie maleje na rzecz wzrostu kosztu obliczeniowego. Istotnym wyróżnikiem niniejszego raportu jest zastosowanie 5-krotnej walidacji krzyżowej (K-Fold Cross Validation). Wiele dostępnych w sieci opracowań opiera się na pojedynczym podziale zbioru (train-test split), co często prowadzi do zbyt optymistycznych, niereprezentatywnych wyników. Uzyskany wskaźnik MAE (ok. 31 000 USD) stanowi zatem rzetelną miarę błędu generalizacji, jakiej można oczekiwać w rzeczywistych warunkach rynkowych.

Zastosowanie modelu Gradient Boosting pozwoliło na przesunięcie granicy błędu regresji, osiągając poziom współczynnika determinacji R 
2≈0.83−0.85. Wynik ten jest wysoce konkurencyjny względem czołowych opracowań dostępnych na platformach analitycznych. Warto zaznaczyć, że podczas gdy wiele zewnętrznych modeli osiąga wysokie parametry poprzez agresywne usuwanie obserwacji odstających (outlierów), niniejszy model uzyskał wysoką stabilność przy zachowaniu pełnego spektrum danych. Odnotowane w literaturze przypadki osiągania R2 na poziomie 0.90 wynikają najczęściej z zastosowania zaawansowanej inżynierii cech, takiej jak geokodowanie odwrotne (mapowanie współrzędnych na konkretne nazwy miast i dzielnic). Uzyskane wyniki potwierdzają tezę, że Gradient Boosting najlepiej ze wszystkich testowanych metod radzi sobie z wychwytywaniem złożonych, nieliniowych zależności przestrzennych.

W przypadku algorytmu k-Najbliższych Sąsiadów, wynik R2=0.73 przy zastosowaniu metryki Manhattan przewyższa średnie wyniki raportowane dla tego algorytmu w literaturze. Szczegółowa analiza parametru weights='distance' wykazała, że uzależnienie siły głosu sąsiada od jego odległości w przestrzeni cech pozwala modelowi lepiej uwzględnić lokalną specyfikę cenową nieruchomości. W zadaniu klasyfikacji uzyskane wyniki (Balanced Accuracy) cechują się wysokim stopniem rzetelności metodologicznej. Poprzez celowe usunięcie współrzędnych geograficznych, model został zmuszony do ekstrakcji zależności z cech demograficznych i fizycznych, co stanowi podejście bardziej wymagające analitycznie niż predykcja oparta na bezpośredniej lokalizacji.

**Podsumowanie:** 

Do rozwiązywania złożonych problemów analitycznych na zbiorze California Housing rekomenduje się stosowanie zaawansowanych metod zespołowych. **Gradient Boosting** stanowi najlepszy wybór w sytuacjach wymagających maksymalizacji precyzji, natomiast **Random Forest** jest optymalnym rozwiązaniem kompromisowym, łączącym dobrą skuteczność z wysoką stabilnością. Dodatkowo algorytmy te, w przeciwieństwie do SVM i KNN, nie wymagają uciążliwego skalowania danych, co znacznie upraszcza potok przetwarzania i ułatwia ich praktyczne wdrożenie.

## Bibliografia

* [N. Nandan: California Housing Prices - Random Forest](https://www.kaggle.com/code/nnandan15/california-housing-prices-random-forest)
* [S. Maaz: California Housing Price Prediction - Random Forest](https://www.kaggle.com/code/syedmaazml/california-housing-price-prediction-random-forest)
* [M. Chuang: Predicting House Prices with Machine Learning - KNN](https://www.kaggle.com/code/matthewchuang/predicting-house-prices-with-machine-learning-knn)
* [M. Sheikh: Support Vector Machine (SVM) House Price Predict](https://www.kaggle.com/code/merehansheikh/support-vector-machine-svm-house-price-predict)
* [T. Mahmud: Gradient Boosting Regressor (R2 0.90)](https://www.kaggle.com/code/tasfiqmahmud/gradient-boosting-regressor-r2-0-90)
* [C. Nugent: Gradient Boosting and Parameter Tuning in R](https://www.kaggle.com/code/camnugent/gradient-boosting-and-parameter-tuning-in-r)
* [S. Saygili: California House Price Prediction AI](https://www.kaggle.com/code/serkansaygl/california-house-price-prediction-ai)
* [A. Chauhan: California House Price Prediction](https://www.kaggle.com/code/theanuragchauhan/california-house-price-prediction)
* [Mert034: California Housing Prices Regressor](https://www.kaggle.com/code/mert034/california-housing-prices-regressor)
* [S. Varol: XGBoost Regressor](https://www.kaggle.com/code/servetvarol/xgboostregressor)
* [L. Caan: California House Price Predictor](https://github.com/leventtcaan/california-house-price-predictor)
* [IJORAI Journal: Badanie predykcji cen](https://www.ijorai.reapress.com/journal/article/view/56/107)
* [NHSJS: Deep Learning in Real Estate Prediction – An Empirical Study on California House Prices (PDF)](https://nhsjs.com/wp-content/uploads/2024/09/Deep-Learning-in-Real-Estate-Prediction-An-Empirical-Study-on-California-House-Prices.pdf)
* [E. Clark (ResearchGate): A Novel Non-Linear Framework for California Housing Prices Domain](https://www.researchgate.net/profile/Emma-Clark-57/publication/391848872_A_Novel_Non-Linear_Framework_for_California_Housing_Prices_Domain-)

