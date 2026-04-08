# Projekt Elementy sztucznej inteligencji

## Wstęp i opis badanych problemów

## Przegląd literatury

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

**Analiza:** Widać wyraźny, monotoniczny spadek błędu RMSE i wzrost R² wraz ze wzrostem liczby epok. Przyrosty jednak maleją - różnica między 10 a 30 epokami (~3 000 RMSE) jest znacznie większa niż między 100 a 200 (~1 200 RMSE). To typowe zachowanie algorytmu gradientowego: na początku optymalizacja przebiega szybko w kierunku minimum, a potem zwalnia w okolicach punktu zbieżności.

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

### Opis wybranych algorytmów

### Wyniki dla problemu regresyjnego

### Wyniki dla problemu klasyfikacji

## Podsumowanie i wnioski końcowe