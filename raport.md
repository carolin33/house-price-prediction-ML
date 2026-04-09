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

## 1. Model Random Forest

### 1.1. Czym jest Random Forest i dlaczego działa?

Random Forest (las losowy) to metoda uczenia zespołowego (*ensemble learning*), która buduje wiele drzew decyzyjnych i łączy ich odpowiedzi w finalną predykcję. Idea kluczowa to zasada tłumu: wiele modeli popełniających różne błędy, połączonych razem, daje lepszy wynik niż pojedynczy model. Właśnie to zapewnia losowość wbudowana w algorytm.

Każde drzewo w lesie jest trenowane na innym, losowo wylosowanym podzbiorze danych treningowych (*bootstrap sampling*). Dodatkowo przy każdym podziale węzła drzewo wybiera najlepszą cechę nie spośród wszystkich cech, ale spośród losowo wybranego ich podzbioru. Ten mechanizm, kontrolowany przez parametr `max_features`, wymusza różnorodność drzew: każde z nich uczy się nieco innych wzorców.

W zadaniu klasyfikacji wynik końcowy to głosowanie większościowe — klasa wskazana przez największą liczbę drzew. W zadaniu regresji wynikiem jest średnia z predykcji wszystkich drzew. To uśrednianie jest kluczowe, ponieważ skutecznie redukuje wariancję modelu i ogranicza ryzyko przeuczenia, które stanowi główną słabość pojedynczych drzew decyzyjnych.

---

### 1.2. Dane i metodologia eksperymentów

Eksperymenty przeprowadzono na zbiorze **California Housing**, zawierającym informacje o nieruchomościach z Kalifornii. Zbiór liczy około 20 000 obserwacji i obejmuje cechy takie jak m.in. mediana dochodu mieszkańców (`median_income`), liczba pokoi, wiek budynku, populacja obszaru czy bliskość oceanu (`ocean_proximity`).

Przeprowadzono dwa oddzielne zadania uczenia maszynowego:

- **klasyfikację** zmiennej `ocean_proximity` (5 klas: `NEAR BAY`, `NEAR OCEAN`, `INLAND`, `ISLAND`, `<1H OCEAN`),
- **regresję** zmiennej `median_house_value` (mediana wartości nieruchomości).

Ważna decyzja projektowa: w zadaniu klasyfikacji celowo usunięto cechy `longitude` i `latitude`. Gdyby pozostawić współrzędne geograficzne, model mógłby bardzo łatwo rozwiązać zadanie bezpośrednio po lokalizacji. Taki wynik byłby wprawdzie numerycznie dobry, ale metodologicznie mało interesujący, ponieważ model nie musiałby uczyć się zależności pośrednich wynikających z cech nieruchomości.

Preprocessing obejmował:

- uzupełnianie brakujących wartości medianą dla cech numerycznych,
- uzupełnianie braków najczęstszą wartością dla cech kategorycznych,
- kodowanie zmiennych kategorycznych metodą **One-Hot Encoding**.

Walidację przeprowadzono przy użyciu **5-krotnej walidacji krzyżowej**:

- `StratifiedKFold` dla klasyfikacji — aby zachować podobny rozkład klas w każdym foldzie,
- `KFold` dla regresji.

Każdy parametr był testowany osobno, przy pozostałych utrzymanych na wartościach bazowych:

- **klasyfikacja:** `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `max_features="sqrt"`
- **regresja:** `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `max_features=1.0`

---

### 1.3. Wpływ liczby drzew (`n_estimators`)

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

### 1.4. Wpływ głębokości drzew (`max_depth`)

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

### 1.5. Wpływ minimalnej liczby próbek do podziału (`min_samples_split`)

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

### 1.6. Wpływ liczby rozważanych cech przy podziale (`max_features`)

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

### 1.7. Podsumowanie analizy Random Forest

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

W kodzie zastosowano `StandardScaler` wewnątrz `Pipeline`, co jest metodologicznie poprawne, ponieważ eliminuje ryzyko **data leakage** — skalowanie jest dopasowywane wyłącznie na częściach treningowych w ramach walidacji krzyżowej.



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

### 2.6. Podsumowanie analizy KNN

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

## 3. Porównanie modeli i wnioski końcowe

### 3.1. Zestawienie najlepszych wyników

| Model / Zadanie | Acc. / R² | Bal. Acc. | F1 Macro / MAE | RMSE | Stabilność |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RF – Klasyfikacja | 0.6821 | 0.5037 | 0.5045 | — | Wysoka |
| KNN – Klasyfikacja | 0.6698 | 0.4918 | 0.4932 | — | Średnia |
| RF – Regresja | R² = 0.8262 | — | MAE = 31 353 | 48 082 | Wysoka |
| KNN – Regresja | R² = 0.7350 | — | MAE = 39 884 | 59 389 | Średnia |

---

### 3.2. Dlaczego Random Forest wypada lepiej?

Przewaga Random Forest wynika z fundamentalnych różnic między modelami.

**Po pierwsze**, RF buduje jawny model predykcyjny, czyli zbiór reguł zapisanych w strukturach drzewiastych. KNN nie buduje modelu w klasycznym sensie — jedynie przechowuje dane i porównuje nowe próbki z zapisanymi przykładami.

**Po drugie**, RF lepiej radzi sobie z cechami częściowo nieistotnymi lub zaszumionymi. Dzięki losowemu wyborowi cech przy splitach nie każda cecha wpływa na każdą decyzję. KNN natomiast uwzględnia wszystkie cechy jednocześnie przy obliczaniu odległości, więc nawet cechy mniej przydatne mogą wprowadzać szum.

**Po trzecie**, Random Forest redukuje wariancję przez uśrednianie dużej liczby drzew. KNN również pewnym sensie uśrednia informacje, ale tylko lokalnie i tylko po sąsiadach, przez co jest znacznie bardziej wrażliwy na lokalny rozkład danych.

**Po czwarte**, RF lepiej radzi sobie z nieliniowymi zależnościami i interakcjami między cechami. KNN opiera się wyłącznie na geometrii przestrzeni wejściowej, a nie na wyuczonych strukturach relacji.

---

### 3.3. Wniosek końcowy

Na zbiorze **California Housing** model **Random Forest** okazał się wyraźnie lepszym rozwiązaniem niż **KNN**. Oferuje wyższą jakość predykcji, większą stabilność i mniejszą wrażliwość na dobór pojedynczych hiperparametrów.

Najlepsze konfiguracje z analizy:

**Random Forest**
- `n_estimators ≈ 200`
- `max_depth = None` lub `25`
- `min_samples_split = 2`
- `max_features = 0.5–0.9`

**KNN**
- `n_neighbors = 10`
- `weights = 'distance'`
- `metric = 'manhattan'`

Podsumowując:  
**Random Forest jest bardziej odpowiednim modelem dla tego zbioru**, zwłaszcza w zadaniu regresji, gdzie uzyskał bardzo wyraźną przewagę.  
**KNN** pozostaje algorytmem prostym, intuicyjnym i wartościowym dydaktycznie, ale na tym konkretnym zbiorze ograniczają go wrażliwość na definicję odległości, konieczność skalowania oraz pogarszająca się jakość w wyższych wymiarach.

## Podsumowanie i wnioski końcowe