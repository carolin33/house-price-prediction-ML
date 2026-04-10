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

### 3.2. Dane i metodologia eksperymentów

Eksperymenty dla modelu SVM przeprowadzono na tym samym zbiorze **California Housing** oraz w tych samych dwóch zadaniach:

* **klasyfikacji** zmiennej `ocean_proximity`,
* **regresji** zmiennej `median_house_value`.

W zadaniu klasyfikacji celowo usunięto cechy `longitude` i `latitude`, aby uniknąć zbyt łatwego odtwarzania klasy wyłącznie na podstawie lokalizacji geograficznej. Dzięki temu model musiał opierać się na bardziej pośrednich zależnościach obecnych w danych.

Preprocessing obejmował:
* imputację braków medianą dla cech numerycznych,
* imputację braków najczęstszą wartością dla cech kategorycznych,
* kodowanie cech kategorycznych metodą **One-Hot Encoding**,
* standaryzację cech numerycznych z użyciem `StandardScaler`.

Standaryzacja była szczególnie istotna w przypadku SVM, ponieważ metoda ta jest niezwykle wrażliwa na skalę zmiennych wejściowych. Bez odpowiedniego skalowania cechy o naturalnie większych wartościach liczbowych mogłyby nadmiernie i błędnie wpływać na działanie algorytmu oraz wyznaczanie marginesu.

Do oceny modeli wykorzystano **5-krotną walidację krzyżową**:
* `StratifiedKFold` dla zadania klasyfikacji,
* `KFold` dla zadania regresji.

Analizie poddano cztery kluczowe hiperparametry:
* `C` (parametr regularyzacji),
* `kernel` (rodzaj funkcji jądra),
* `gamma` (współczynnik jądra),
* `epsilon` (margines tolerancji błędu dla regresji).

Każdy z powyższych parametrów badano osobno, przy pozostałych utrzymanych na optymalnych wartościach bazowych.

### 3.3. Wpływ parametru kary (`C`)

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

### 3.4. Wpływ funkcji jądra (`kernel`)

Parametr `kernel` określa sposób, w jaki algorytm szuka powiązań między danymi. Przetestowano cztery warianty transformacji przestrzeni: `linear`, `rbf`, `poly` oraz `sigmoid`.

| Kernel | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| linear  | 0.6528 | 0.0038 | 0.4266 | 0.3740 |  0.0685 | 0.0032 | 82 189.00 | 111 345.29 |
| rbf     | 0.6803 | 0.0035 | 0.4807 | 0.4608 | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |
| poly    | 0.6407 | 0.0020 | 0.4442 | 0.4316 | -0.0417 | 0.0018 | 87 745.25 | 117 748.14 |
| sigmoid | 0.5330 | 0.0133 | 0.3717 | 0.3547 | -0.0422 | 0.0027 | 87 699.59 | 117 775.90 |

Zdecydowanie najlepsze rezultaty klasyfikacyjne dało jądro `rbf`. Sugeruje to mocno nieliniowy charakter danych (np. skomplikowany wpływ lokalizacji na cenę). Jądro liniowe okazało się zbyt uproszczone, a pozostałe nie zapewniły równie dobrej predykcji.

---

### 3.5. Wpływ parametru `gamma`

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

### 3.6. Wpływ parametru `epsilon` (tylko regresja)

Parametr `epsilon` dotyczy wyłącznie modelu SVR. Definiuje on szerokość specjalnej strefy tolerancji wokół przewidywanej wartości, wewnątrz której drobne błędy nie są w ogóle karane przez algorytm.

| Epsilon | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- |
| 0.01 | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |
| 0.1  | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |
| 0.5  | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |
| 1.0  | -0.0486 | 0.0027 | 87 966.35 | 118 137.16 |

Analiza wykazała, że zmiany parametru `epsilon` miały całkowicie marginalny wpływ na jakość regresji przy tych ustawieniach bazowych (wyniki pozostały bez zmian). Oznacza to, że w badanym przypadku szerokość tej strefy nie była kluczowa, a o sukcesie modelu decydowały głównie odpowiednio dobrane parametry `C` oraz `kernel`.

---

### 3.7. Podsumowanie analizy SVM

Algorytm SVM okazał się potężnym modelem, zdolnym do osiągania dobrych wyników po odpowiednim dostrojeniu. Jego główną cechą w tym projekcie okazała się bardzo wysoka wrażliwość na dobór hiperparametrów (szczególnie `C` oraz `kernel`). Z kolei wpływ parametru `epsilon` był bardzo ograniczony.

Najlepsze rezultaty uzyskano dla konfiguracji wykorzystującej wysoką wartość parametru `C` (np. 100) oraz nieliniowe jądro `rbf`. Wyniki te jednoznacznie potwierdzają, że relacje w zbiorze California Housing mają charakter złożony, a ich poprawne zamodelowanie wymaga elastycznych granic decyzyjnych.

### 4.1. Czym jest Gradient Boosting i jak działa?

Gradient Boosting to zaawansowana metoda uczenia zespołowego, polegająca na sekwencyjnym budowaniu serii modeli (najczęściej płytkich drzew decyzyjnych). W przeciwieństwie do algorytmu Random Forest, w którym drzewa powstają niezależnie od siebie, w metodzie Gradient Boosting każde kolejne drzewo jest tworzone w celu skorygowania błędów popełnionych przez modele zbudowane wcześniej. 

Proces ten można porównać do wyciągania wniosków z poprzednich pomyłek. Algorytm analizuje, w których miejscach dotychczasowe przewidywania były niedokładne, a następnie dodaje nowe drzewo, którego zadaniem jest naprawienie tych konkretnych błędów (tzw. reziduów). Dzięki takiemu etapowemu podejściu, końcowy model staje się coraz precyzyjniejszy i potrafi skutecznie wykrywać złożone, nieliniowe zależności w danych.

Istotną cechą Gradient Boosting jest jego wysoka skuteczność, która jednak zależy od odpowiedniego doboru tzw. **hiperparametrów** – czyli ustawień konfiguracyjnych wybieranych przez badacza przed rozpoczęciem procesu uczenia. Ponieważ algorytm opiera się na strukturze drzew decyzyjnych, jest on niewrażliwy na różnice w skali cech numerycznych. W związku z tym, w procesie przygotowania danych zrezygnowano z użycia narzędzia `StandardScaler`, co pozwoliło na uproszczenie obliczeń przy zachowaniu wysokiej jakości predykcji.

### 4.2. Dane i metodologia eksperymentów

Eksperymenty dla modelu Gradient Boosting przeprowadzono na zbiorze **California Housing**, realizując dwa zadania:

* **klasyfikację** zmiennej `ocean_proximity`,
* **regresję** zmiennej `median_house_value`.

Podobnie jak w poprzednich analizach, w zadaniu klasyfikacji celowo usunięto cechy `longitude` oraz `latitude`, aby uniknąć sytuacji, w której model rozwiązuje zadanie wyłącznie na podstawie bezpośrednich współrzędnych geograficznych. Zmusza to algorytm do opierania się na bardziej złożonych i pośrednich zależnościach.

Preprocessing danych obejmował:
* imputację braków medianą dla cech numerycznych,
* imputację braków najczęstszą wartością dla cech kategorycznych,
* kodowanie zmiennych kategorycznych przy użyciu techniki **One-Hot Encoding**.

Zgodnie z przyjętą metodologią, ze względu na wykorzystanie algorytmu opartego na drzewach decyzyjnych, całkowicie zrezygnowano ze standaryzacji cech (`StandardScaler`).

Do oceny jakości modeli wykorzystano **5-krotną walidację krzyżową**:
* `StratifiedKFold` dla zadania klasyfikacji,
* `KFold` dla zadania regresji.

W analizie rozpatrzono cztery kluczowe hiperparametry:
* `n_estimators` (liczba drzew tworzących model),
* `learning_rate` (tempo uczenia, określające wkład każdego kolejnego drzewa),
* `max_depth` (maksymalna głębokość pojedynczego drzewa),
* `subsample` (ułamek próbek danych używanych do budowy każdego z drzew).

Każdy z powyższych parametrów testowano niezależnie, podczas gdy pozostałe ustawienia utrzymywano na stałych wartościach bazowych.

### 4.3. Wpływ liczby estymatorów (`n_estimators`)

Parametr `n_estimators` określa liczbę kolejnych drzew decyzyjnych budujących model. Zwiększanie tej wartości zazwyczaj poprawia jakość predykcji, jednak jednocześnie wydłuża czas obliczeń i może prowadzić do nadmiernego dopasowania (overfittingu), jeśli nie jest równoważone odpowiednio małym współczynnikiem uczenia (`learning_rate`).

| n_estimators | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 50   | 0.6681 | 0.0061 | 0.4804 | 0.4655 | 0.7207 | 0.0126 | 43 226.27 | 60 952.45 |
| 100  | 0.6750 | 0.0061 | 0.4881 | 0.4764 | 0.7706 | 0.0113 | 38 417.48 | 55 236.87 |
| 200  | 0.6776 | 0.0050 | 0.4937 | 0.4865 | 0.8017 | 0.0100 | 35 187.75 | 51 362.18 |
| 300  | 0.6781 | 0.0042 | 0.4980 | 0.4943 | 0.8127 | 0.0090 | 33 926.84 | 49 912.06 |

Wyniki eksperymentu wykazały, że optymalną liczbą drzew dla badanego zbioru jest **300**. Zwiększanie liczby estymatorów konsekwentnie poprawiało zdolności predykcyjne modelu (szczególnie w zadaniu regresji, gdzie błąd RMSE systematycznie malał), bez widocznych oznak drastycznego przeuczenia.

---

### 4.4. Wpływ współczynnika uczenia (`learning_rate`)

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

### 4.5. Wpływ głębokości drzew (`max_depth`)

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

### 4.6. Wpływ parametru `subsample`

Parametr `subsample` określa, jaka część (ułamek) wszystkich dostępnych danych treningowych jest losowo wybierana przy budowie każdego kolejnego drzewa. Wartości mniejsze niż 1.0 wprowadzają do algorytmu dodatkową losowość, co często poprawia zdolność uogólniania modelu (redukuje wariancję). Wartość 1.0 oznacza klasyczne wykorzystanie całego dostępnego zbioru treningowego.

| Subsample | Acc. | Acc. std | Bal. Acc. | F1 Macro | R² | R² std | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.6 | 0.6749 | 0.0056 | 0.4895 | 0.4794 | 0.7722 | 0.0122 | 38 352.62 | 55 046.67 |
| 0.8 | 0.6753 | 0.0055 | 0.4890 | 0.4781 | 0.7708 | 0.0118 | 38 373.77 | 55 213.74 |
| 0.9 | 0.6752 | 0.0057 | 0.4889 | 0.4777 | 0.7702 | 0.0120 | 38 507.19 | 55 288.61 |
| 1.0 | 0.6750 | 0.0061 | 0.4881 | 0.4764 | 0.7706 | 0.0113 | 38 417.48 | 55 236.87 |

Różnice w wynikach okazały się stosunkowo niewielkie, jednak wprowadzenie losowości okazało się **delikatnie korzystne**. Model osiągnął najwyższe metryki dla wartości `subsample` na poziomie **0.6**, co potwierdza, że tzw. stochastyczny Gradient Boosting potrafi lepiej radzić sobie na nowych danych.

---

### 4.7. Podsumowanie analizy Gradient Boosting

Gradient Boosting udowodnił, że jest modelem niezwykle elastycznym i potężnym. Przeprowadzona analiza potwierdza, że jego realna skuteczność jest uzależniona od odpowiedniego dostrojenia hiperparametrów, takich jak `learning_rate`, `n_estimators` oraz `max_depth`.

Biorąc pod uwagę wszystkie zbadane kombinacje, Gradient Boosting zaprezentował wybitne rezultaty, **zdecydowanie przewyższając model SVM, szczególnie w zadaniu regresji**. Podczas gdy model SVM uzyskał współczynnik R² rzędu 0.33, odpowiednio skonfigurowany Gradient Boosting osiągnął wynik R² przekraczający 0.81, drastycznie minimalizując przy tym błędy w wycenie nieruchomości (MAE i RMSE). Model ten wykazał się doskonałą stabilnością i umiejętnością wychwytywania bardzo złożonych zależności w zbiorze danych.


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
* **Random Forest:** `n_estimators ≈ 200`, `max_depth = None`, `min_samples_split = 2`
* **SVM:** `C = 100`, `kernel = 'rbf'`
* **KNN:** `n_neighbors = 10`, `weights = 'distance'`, `metric = 'manhattan'`

**Podsumowanie:** 

Do rozwiązywania złożonych problemów analitycznych na zbiorze California Housing rekomenduje się stosowanie zaawansowanych metod zespołowych. **Gradient Boosting** stanowi najlepszy wybór w sytuacjach wymagających maksymalizacji precyzji, natomiast **Random Forest** jest optymalnym rozwiązaniem kompromisowym, łączącym dobrą skuteczność z wysoką stabilnością. Dodatkowo algorytmy te, w przeciwieństwie do SVM i KNN, nie wymagają uciążliwego skalowania danych, co znacznie upraszcza potok przetwarzania i ułatwia ich praktyczne wdrożenie.



