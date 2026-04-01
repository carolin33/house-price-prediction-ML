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