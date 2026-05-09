# Projekt Elementy sztucznej inteligencji

## 1. Wstęp i opis badanych problemów

W niniejszym projekcie porównano skuteczność wybranych metod uczenia maszynowego w zadaniu regresji. Do analizy wykorzystano zbiór danych California Housing, który jest często używany do testowania modeli predykcyjnych. Celem projektu było sprawdzenie, jak zmiany wybranych hiperparametrów wpływają na jakość działania modeli oraz ich zdolność do przewidywania wartości na nowych danych. Chodziło więc nie tylko o uzyskanie jak najlepszego wyniku, ale też o lepsze zrozumienie, jak zachowują się różnee metody przy zmianie ustawień.

W projekcie porównano kilka metod:

- sztuczne sieci neuronowe (MLP),
- lasy losowe (Random Forest),
- algorytm k najbliższych sąsiadów (k-NN),
- Gradient Boosting,
- SVM.

Wszystkie te metody wykorzystano do rozwiązania problemu regresji, czyli przewidywania wartości zmiennej median_house_value. Zadanie polegało na estymacji mediany cen nieruchomości na podstawie dostępnych cech opisujących dany obszar. Nie jest to łatwy problem, bo ceny nieruchomości zależą od wielu różnych czynników, a zależności między nimi często są nieliniowe.

Do oceny jakości modeli wykorzystano miary MAE,RMSE oraz R². Dzięki nim można było sprawdzić, jak duży jest błąd przewidywań i jak dobrze model dopasowuje się do danych.  

## 2. Przegląd literatury

Analiza rozwiązań udostępnionych na platformach takich jak kaggle czy github pokazuje, że w problemach estymacji cen nieruchomości w Kalifornii dominują algorytmy takie jak drzewa decyzyjne, Random Forest oraz XGBoost. Osiągają R² na poziomie 0.8-0.84. Rozwiązania oparte o sztuczne sieci neuronowe są również popularne, ale badacze podkreślają, że aby pokonać regresję liniową, wymagają one głębszych architektur oraz rygorystycznego skalowania danych.  

Niezależnie od wybranej architektury, twórcy modeli są zgodni co do hierarchii zmiennych. Median_income jest kluczową cechą, która ma znacznie większą moc przewidywania niż parametry fizyczne budynków. Ciekawym wątkiem jest fakt, że zbiór danych California Housing ma ustaloną górną granicę ceny na 500 000$. Wielu badaczy z tego powodu decyduje się na usunięcie tych rekordów.

## 3. Sztuczne Sieci Neuronowe

### 3.1 Czym są Sztuczne Sieci Neuronowe

Sztuczne Sieci Neuronowe (SSN) to zaawansowane modele matematyczne i algorytmiczne, inspirowane budową biologiczną ludzkiego mózgu. MLP składa się z warstwy wejściowej, ukrytej i wyjściowej. Proces uczenia sieci opiera się na dwóch głównych mechanizmach: propagacji wprzód i propagacji wstecz. W propagacji wprzód dane przechodzą przez sieć, a każdy neuron sumuje iloczyn wejść i wag, po czym przepuszcza wynik przez nieliniową funkcję aktywacji. Funckje te są w stanie modelować nieliniowe zależności między danymi. W propagacji wstecznej po wygenerowaniu wyniku, sieć oblicza błąd w stosunku do wartości rzeczywistej, a następnie wykorzystując algorytmy optymalizacyjne takie jak spadek gradientu, sieć cofa się i koryguje wagi w taki sposób, aby w kolejnych epokach ten błąd był jak najmniejszy.

### 3.2 Analiza wpływu parametrów na skuteczność działania sieci

Model bazowy:

- Proporcja zbioru testowego = 0.2
- Architektura warstw ukrytych = [32]
- Funkcja aktywacji = relu
- Skala inicjalizacji wag = 1.0
- Współczynnik uczenia = 0.001
- Liczba epok = 100
- Liczba powtórzeń = 3

Badanie przeprowadzono, testując kolejno jeden parametr w każdym kroku, na ten który wykazał się najmniejszym RMSE. Po znalezieniu najlepszego wyniku, skrypt nadpisywał model bazowy optymalną wartością. Każdy eksperyment powtarzano trzykrotnie, co było absolutnym minimum, w celu uśrednienia wyników i wyeliminowania wpływu losowej inicjalizacji wag oraz podziału danych. Taki sposób oparty jedynie na minimalizacji błędu, miał jeden zasadniczy problem. Skrypt, działający automatycznie, mógł wybrać model przeuczony, co dostrzegliśmy na początku analizy. Z tego powodu wprowadziliśmy do kodu dodatkowy bezpiecznik. Zanim skrypt wybrał parametr z najniższym RMSE, sprawdzał różnicę między wynikiem R² na zbiorze treningowym a testowym. Ustaliliśmy próg na poziomie 3%. Jeśli różnica przekraczała tę wartość, model automatycznie był odrzucany.

### 3.2.1 Wielkość próby testowej (test_ratio)

W pierwszym kroku zbadaliśmy wpływ proporcji podziału danych na zbiór uczący i testowy. Testowano wartości: 0.1, 0.15, 0.2, 0.3, 0.4.

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| 0.1 | 54512.81 | 55426.59 | 0.7763 | 0.7724 | 0.7704 |
| 0.15 | 54178.14 | 54361.03 | 0.7795 | 0.7779 | 0.7778 |
| 0.2 | 54478.91 | 55568.48 | 0.7769 | 0.7681 | 0.7810 |
| 0.3 | 54625.10 | 55421.69 | 0.7755 | 0.7699 | 0.7701 |
| 0.4 | 55518.36 | 56628.78 | 0.7672 | 0.7607 | 0.7695 |

Najlepszym wynikiem okazała się proporcja 0.15, osiągając najniższy błąd na zbiorze testowym RMSE = 54361.03. Choć w uczeniu maszynowym najczęściej stosuje się podział 80/20, który u nas zdobył niewiele gorszy wynik RMSE = 55568.48, to w tym eksperymencie zmniejszenie zbioru testowego, prawdopodobnie pozwoliło na zasilenie sieci dodatkową pulą danych treningowych. Powiększenie zbioru testowego do wartości 0.3, 0.4 powodowało spadek jakości modelu. Ze względu na te wyniki, to właśnie wartość 0.15 została wybrana do dalszych eksperymentów.

### 3.2.2 Głębokość sieci liczba warstw ukrytych

W drugim etapie badaliśmy wpływ głebokości sieci. Testowaliśmy od 0 do 4 warstw ukrytych o stałej liczbie neuronów 16.

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
| --- | --- | --- | --- | --- | --- |
| [] (0 warstw) | 69046.58 | 68877.29 | 0.6418 | 0.6434 | 0.6493 |
| [16] | 55998.80 | 55825.74 | 0.7644 | 0.7657 | 0.7613 |
| [16, 16] | 52655.37 | 53031.54 | 0.7917 | 0.7886 | 0.7900 |
| [16, 16, 16] | 51461.20 | 52513.99 | 0.8010 | 0.7927 | 0.7933 |
| [16, 16, 16, 16] | 50824.52 | 52168.90 | 0.8059 | 0.7954 | 0.7962 |

![Wykres głębokości](wykres_depth.png)

Przy braku warstw ukrytych widać, że model osiąga najsłabszy wynik i wyjaśnia jedynie 64% zmienności ceny.
Wprowadzenie już pierwszej warstwy ukrytej [16] znacząco poprawia wyniki, skok 12 punktów procentowych R². Kolejne warstwy poprawiają wyniki, ale nie tak drastycznie jak pierwszy przeskok. Zwycięzcą okazał się model z 4 warstwami ukrytymi, który osiągnął najlepszy wynik 52168.90 RMSE, dlatego też ta architektura 4-warstwowa została wybrana do dalszej optymalizacji.

### 3.2.3 Topologia sieci 

W trzecim etapie badania testowaliśmy rozkład neuronów w naszej 4-warstwowej sieci. Testowaliśmy głównie strukturę, w której liczba neuronów maleje w głąb sieci, ale również taką gdzie liczba neuronów była stała.

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| [8, 4, 4, 2] | 76718.92 | 76621.35 | 0.5013 | 0.5040 | 0.7533 |
| [16, 8, 4, 2] | 52779.60 | 53684.18 | 0.7907 | 0.7834 | 0.7866 |
| [32, 16, 8, 4] | 50008.41 | 51922.69 | 0.8121 | 0.7974 | 0.7969 |
| [64, 32, 16, 8] | 47788.03 | 51787.94 | 0.8284 | 0.7984 | 0.7978 |
| [32, 32, 32, 32] | 48375.93 | 51682.46 | 0.8242 | 0.7992 | 0.8002 |

Wąska topologia sieci [8, 4, 4, 2] okazała się zdecydowanie zbyt wąska, nie była w stanie przetworzyć naszych danych, osiągając wynik 50% R². Minimalne zwiększenie do [16, 8, 4, 2] podniosło wynik o 28 punktów procentowych R². Architektura [32, 32, 32, 32] osiągnęła najlepszy wynik R² = 0.7992, przekraczając w pojedynczej próbie barierę 0.8. Pozostałe warianty, jak [32,16,8,4] i [64,32,16,8] uzyskały zbliżone, ale ostatecznie słabsze wyniki pod względem RMSE. Ostatecznie algorytm wybrał układ o stałej szerokości [32,32,32,32], ponieważ miał on najlepszą celność RMSE = 51682.46.

### 3.2.4 Funkcja aktywacji

W czwartym kroku sprawdziliśmy, funkcję aktywacji. Testowaliśmy funkcje: ReLU, Leaky ReLU, Tanh i Sigmoid.

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| relu | 48375.93 | 51682.46 | 0.8242 | 0.7992 | 0.8002 |
| leaky_relu | 47822.17 | 51219.45 | 0.8282 | 0.8028 | 0.8035 |
| tanh | 49221.07 | 50906.53 | 0.8180 | 0.8052 | 0.8084 |
| sigmoid | 59534.01 | 58378.58 | 0.7337 | 0.7438 | 0.7437 |

Najsłabszy wynik uzyskała funkcja Sigmoid średni test R² = 0.7438. Najciekawszym wynikiem tego etapu była rywalizacja między ReLU a Tanh. Funkcja ReLU jest standardem w głębokim uczeniu, jednak w naszym eksperymencie funkcja Tanh osiągnęła nieznacznie lepszy wynik  R² = 0.8052. Różnica RMSE wyniosła między funkcjami 775,93 na przewagę Tanh. Z tego względu to funkcja Tanh została wybrana do dalszej optymalizacji.

### 3.2.5 Skala inicjalizacji wag

W tym etapie sprawdziliśmy, jak zmiana skali (mnożnika) początkowych wag wpływa na naukę sieci. Testowane wartości: 0.01, 0.1, 1.0, 2.0, 5.0

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| 0.01 | 115418.29 | 115439.63 | -0.0008 | -0.0016 | -0.0043 |
| 0.1 | 56776.69 | 55967.42 | 0.7578 | 0.7645 | 0.7749 |
| 1.0 | 49221.07 | 50906.53 | 0.8180 | 0.8052 | 0.8084 |
| 2.0 | 46981.72 | 52262.53 | 0.8342 | 0.7947 | 0.7927 |
| 5.0 | 52829.00 | 63793.70 | 0.7903 | 0.6938 | 0.7189 |

Najgorszy wynik uzyskaliśmy dla skali 0.01. Przy tak małych wagach sieć praktycznie się nie uczy, co widać po ujemnym R², oznaczającym błąd większy niż przy zwykłym zgadywaniu średniej ceny. Z kolei przy skali 5.0 wagi były zbyt duże, co widać na bardzo wysokim błędzie na zbiorze testowym. Najlepszą wartością okazała się skala 1.0, czyli standardowa. Uzyskała ona RMSE = 50906.53. Warto zauważyć, że skala 2.0 dała lepszy wynik na treningu 0.8342, jednak gorszy na testowym i różnica była spora bo aż 4 punkty procentowe R², może to świadczyć o tym że model zaczął się przeuczać. Dlatego do końcowego modelu wybraliśmy skalę 1.0.

### 3.2.6 Współczynnik uczenia

W szóstym kroku zbadaliśmy współczynnik uczenia. Testowane wartości: 0.01, 0.005, 0.001, 0.0005, 0.0001

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| 0.01 | 44950.77 | 51748.24 | 0.8482 | 0.7987 | 0.7966 |
| 0.005 | 45918.59 | 51007.88 | 0.8415 | 0.8043 | 0.8122 |
| 0.001 | 49221.07 | 50906.53 | 0.8180 | 0.8052 | 0.8084 |
| 0.0005 | 50940.22 | 51526.85 | 0.8050 | 0.8004 | 0.8012 |
| 0.0001 | 56166.90 | 55475.32 | 0.7630 | 0.7687 | 0.7670 |

![Wykres Współczynnika Uczenia](wykres_lr.png)

Testy wykazały klasyczną zależność, zbyt wysoki współczynnik 0.01 prowadził do przeuczenia, różnica między R² na zbiorze treningowym a testowym wyniosła około 5 punktów procentowych. Natomiast zbyt niski współczynnik 0.0001 powodował niedouczenie wynik R² wyniósł 0.7687. Najlepszą wartością okazało się 0.001 z najniższym RMSE = 50906.53. Okazało się lepsze minimalnie od naszej wartości początkowej 0.005.

### 3.2.7 Liczba epok

Przedostatnim badanym parametrem była liczba epok. Testowane liczby epok: 10,30,50,100,200

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| 10 | 57275.63 | 56457.85 | 0.7535 | 0.7604 | 0.7594 |
| 30 | 53974.35 | 53802.17 | 0.7811 | 0.7825 | 0.7879 |
| 50 | 52317.66 | 52720.65 | 0.7944 | 0.7910 | 0.7853 |
| 100 | 49221.07 | 50906.53 | 0.8180 | 0.8052 | 0.8084 |
| 200 | 46208.40 | 50458.34 | 0.8396 | 0.8086 | 0.8086 |

![Wykres Epok](wykres_epoki.png)

Wraz ze wzrostem liczby epok błąd na zbiorze testowym jak również na zbiorze treningowym malał. Najlepszy wynik na zbiorze testowym osiągneliśmy dla 200 epok R² = 0.8086. Ta wartość przekroczyła narzucony przez nas próg 0.03 różnicy w R² między zbiorem testowym, a treningowym. Widać, że w porównaniu do 100 epok, wzrost na zbiorze testowym jest niewielki tylko o 0.003, a na treningowym o 0.02, są to pierwsze sygnały przeuczenia. Odrzuciliśmy więc 200 epok, wybierając 100.

### 3.2.8 Liczba powtórzeń

W ostatnim etapie przeanalizowaliśmy wpływ uśredniania wyników na ocenę jakości modelu. Testowana liczba powtórzeń: 1,3,5,10.

| Wartość parametru | Śr. RMSE (Trening) | Śr. RMSE (Test) | Śr. R² (Trening) | Śr. R² (Test) | Najlepsze R² (Test) |
|---|---|---|---|---|---|
| 1 | 48871.61 | 49672.24 | 0.8216 | 0.8084 | 0.8084 |
| 3 | 49221.07 | 50906.53 | 0.8180 | 0.8052 | 0.8084 |
| 5 | 49400.75 | 50715.63 | 0.8168 | 0.8054 | 0.8126 |
| 10 | 49438.36 | 51486.73 | 0.8162 | 0.8014 | 0.8126 |

Wynik dla zaledwie 1 powtórzenia może się wydawać najlepszy RMSE = 49672.24, ale w praktyce jest to szczęśliwe wylosowanie inicjalizacji wag lub/i podziału danych. Zwiększenie liczby powtórzeń do 3, 5 i 10 urealnia wyniki, niwelując wpływ przypadkowości. Różnice między wariantami są już znacznie mniejsze. Początkowe założenie o wykonaniu minimum 3 powtórzeń w każdym kroku było słuszne, jest to liczba, która pozwoliła zachować balans pomiędzy istotnością statystyczną, a czasem wykonywania skryptu.

### 3.3 Podsumowanie i Wnioski

Po przejściu przez wszystkie etapy optymalizacji, ostateczna konfiguracja sieci neuronowej:

- Proporcja zbioru testowego: 0.15
- Głębokość i topologia: 4 warstwy ukryte [32,32,32,32]
- Funkcja aktywacji: Tanh
- Skala inicjalizacji wag: 1.0
- Współczynnik uczenia: 0.001
- Liczba epok: 100
- Liczba powtórzeń: 3

Architektura ta pozwoliła na osiągnięcie wyniku na zbiorze testowym na poziomie R² = 0.8052 oraz błedu RMSE = 50906.53. Oznacza to, że nasz model jest w stanie wyjaśnić ponad 80 % wariancji ceny domu, co jest wynikiem satysfakcjonującym, jednak wciąż jest miejsce na poprawę. Przy zastosowaniu nowocześniejszego podejścia, użycie bibliotek takich jak TensorFlow lub PyTorch mogłoby pozwolić na znaczną poprawę wyników. Naszym małym sukcesem było to, że odrzucenie wartości parametru, który cechował się różnicą w R² między zbiorem treningowym a testowym większą niż 0.03 było słuszne i pozwoliło uniknąć przeuczenia. 

## 4. Random Forest

### 4.1 Czym jest Random Forest

Random Forest, czyli las losowy, to algorytm uczenia maszynowego oparty na wielu drzewach decyzyjnych. Zamiast polegać na jednym drzewie, tworzymy cały las drzew, z których każde uczy się na trochę innym, losowo wybranym fragmencie danych i zwykle korzysta tylko z części dostępnych cech. Działanie polega na tym, że każde drzewo podejmuje własną decyzję, a następnie wyniki wszystkich drzew są łączone. W klasyfikacji wybierana jest najczęściej wskazywana klasa, a w regresji  obliczana jest średnia z przewidywań drzew. Dzięki temu model jest stabilniejszy i mniej podatny na błędy pojedynczego drzewa. Najważniejsza idea Random Forest polega na połączeniu wielu prostszych modeli oraz wprowadzeniu losowości, co zmniejsza ryzyko przeuczenia i poprawia jakość przewidywań

### 4.2. Dobór hiperparametrów modelu Random Forest

Konfiguracje modelu Random Forest wykonanaliśmy metodą greedy. Oznacza to, że hiperparametry były dobierane kolejno, jeden po drugim. W każdym kroku zmieniano tylko jeden parametr, a pozostałe miały aktualnie najlepsze znalezione wartości. Dzięki temu można było sprawdzić, jak konkretna zmiana wpływa na jakość modelu. Do oceny jakości modelu wykorzystano 5-krotną walidację krzyżową na zbiorze treningowym. Zbiór testowy nie był używany podczas doboru parametrów, ponieważ został zostawiony do końcowej oceny modelu.

Wyniki oceniano za pomocą trzech metryk:

- R² – pokazuje, jak dobrze model wyjaśnia zmienność danych,
- MAE – średni błąd bezwzględny,
- RMSE – pierwiastek z błędu średniokwadratowego.

Najważniejszą metryką przy wyborze najlepszych parametrów było R². Im wyższa wartość R², tym lepiej model dopasowuje się do danych.

---

### 4.2.1. Strojenie parametru n_estimators

Na początku sprawdzono wpływ liczby drzew w lesie. Parametr n_estimators określa, ile drzew decyzyjnych zostanie utworzonych w modelu Random Forest. Zwykle większa liczba drzew poprawia stabilność modelu, ale jednocześnie zwiększa czas uczenia.

Testowano wartości od 50 do 500.

| n_estimators | R² | Odchylenie R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| 50  | 0.8146 | 0.0104 | 32457 | 49708 |
| 100 | 0.8169 | 0.0094 | 32255 | 49404 |
| 200 | 0.8175 | 0.0091 | 32169 | 49316 |
| 300 | 0.8175 | 0.0093 | 32175 | 49324 |
| 500 | 0.8178 | 0.0093 | 32154 | 49274 |

Najlepszy wynik uzyskano dla n_estimators = 500, gdzie R² wyniosło 0.8178. Widzimy, że zwiększanie liczby drzew poprawiało wynik, ale różnice między 200, 300 i 500 drzewami były już bardzo małe. Oznacza to, ze po pewnym momencie dodawanie kolejnych drzew nie daje już dużego zysku, a tylko wydłuża czas działania modelu.

---

### 4.2.2. Strojenie parametru max_depth

Następnie sprawdziliśmy parametr max_depth, czyli maksymalną głębokość drzew. Parametr  decyduje o tym, jak bardzo szczegółowe mogą być pojedyncze drzewa w lesie. Zbyt mała głębokość może powodować niedouczenie modelu, natomiast zbyt duża może prowadzić do przeuczenia.

| max_depth | R² | Odchylenie R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| 5    | 0.6590 | 0.0094 | 48030 | 67423 |
| 10   | 0.7849 | 0.0105 | 36379 | 53543 |
| 15   | 0.8138 | 0.0096 | 32747 | 49809 |
| 25   | 0.8178 | 0.0095 | 32159 | 49276 |
| None | 0.8178 | 0.0093 | 32154 | 49274 |

Najgorszy wynik uzyskano dla max_depth = 5, co oznacza, że drzewa były wtedy zbyt płytkie i nie były w stanie dobrze odwzorować zależności w danych. Wraz ze zwiększaniem głębokości wynik modelu wyraźnie się poprawiał.

Najlepszy wynik uzyskano dla max_depth = None, czyli bez ograniczenia głębokości drzewa. Wynik był jednak bardzo podobny do max_depth = 25, dlatego można stwierdzić, że po osiągnięciu odpowiedniej głębokości dalsze zwiększanie złożoności modelu nie daje już dużej poprawy.

---

### 4.2.3. Optymalizacja parametru min_samples_split

Kolejnym analizowanym parametrem był min_samples_split. Określa on minimalną liczbę próbek wymaganą do podziału węzła w drzewie. Im większa wartość tego parametru, tym prostsze stają się drzewa, ponieważ trudniej jest tworzyć kolejne podziały.

| min_samples_split | R² | Odchylenie R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| 2  | 0.8178 | 0.0093 | 32154 | 49274 |
| 5  | 0.8173 | 0.0094 | 32224 | 49341 |
| 10 | 0.8148 | 0.0097 | 32561 | 49679 |
| 20 | 0.8088 | 0.0098 | 33328 | 50484 |
| 50 | 0.7897 | 0.0106 | 35530 | 52938 |

Najlepsza okazała się wartość min_samples_split = 2. Przy większych wartościach wynik stopniowo się pogarszał. Oznacza to, że w tym przypadku model lepiej działał, gdy drzewa mogły wykonywać bardziej szczegółowe podziały.

Największy spadek jakości widać dla min_samples_split = 50, gdzie R² spadło do 0.7897. Taka wartość zbyt mocno ograniczała model i powodowała, że nie mógł on wystarczająco dobrze dopasować się do danych.

---

### 4.2.4. Strojenie parametru max_features

Ostatnim testowanym parametrem był max_features. Określa on, jaka część cech jest brana pod uwagę przy szukaniu najlepszego podziału w drzewie. W Random Forest często korzystne jest używanie tylko części cech, ponieważ zwiększa to różnorodność drzew i może poprawić jakość całego modelu.

| max_features | R² | Odchylenie R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| 0.3 | 0.8153 | 0.0085 | 33288 | 49612 |
| 0.5 | 0.8216 | 0.0087 | 32127 | 48757 |
| 0.7 | 0.8207 | 0.0089 | 32066 | 48885 |
| 0.9 | 0.8194 | 0.0090 | 32083 | 49060 |
| 1.0 | 0.8178 | 0.0093 | 32154 | 49274 |

Najlepszy wynik, uzyskano dla max_features = 0.5, gdzie R² wyniosło 0.8216. Oznacza to, że model osiągnął najlepsze rezultaty, gdy przy każdym podziale analizował około połowę dostępnych cech.

dla  max_features = 0.7 wynik był bardzo podobny, a MAE było nawet minimalnie niższe. Jednak ponieważ głównym kryterium wyboru było R², jako najlepszą wartość przyjęto 0.5.

---

### 4.2.5. Wizualizacja procesu strojenia

Poniższy wykres przedstawia zmianę wartości R² dla kolejnych sprawdzanych parametrów. Czerwonym punktem oznaczono najlepszą wartość w danym kroku optymalizacji.

![Greedy optymalizacja hiperparametrów](wykres_05_greedy_optymalizacja.png)

Na wykresie widać, że największa poprawa wystąpiła przy zmianie parametru max_depth. Dla bardzo małej głębokości model osiągał słabe wyniki, a po zwiększeniu głębokości R² szybko wzrosło.

W przypadku n_estimators poprawa była bardziej stopniowa. Dodawanie kolejnych drzew lekko poprawiało wynik, ale po przekroczeniu około 200 drzew różnice były już niewielkie.

Dla parametru min_samples_split najlepszy wynik był przy najmniejszej wartości, czyli 2. Zwiększanie tego parametru pogarszało wynik, ponieważ model stawał się mniej szczegółowy.

Najlepszy wynik dla max_features uzyskano przy wartości 0.5. Pokazuje to, że losowe wybieranie tylko części cech przy podziałach było korzystniejsze niż korzystanie ze wszystkich cech.

---

### 4.2.6. Końcowa konfiguracja modelu

Po zakończeniu strojenia jako najlepszy zestaw hiperparametrów wybrano:

| Parametr | Najlepsza wartość |
|---|---:|
| n_estimators | 500 |
| max_depth | None |
| min_samples_split | 2 |
| max_features | 0.5 |

Dla tej konfiguracji najlepszy średni wynik w walidacji krzyżowej wyniósł:

| Metryka | Wartość |
|---|---:|
| R² | 0.8216 |
| MAE | 32127 |
| RMSE | 48757 |

Można więc uznać, że strojenie hiperparametrów poprawiło jakość modelu. Początkowo model uzyskiwał R² na poziomie około 0.8146, a po dobraniu parametrów końcowy wynik walidacji krzyżowej wzrósł do 0.8216.

Poprawa nie była bardzo duża, ale była zauważalna. Największe znaczenie miały parametry max_depth oraz max_features. Liczba drzew również wpływała na wynik, ale po pewnym momencie jej zwiększanie dawało już tylko niewielką poprawę.

---

### 4.2.7. Ocena finalnego modelu na zbiorze testowym

Po wybraniu najlepszych parametrów model został ponownie wytrenowany na całym zbiorze treningowym. Następnie oceniono go na zbiorze testowym, który nie był używany podczas strojenia.

| Zbiór | R² | MAE | RMSE |
|---|---:|---:|---:|
| Train | 0.9765 | 11579 | 17699 |
| Test | 0.8225 | 31416 | 48415 |

![Porównanie Train vs Test](wykres_06_train_vs_test_metryki.png)

Model osiągnął bardzo wysoki wynik na zbiorze treningowym, gdzie R² wyniosło 0.9765. Na zbiorze testowym wynik był niższy i wyniósł 0.8225. Taka różnica może wskazywać na częściowe przeuczenie modelu, ponieważ model lepiej radzi sobie z danymi, na których był uczony. Mimo tego wynik testowy nadal można uznać za dobry. Model wyjaśnia około 82% zmienności cen mieszkań w danych testowych. Średni błąd bezwzględny wyniósł około 31 416 dolarów, co oznacza, że przeciętna predykcja modelu różniła się od rzeczywistej wartości właśnie o taką kwotę. RMSE było wyższe od MAE i wyniosło około 48 415 dolarów, co oznacza, że w danych występowały także większe błędy predykcji.

---



## 5. Model K-Nearest Neighbors (KNN)

### 5.1. Opis KNN

KNN, czyli metoda k najbliższych sąsiadów, to algorytm uczenia maszynowego, który klasyfikuje nowy obiekt na podstawie podobieństwa do wcześniej zapisanych danych.
Działa tak, że szuka k najbliższych obiektów i sprawdza, do jakich klas należą. Następnie wybiera tę klasę, która pojawia się najczęściej wśród sąsiadów.
Najważniejsza idea KNN jest taka, że podobne obiekty znajdują się blisko siebie. Algorytm jest łatwy do zrozumienia, ale przy dużej liczbie danych może działać wolniej.

### 5.2. Dostoswanie hiperparametrów modelu KNN

Dostoswanie modelu KNN wykonano metodą greedy, podobnie jak w przypadku modelu Random Forest. Oznacza to, że parametry były dobierane kolejno. W każdym kroku zmieniano jeden parametr, a pozostałe miały aktualnie najlepsze znalezione wartości.


---

### 5.2.1. Strojenie parametru n_neighbors

Na początku sprawdzono parametr n_neighbors, czyli liczbę najbliższych sąsiadów branych pod uwagę podczas predykcji. Jest to jeden z najważniejszych parametrów w metodzie KNN.

Mała liczba sąsiadów może powodować zbyt duże dopasowanie do pojedynczych obserwacji, natomiast zbyt duża liczba sąsiadów może nadmiernie uśredniać wyniki i osłabiać dokładność predykcji.

Testowano następujące wartości:

| n_neighbors | R² | Odchylenie R² | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 1  | 0.5706 | 0.0083 | 49489 | 75663 |
| 3  | 0.6929 | 0.0140 | 42874 | 63973 |
| 5  | 0.7149 | 0.0129 | 41423 | 61638 |
| 10 | 0.7256 | 0.0121 | 40924 | 60470 |
| 20 | 0.7205 | 0.0126 | 41758 | 61036 |
| 50 | 0.7000 | 0.0100 | 43894 | 63235 |

Najlepszy wynik uzyskano dla n_neighbors = 10, gdzie R² wyniosło 0.7256. Widać, że dla n_neighbors = 1 model działał  słabiej, ponieważ opierał predykcję tylko na jednym najbliższym sąsiedzie. Taki wynik był mało stabilny i dawał największe błędy, czyli MAE = 49 489 oraz RMSE = 75 663. Wraz ze zwiększaniem liczby sąsiadów jakość modelu poprawiała się aż do wartości 10. Potem wynik zaczął spadać. Dla n_neighbors = 50 R² wyniosło już tylko 0.7000, a błędy ponownie wzrosły. Oznacza to, że zbyt duża liczba sąsiadów powodowała zbyt mocne uśrednianie predykcji.


---

### 5.2.2. Strojenie parametru weights

Następnie sprawdzono parametr weights, który określa sposób ważenia sąsiadów.

Testowano dwa warianty:

| weights | R² | Odchylenie R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| uniform | 0.7256 | 0.0121 | 40924 | 60470 |
| distance | 0.7302 | 0.0124 | 40448 | 59959 |

Lepszy wynik uzyskano dla weights = distance. Oznacza to, że model działał lepiej, gdy bliżsi sąsiedzi mieli większy wpływ na końcową predykcję niż dalsi sąsiedzi.



---

### 5.2.3. Strojenie parametru metric

Ostatnim sprawdzanym parametrem była metryka odległości, czyli sposób obliczania podobieństwa między obserwacjami.

Testowano cztery metryki:

| metric | R² | Odchylenie R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| euclidean | 0.7302 | 0.0124 | 40448 | 59959 |
| manhattan | 0.7418 | 0.0113 | 39677 | 58657 |
| chebyshev | 0.7145 | 0.0133 | 41791 | 61677 |
| minkowski | 0.7302 | 0.0124 | 40448 | 59959 |

Najlepszy wynik uzyskano dla metryki manhattan, gdzie R² wyniosło 0.7418. Był to najlepszy wynik spośród wszystkich testowanych ustawień w walidacji krzyżowej.

Najgorzej wypadła metryka chebyshev, dla której R² spadło do 0.7145. Oznacza to, że ten sposób liczenia odległości gorzej pasował do analizowanych danych.

---

### 5.2.4. Wizualizacja procesu strojenia

Poniższy wykres przedstawia zmianę wartości R² dla kolejnych sprawdzanych parametrów modelu KNN. Czerwonym punktem oznaczono najlepszą wartość dla danego parametru.

![Greedy optymalizacja KNN](knn_wykres_05_greedy_optymalizacja.png)

Na wykresie widać, że największą różnicę zrobił parametr n_neighbors. Dla wartości 1 wynik był słaby, natomiast po zwiększeniu liczby sąsiadów jakość modelu wyraźnie wzrosła.

W przypadku parametru weights lepszy wynik uzyskano dla distance, czyli wtedy, gdy bliżsi sąsiedzi mieli większy wpływ na predykcję.

Dla metryki odległości najlepsza okazała się manhattan. Oznacza to, że sposób liczenia odległości miał zauważalny wpływ na jakość modelu.

---

### 5.2.5. Końcowa konfiguracja modelu

Po zakończeniu strojenia jako najlepszy zestaw hiperparametrów wybrano:

| Parametr | Najlepsza wartość |
|---|---:|
| n_neighbors | 10 |
| weights | distance |
| metric | manhattan |
| p | 2 |

Dla tej konfiguracji najlepszy średni wynik w walidacji krzyżowej wyniósł:

| Metryka | Wartość |
|---|---:|
| R² | 0.7418 |
| MAE | 39677 |
| RMSE | 58657 |

Można zauważyć, że strojenie hiperparametrów poprawiło jakość modelu KNN. Największe znaczenie miała liczba sąsiadów oraz wybór metryki odległości.

---

### 5.2.6. Ocena finalnego modelu na zbiorze testowym

Po wybraniu najlepszych parametrów model został wytrenowany na całym zbiorze treningowym, a następnie oceniony na zbiorze testowym.

| Zbiór | R² | MAE | RMSE |
|---|---:|---:|---:|
| Train | 1.0000 | 0 | 0 |
| Test | 0.7378 | 38946 | 58834 |

![Porównanie Train vs Test KNN](knn_wykres_06_train_vs_test_metryki.png)

Model uzyskał wynik R² = 1.0000 na zbiorze treningowym, co oznacza idealne dopasowanie do danych treningowych. Wynika to z zastosowania weights = distance, ponieważ przy predykcji na danych treningowych najbliższym sąsiadem obserwacji jest ona sama, więc błąd może być równy zero.

Na zbiorze testowym wynik był niższy i wyniósł R² = 0.7378. Oznacza to, że model wyjaśnia około 74% zmienności cen mieszkań w danych testowych.

MAE na zbiorze testowym wyniosło około 38 946 dolarów, czyli przeciętna predykcja różniła się od rzeczywistej wartości o około 39 tysięcy dolarów. RMSE wyniosło około 58 834 dolarów, co pokazuje, że występowały również większe błędy predykcji.

Różnica między wynikiem treningowym i testowym wskazuje na przeuczenie modelu. KNN bardzo dobrze zapamiętał dane treningowe, ale gorzej radził sobie na nowych danych. Mimo tego wynik testowy nadal pokazuje, że model potrafił uchwycić część zależności w danych.

## 6. Model SVM

### 6.1. Opis SVM

SVM (Support Vector Machine) to metoda uczenia maszynowego, która polega na znalezieniu funkcji najlepiej dopasowującej się do danych. W przypadku regresji (SVR) model stara się znaleźć funkcję, która mieści się w określonym marginesie błędu (`epsilon`), jednocześnie zachowując jak największą prostotę.

SVM może wykorzystywać różne funkcje jądra (kernel), które pozwalają modelowi uchwycić zależności liniowe i nieliniowe.

---

### 6.2. Strojenie hiperparametrów modelu SVM

Strojenie modelu wykonano poprzez analizę wpływu poszczególnych parametrów na jakość modelu.



---

### 6.2.1. Strojenie parametru `C`

Parametr `C` kontroluje stopień dopasowania modelu.

| `C` | R² |
|---:|---:|
| 0.1 | -0.0561 |
| 1   | -0.0512 |
| 10  | -0.0064 |
| 100 | 0.287 |

Najlepszy wynik uzyskano dla `C = 100`. Wraz ze wzrostem wartości parametru model lepiej dopasowywał się do danych co pokazuje wyższy wskaźnik R².

---

### 6.2.2. Strojenie parametru `kernel`

| `kernel` | R² |
|---|---:|
| linear | 0.6166 |
| rbf | 0.287 |
| poly | 0.177 |
| sigmoid | 0.4511 |

Najlepszy wynik uzyskano dla jądra **linear**, co sugeruje, że zależności pomiędzy zmiennymi miały głównie charakter liniowy. Pozostałe jądra osiągnęły niższe wartości R², przez co gorzej dopasowywały model do danych.

---

### 6.2.3. Strojenie parametrów `gamma` i `epsilon`

Parametr `gamma` określa wpływ pojedynczych obserwacji na model, natomiast `epsilon` odpowiada za szerokość marginesu błędu akceptowanego przez model SVR.
Parametry były testowane dla kilku różnych wartości w celu sprawdzenia ich wpływu na jakość predykcji.

Zmiany parametrów `gamma` oraz `epsilon` nie miały istotnego wpływu na jakość modelu. Dla wszystkich testowanych wartości uzyskano bardzo podobne wyniki R², MAE oraz RMSE.
Największy wpływ na skuteczność modelu miał wybór jądra (`kernel`) oraz parametru `C`.

---

### 6.2.4. Końcowa konfiguracja modelu

| Parametr | Wartość |
|---|---:|
| `C` | 100 |
| `kernel` | linear |
| `gamma` | scale |
| `epsilon` | 0.01 |

---

### 6.2.5. Ocena modelu na zbiorze testowym

Po wybraniu najlepszych parametrów model został wytrenowany na zbiorze treningowym, a następnie oceniony na zbiorze testowym.

| Zbiór | R² | MAE | RMSE |
|---|---:|---:|---:|
| Train | 0.62 | 49031 | 71146 |
| Test | 0.60 | 49198 | 72474 |

Model osiągnął umiarkowaną jakość dopasowania. Wartość R² na poziomie około **0.60** oznacza, że model wyjaśnia około 60% zmienności danych.

Wartości MAE i RMSE pokazują, że błędy predykcji są nadal dość wysokie. Niewielka różnica pomiędzy wynikami dla zbioru treningowego i testowego sugeruje jednak, że model nie jest mocno przeuczony i dość dobrze radzi sobie z nowymi danymi.

---

## 7. Model Gradient Boosting

### 7.1. Opis Gradient Boosting

Gradient Boosting to metoda oparta na wielu drzewach decyzyjnych budowanych sekwencyjnie. Każde kolejne drzewo poprawia błędy poprzednich, co pozwala modelowi uchwycić złożone zależności w danych.

---

### 7.2. Strojenie hiperparametrów modelu Gradient Boosting

Przeprowadzono strojenie hiperparametrów modelu w celu znalezienia konfiguracji dającej najlepsze wyniki.

---

### 7.2.1. Strojenie parametru `n_estimators`

| `n_estimators` | R² | MAE | RMSE |
|---:|---:|---:|---:|
| 50  | 0.7249 | 43104 | 60566 |
| 100 | 0.7744 | 38297 | 54845 |
| 200 | 0.8020 | 35342 | 51379 |
| 300 | 0.8117 | 34256 | 50108 |

Wraz ze wzrostem liczby estymatorów poprawiała się jakość modelu. Rosły wartości R², a błędy MAE i RMSE malały.
Najlepszy wynik uzyskano dla `n_estimators = 300`.

---

### 7.2.2. Strojenie parametru `learning_rate`

Parametr określa, jak duże zmiany model wprowadza podczas kolejnych etapów uczenia.

| `learning_rate` | R² | MAE | RMSE |
|---:|---:|---:|---:|
| 0.01 | 0.6702 | 48240 | 66311 |
| 0.05 | 0.7924 | 36353 | 52614 |
| 0.1  | 0.8117 | 34256 | 50108 |
| 0.2  | 0.8202 | 33164 | 48952 |

Najlepszy wynik uzyskano dla `learning_rate = 0.2`.

---

### 7.2.3. Strojenie parametru `max_depth`

| `max_depth` | R² | MAE | RMSE |
|---:|---:|---:|---:|
| 2 | 0.7974 | 35946 | 51969 |
| 3 | 0.8202 | 33164 | 48952 |
| 4 | 0.8277 | 32032 | 47929 |
| 5 | 0.8285 | 31670 | 47816 |

Parametr `max_depth` kontroluje poziom złożoności drzew w modelu.
Większe wartości poprawiały wyniki modelu. Najlepszy rezultat uzyskano dla wartości 5.

---

### 7.2.4. Strojenie parametru `subsample`

| `subsample` | R² | MAE | RMSE |
|---:|---:|---:|---:|
| 0.6 | 0.8209 | 32511 | 48867 |
| 0.8 | 0.8269 | 31952 | 48033 |
| 0.9 | 0.8280 | 31868 | 47879 |
| 1.0 | 0.8285 | 31670 | 47816 |

Parametr `subsample` określa, jaka część danych jest wykorzystywana podczas uczenia kolejnych drzew.
Wraz ze wzrostem wartości, model osiągał nieco lepsze wyniki. Najlepszy rezultat uzyskano dla `subsample = 1.0`.

---

### 7.2.5. Końcowa konfiguracja modelu

| Parametr | Wartość |
|---|---:|
| `n_estimators` | 300 |
| `learning_rate` | 0.2 |
| `max_depth` | 5 |
| `subsample` | 1.0 |

---

### 7.2.6. Ocena modelu na zbiorze testowym

| Zbiór | R² | MAE | RMSE |
|---|---:|---:|---:|
| Train | 0.9450 | 19494 | 27072 |
| Test | 0.8356 | 30623 | 46587 |

Model osiągnął bardzo wysoką jakość dopasowania. Na zbiorze testowym wyjaśnia około **84% zmienności cen nieruchomości**, co oznacza bardzo dobrą skuteczność.
Widać różnicę między train a test, co może wskazywać na lekkie przeuczenie, jednak wynik testowy nadal pozostaje bardzo dobry.

---

## 8. Porównanie modeli

W projekcie porównano pięć modeli:

- MLP
- Random Forest
- KNN
- Gradient Boosting
- SVM

Porównanie wykonano na podstawie wyników uzyskanych na zbiorze testowym.

### 8.1. Porównanie wartości R²

| Model | R² (Test) |
|---|---:|
| Gradient Boosting | 0.8356 |
| Random Forest | 0.8225 |
| MLP | 0.8052 |
| KNN | 0.7378 |
| SVM | 0.6022 |

Najlepszy wynik uzyskał model **Gradient Boosting**, który wyjaśnia około **84% zmienności danych**. Bardzo zbliżony wynik osiągnął **Random Forest**.

Model **MLP** także uzyskał dobre wyniki i osiągnął wartość R² powyżej 0.80, jednak był nieco słabszy od modeli drzewiastych.

Model **KNN** osiągnął umiarkowaną jakość, natomiast **SVM** uzyskał najsłabszy wynik spośród analizowanych modeli.

---

### 8.2. Porównanie błędów predykcji

| Model | RMSE | MAE |
|---|---:|---:|
| Gradient Boosting | 46587 | 30623 |
| Random Forest | 48415 | 31416 |
| MLP | 50907 | — |
| KNN | 58834 | 38946 |
| SVM | 72474 | 49198 |

Najmniejsze błędy uzyskał model **Gradient Boosting**, co potwierdza jego wysoką skuteczność.

Największe błędy wystąpiły w modelu **SVM**, co wskazuje na jego ograniczoną zdolność do dokładnego przewidywania cen nieruchomości.

---

### 8.3. Wnioski z porównania

Na podstawie przeprowadzonej analizy można stwierdzić, że:

- **Gradient Boosting** był najlepszym modelem – osiągnął najwyższe R² oraz najniższe błędy
- **Random Forest** uzyskał bardzo zbliżone wyniki i również dobrze radził sobie z danymi
- **MLP** osiągnął dobry wynik, ale był nieco słabszy od modeli drzewiastych
- **KNN** dawał umiarkowane rezultaty, ale był podatny na przeuczenie
- **SVM** osiągnął najniższe wartości R² oraz największe błędy predykcji
- modele drzewiaste (Gradient Boosting, Random Forest) najlepiej radziły sobie z analizowanym problemem

Można więc uznać, że dla tego typu danych najbardziej odpowiednie są modele oparte na drzewach decyzyjnych, które potrafią uchwycić nieliniowe zależności między zmiennymi.
