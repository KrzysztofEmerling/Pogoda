# Projekt Prognozowania Pogody z Wykorzystaniem Sieci Rekurencyjnych i Transformerów

## Opis projektu

Celem projektu jest porównanie dwóch typów modeli sieci neuronowych — sieci rekurencyjnej (RNN) oraz modelu transformera — w kontekście prognozowania pogody. Oba modele zostały zaimplementowane i przetestowane w Jupyter Notebook.

## Zawartość projektu

- **Jupyter Notebook**: Główny plik projektu, zawierający wszystkie kroki analizy, od przygotowania danych po trenowanie modeli i ich ewaluację.
- **Wyniki**: Porównanie wyników obu modeli oraz wizualizacja prognozowanych danych pogodowych.

## Modele

### Sieć Rekurencyjna (RNN)

- Klasyczna architektura do analizy sekwencji danych, zaprezentowana na zajęciach jako przykład do prognozowania szeregów czasowych, wykorzystana jako punkt odniesienia.
- Dobrze sprawdza się w prognozowaniu szeregów czasowych, takich jak dane pogodowe.

### Transformer

- Nowoczesna architektura, która wykorzystuje mechanizm uwagi do przetwarzania sekwencji.
- Wymaga zastosowania kodowania pozycyjnego, co pozwala modelowi na rozumienie kolejności przetwarzanych danych.

## Kluczowe Wnioski

- Najważniejsza zmiana w projekcie okazała się modyfikacja funkcji straty, by uwzględniała odległość prognozy, co pozwoliło uzyskać znacznie lepsze wyniki.
- Oba modele są skuteczne w prognozowaniu pogody, jednak transformer wykazuje lepsze wyniki po odpowiednich modyfikacjach. Po tuningu wykazuje znacznie mniejsze przeuczenie niż RNN.
- Kodowanie pozycyjne jest kluczowe dla efektywności modelu transformera, ponieważ umożliwia mu zrozumienie kontekstu sekwencyjnego.

## Metoda Prezentacji Danych

W projekcie zaimplementowano prostą metodę wizualizacji przewidywanych danych pogodowych, która ułatwia interpretację wyników i porównanie prognoz z rzeczywistymi danymi.

## Wnioski Osobiste

Podczas pracy nad projektem zdobyłem cenną wiedzę na temat modeli uczenia maszynowego, szczególnie w zakresie architektur transformera. Zrozumienie ich działania oraz zastosowanie w praktyce znacznie poszerzyło moje umiejętności w dziedzinie analizy danych i prognozowania.

## Jak uruchomić projekt

1. **Zainstaluj wymagane biblioteki**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Uruchom Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Otwórz plik `Pogoda-Transformer.ipynb`** i wykonaj wszystkie komórki.
