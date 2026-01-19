
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from przygotowywanie_danych import filtr_pasmowoprzepustowy

def analiza_danych_oddechowych(plik_csv, dolna=0.1, gorna=0.5, rzad=3, ostatnie_oddechy=5):
    # Wczytanie danych
    df = pd.read_csv(plik_csv, header=None)
    df = df[df.iloc[:, 1] != 999.0]
    sygnal = df.iloc[:, 0].values
    etykieta = df.iloc[:, 1].values
    czas = df.iloc[:, 2].values

    # Częstotliwość próbkowania
    fs = len(czas) / (czas[-1] - czas[0])
    sygnal_filtrowany = filtr_pasmowoprzepustowy(sygnal, fs, dolna, gorna, rzad)

    # Normalizacja do [-1, 1]
    max_abs = np.max(np.abs(sygnal_filtrowany))
    if max_abs > 0:
        sygnal_filtrowany = sygnal_filtrowany / max_abs

    # Szukanie segmentów wdechu (etykieta == 1.0)
    maski_segmentow = []
    aktualna_maska = []

    for i in range(len(etykieta)):
        if etykieta[i] == 1.0:
            aktualna_maska.append(i)
        elif aktualna_maska:
            maski_segmentow.append(aktualna_maska)
            aktualna_maska = []
    if aktualna_maska:
        maski_segmentow.append(aktualna_maska)

    wartosci_szczytow = []
    czasy_szczytow = []

    for segment in maski_segmentow:
        indeksy = np.array(segment)
        fragment = sygnal_filtrowany[indeksy]
        max_idx_local = np.argmax(fragment)
        wartosc_max = fragment[max_idx_local]
        czas_max = czas[indeksy[max_idx_local]]
        wartosci_szczytow.append(wartosc_max)
        czasy_szczytow.append(czas_max)

    wartosci_szczytow = np.array(wartosci_szczytow)
    czasy_szczytow = np.array(czasy_szczytow)

    # Odrzucenie 5% najmniejszych wartości
    if len(wartosci_szczytow) >= 10:
        min_wartosc_oddechu = np.percentile(wartosci_szczytow, 2)
    else:
        min_wartosc_oddechu = np.min(wartosci_szczytow)

    # Średni czas między ostatnimi X oddechami
    if len(czasy_szczytow) >= ostatnie_oddechy + 1:
        roznice = np.diff(czasy_szczytow[-(ostatnie_oddechy+1):])
        sredni_czas_oddechu = np.mean(roznice)
    else:
        sredni_czas_oddechu = np.nan

    # Wykres pomocniczy
    plt.figure(figsize=(10, 4))
    plt.plot(czas, sygnal_filtrowany, label="Sygnał filtrowany (znormalizowany)")
    plt.plot(czasy_szczytow, wartosci_szczytow, "ro", label="Szczyty wdechów")
    plt.axhline(min_wartosc_oddechu, color='gray', linestyle='--', label='Próg 5%')
    plt.xlabel("Czas (s)")
    plt.ylabel("Amplituda")
    plt.title("Punkty szczytowe fazy wdechów")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return min_wartosc_oddechu, sredni_czas_oddechu, fs
