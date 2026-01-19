import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt,  find_peaks
from ga_approx import ga_approximate
from trygonometryczna import test_trigonometric_approximation, aproksymacja_trygonometryczna
from przygotowywanie_danych import wyznacz_istotne_minima_maksima, podziel_na_fragmenty, filtr_pasmowoprzepustowy, oddechy_na_minute, rysuj_dane_szum_bez_szum, podziel_na_przedzialy, skleij_fragmenty
from analiza_etykietowanych_danych import analiza_danych_oddechowych
from scipy.signal import find_peaks
import numpy as np
from ga_oddech import run_ga_oddech
from ga_oddech import generuj_oddech





def main():
    # 1. Wczytanie i filtracja
    acc_third_subject = pd.read_csv("acc_normal.csv")
    acc_x = acc_third_subject.iloc[:, 0].values
    acc_y = acc_third_subject.iloc[:, 1].values
    # Ucięcie końcówki
    acc_x = acc_x[:-int(0.03 * len(acc_x))]
    acc_y = acc_y[:-int(0.03 * len(acc_y))]

    fs = len(acc_x) / (acc_x[-1] - acc_x[0])
    print(f"Sampling frequency: {fs:.2f} Hz")

    filtered_signal = filtr_pasmowoprzepustowy(acc_y, fs)
    filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))
    
    
    #oddechy_na_minute(filtered_signal, fs)
    rysuj_dane_szum_bez_szum(acc_x, acc_y, filtered_signal)
    prog, czas_oddechu, fs = analiza_danych_oddechowych("acc_normal_labelled.csv")
    print("Minimalna wartość oddechu (do progowania):", prog)
    print("Średni czas między 5 ostatnimi oddechami:", czas_oddechu)
    print("Częstotliwość próbkowania:", fs)

    minima_idx,minima_filtrowane, maksima_idx = wyznacz_istotne_minima_maksima(
    filtered_signal,
    prog_min=prog,
    prog_max=prog,
    min_odstep=50,
    rysuj=True
)
    liczba_oddechow_na_fragment=5
    M=liczba_oddechow_na_fragment
    krok=liczba_oddechow_na_fragment
    # WARIANT B
    #minima_idx = minima_filtrowane

    przedzialy = podziel_na_fragmenty(
    czas=acc_x,
    sygnal=filtered_signal,
    minima_idx=minima_idx,
    maksima_idx=maksima_idx,
    liczba_oddechow_na_fragment=liczba_oddechow_na_fragment  
)
    lista_wynikow = []

    # 3. GA aproksymacja trygonometryczna
    for i, (xf, yf, minima_local, maksima_local) in enumerate(przedzialy):
        if len(yf) == 0:
            i += 1
            continue

        print(f"\n--- Fragment {i + 1}: od {xf[0]:.2f}s do {xf[-1]:.2f}s, próbek: {len(yf)} ---")

        M = min(5, len(minima_local) - 1)  # liczba harmonicznych dostosowana do długości fragmentu

        a, b, fitness = ga_approximate(
        y_fragment=yf,
        M=M,
        minima_idx=minima_local,
        maksima_idx=maksima_local,
        pop_size=100,
        num_generations=200,
        elite_size=2,
        pcross=0.8,
        pmut=0.05,
        sigma_factor=0.05,
        postep=True
    )

        N = len(yf)
        x_idx = np.arange(N)
        y_ga = aproksymacja_trygonometryczna(x_idx, N, M, a, b)
        lista_wynikow.append((xf, y_ga))
    # 4. Sklejone wyniki GA
    if lista_wynikow:
        x_sklejone, y_sklejone = skleij_fragmenty(lista_wynikow)
        plt.figure(figsize=(10, 4))
        plt.plot(acc_x, filtered_signal, "b--", label="Przefiltrowany pełny sygnał", alpha=0.4)
        plt.plot(x_sklejone, y_sklejone, "r-", label="GA: sklejenie fragmentów", linewidth=1.0)
        plt.title("Cały sygnał: filtrowany vs. GA: sklejenie fragmentów")
        plt.xlabel("Czas (s)")
        plt.ylabel("Wartość sygnału")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()

