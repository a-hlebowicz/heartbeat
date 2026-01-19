import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt,  find_peaks

def filtr_pasmowoprzepustowy(signal, fs, dolna_czest_odciecia=0.1, gorna_czest_odciecia=0.5, rzad_filtra=3):
    nyquist = 0.5 * fs
    czestotliwosc_dolna = dolna_czest_odciecia / nyquist
    czestotliwosc_gorna = gorna_czest_odciecia / nyquist
    b, a = butter(rzad_filtra, [czestotliwosc_dolna, czestotliwosc_gorna], btype='band')
    #print(type(butter(3, [0.1, 0.5], btype='band')))
    przefiltrowane = filtfilt(b, a, signal)
    return przefiltrowane

def oddechy_na_minute(signal, fs):
    szczyty_idx, _ = find_peaks(signal, distance=fs*2)

    wektor_czasu = np.arange(len(signal)) / fs
    wystepowanie_szczytow = wektor_czasu[szczyty_idx]

    czas_trwania_sygnalu = len(signal) / fs
    czas_przesuwania_okna = 60
    step_size = 10

    czestosc_oddechow = []
    time_stamps = []

    for start in np.arange(0, czas_trwania_sygnalu - czas_przesuwania_okna, step_size):
        end = start + czas_przesuwania_okna
        count = np.sum((wystepowanie_szczytow >= start) & (wystepowanie_szczytow < end))
        breath_rate = count
        czestosc_oddechow.append(breath_rate)
        time_stamps.append(start + czas_przesuwania_okna / 2)

    # Wykres
    plt.figure(figsize=(10, 4))
    plt.plot(time_stamps, czestosc_oddechow, marker='o')
    plt.xlabel("Czas (s)")
    plt.ylabel("Oddechy na minutę")
    plt.title("Liczba oddechów na minutę w czasie")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def rysuj_dane_szum_bez_szum(acc_third_subject_x, acc_third_subject_y, filtered_signal):
    plt.figure(figsize=(12, 6))
    plt.plot(acc_third_subject_x, acc_third_subject_y, color='blue', label='Dane oryginalne')
    plt.plot(acc_third_subject_x, filtered_signal, color='green', label='Dane filtrowane')
    plt.xlabel('x')
    plt.ylabel('Wartość sygnału')
    plt.title("Dane oryginalne i filtrowane")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def podziel_na_przedzialy(x, y, dlugosc_przedzialu):
    czas_start = x[0]
    czas_end = x[-1]
    liczba_przedzialow = int(np.floor((czas_end - czas_start) / dlugosc_przedzialu))
    lista_przedzialow = []

    for i in range(liczba_przedzialow):
        start_t = czas_start + i * dlugosc_przedzialu
        stop_t = start_t + dlugosc_przedzialu
        maska = (x >= start_t) & (x < stop_t)
        x_fragment = x[maska]
        y_fragment = y[maska]
        lista_przedzialow.append((x_fragment, y_fragment))

    return lista_przedzialow

def skleij_fragmenty(lista_fragmentow):
   
    x_sklejone = []
    y_sklejone = []

    for i, (x_frag, y_frag) in enumerate(lista_fragmentow):
        if i == 0:
            x_sklejone.extend(x_frag)
            y_sklejone.extend(y_frag)
        else:
            # porównaj ostatnią próbkę z pierwszą nowego fragmentu
            if x_sklejone[-1] == x_frag[0]:
                x_frag = x_frag[1:]
                y_frag = y_frag[1:]
            x_sklejone.extend(x_frag)
            y_sklejone.extend(y_frag)

    return np.array(x_sklejone), np.array(y_sklejone)

def podziel_na_fragmenty(czas, sygnal, minima_idx, maksima_idx, liczba_oddechow_na_fragment=1):
    fragmenty = []
    i = 0
    while i + liczba_oddechow_na_fragment < len(minima_idx):
        start = minima_idx[i]
        end = minima_idx[i + liczba_oddechow_na_fragment]

        x_fragment = czas[start:end + 1]
        y_fragment = sygnal[start:end + 1]

        minima_local = [idx - start for idx in minima_idx if start <= idx <= end]
        maksima_local = [idx - start for idx in maksima_idx if start <= idx <= end]

        fragmenty.append((x_fragment, y_fragment, minima_local, maksima_local))
        i += liczba_oddechow_na_fragment

    #ostatni fragment, jeśli zostały jeszcze ≥2 minima
    if i + 1 < len(minima_idx):
        start = minima_idx[i]
        end = minima_idx[-1]

        x_fragment = czas[start:end + 1]
        y_fragment = sygnal[start:end + 1]

        minima_local = [idx - start for idx in minima_idx if start <= idx <= end]
        maksima_local = [idx - start for idx in maksima_idx if start <= idx <= end]

        fragmenty.append((x_fragment, y_fragment, minima_local, maksima_local))

    return fragmenty

def wyznacz_istotne_minima_maksima(y, prog_min=0.2, prog_max=0.2, min_odstep=30, rysuj=True):
    minima_wszystkie, _ = find_peaks(-y, height=-prog_min, distance=min_odstep)
    maksima, _ = find_peaks(y, height=prog_max, distance=min_odstep)

    minima_wszystkie = minima_wszystkie.tolist()
    istotne_maksima = []

    #filtrowanie minimów zbyt płytkich lub leżących na płaskim fragmencie
    minima_filtrowane = []
    for i in range(len(minima_wszystkie)):
        idx = minima_wszystkie[i]
        okno = y[max(0, idx - 10): idx + 10]
        lokalny_srodek = y[idx]
        lokalna_srednia = np.mean(okno)
        warunek_plytkosci = lokalny_srodek > lokalna_srednia - 0.02  #nie wystaje w dół
        if not warunek_plytkosci:
            minima_filtrowane.append(idx)

    #maksima tylko pomiędzy wszystkimi minimami
    for i in range(len(minima_wszystkie) - 1):
        m1 = minima_wszystkie[i]
        m2 = minima_wszystkie[i + 1]
        lokalne_maksima = [k for k in maksima if m1 < k < m2]
        if lokalne_maksima:
            best = max(lokalne_maksima, key=lambda k: y[k])
            istotne_maksima.append(best)

    if rysuj:
        x = np.arange(len(y))
        plt.figure(figsize=(10, 4))
        plt.plot(x, y, label="Sygnał filtrowany")
        plt.plot(minima_wszystkie, y[minima_wszystkie], "go", label="Minima wszystkie (A)")
        plt.plot(minima_filtrowane, y[minima_filtrowane], "gx", label="Minima filtrowane (B)")
        plt.plot(istotne_maksima, y[istotne_maksima], "ro", label="Maksima")
        plt.xlabel("t - próbka")
        plt.ylabel("Wartość")
        plt.title("Porównanie wariantów minimów")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return minima_wszystkie, minima_filtrowane, istotne_maksima