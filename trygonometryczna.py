import pandas
import matplotlib.pyplot as plt
import numpy as np


def aproksymacja_trygonometryczna(x_fine, N, M, a, b):
    ya = np.full(len(x_fine), a[0])
    M = min(M, len(a) - 1)
    for n in range(1, M + 1):
        #zgodnie z wzorem
        ya += a[n] * np.cos(2 * np.pi * n * x_fine / N) + b[n] * np.sin(2 * np.pi * n * x_fine / N)
    return ya

def oblicz_a_b(y):
    N = len(y)
    Y = np.fft.fft(y)
    #k_max to maksymalny numer użytecznej harmonicznej
    k_max = N // 2
    rozmiar = k_max + 1
    a = np.zeros(rozmiar)
    b = np.zeros(rozmiar + 1)
    a[0] = Y[0].real / N
    #b[0] = 0, wiec nie trzeba ustawiac
    for n in range(1, rozmiar):
        if N % 2 == 0 and n == N // 2:
            #jesli N jest parzyste i jest to skladnik w polowie FFT to nie ma czesci urojonej
            a[n] = Y[n].real / N
            b[n] = 0
        else:
            #razy 2 bo, symetrycznie w zespolonej dziedzinie /N bo fft nie normalizuje
            a[n] = 2 * Y[n].real / N
            b[n] = -2 * Y[n].imag / N
    return a, b

def test_trigonometric_approximation(y, M, max_harmonics=50):
    N = len(y)
    x = np.arange(N)
    Mx = 4
    x_fine = np.linspace(0, N - 1, Mx * (N - 1) + 1)
    a, b = oblicz_a_b(y)
    ya = aproksymacja_trygonometryczna(x_fine, N, M, a, b)
    rmse_wartosci = []
    for k in range(1, min(max_harmonics + 1, len(a))):
        ya_k = aproksymacja_trygonometryczna(x, N, k, a, b)
        rmse = np.sqrt(np.mean((y - ya_k) ** 2))
        rmse_wartosci.append(rmse)
        if k == M:
            y_M = ya_k.copy()
            rmse_M = rmse

    # Wykres
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(rmse_wartosci) + 1), rmse_wartosci, 'bo-')
    plt.grid()
    plt.title("RMSE w zależności od liczby harmonicznych")
    plt.xlabel("Liczba harmonicznych (M)")
    plt.ylabel("RMSE")
    plt.subplot(2, 1, 2)
    plt.plot(x, y, 'b--', label="Dane filtrowane", linewidth=1.5)
    plt.plot(x_fine, ya, 'r-', label=f"Aproksymacja (M={M})",linewidth=1.5)
    plt.grid()
    plt.xlabel("t - próbka")
    plt.ylabel("Wartość sygnału")
    plt.legend()
    plt.tight_layout()
    #plt.show()

    return rmse_wartosci,ya ,y_M, rmse_M,
def compute_fft_rmse(y_fragment, M):
    N = len(y_fragment)
    if N == 0:
        return np.array([]), np.inf

    # 1. Oblicz współczynniki a, b przez FFT:
    a_full, b_full = oblicz_a_b(y_fragment)

    # 2. Składamy wektor aproksymacji na siatce [0..N-1]:
    x_idx = np.arange(N)
    # ograniczamy M do długości a_full-1
    M_eff = min(M, len(a_full) - 1)
    a = a_full[: M_eff + 1]            # a[0]..a[M_eff]
    # b_full zawiera b[0], b[1].. b[k_max]; my potrzebujemy b[1..M_eff]
    b = np.concatenate([[0], b_full[1 : M_eff + 1]])  

    # 3. Generujemy y_fft:
    y_fft = np.full(N, a[0], dtype=float)
    for k in range(1, M_eff + 1):
        y_fft += a[k] * np.cos(2 * np.pi * k * x_idx / N) \
               + b[k] * np.sin(2 * np.pi * k * x_idx / N)

    # 4. RMSE:
    diff = y_fragment - y_fft
    rmse_fft = np.sqrt(np.mean(diff * diff))

    return y_fft, rmse_fft