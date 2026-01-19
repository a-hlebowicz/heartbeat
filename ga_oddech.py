import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# === MODEL FUNKCJI ODDECHOWEJ ===

def generuj_oddech(x, amplitudy, szerokosci, pozycje):
    y = np.zeros_like(x, dtype=float)
    for Ai, wi, xi in zip(amplitudy, szerokosci, pozycje):
        y += Ai * np.exp(-((x - xi) ** 2) / (2 * wi ** 2))
    return y

# === FITNESS ===

def fitness_gaussowy(y_true, chromosom, pozycje):
    n = len(pozycje)
    amplitudy = chromosom[:n]
    szerokosci = chromosom[n:]

    x = np.arange(len(y_true))
    y_hat = generuj_oddech(x, amplitudy, szerokosci, pozycje)

    # 1. Zgodność w pozycjach szczytów
    err_szcyty = np.sum((y_true[pozycje] - y_hat[pozycje]) ** 2)

    # 2. Gładkość całego sygnału (druga pochodna)
    d2 = np.diff(y_hat, n=2)
    roughness = np.mean(np.abs(d2))

    # 3. Szerokości: penalizacja za zbyt małe lub zbyt duże wi
    penalty_szerokosci = np.mean(np.clip(np.abs(szerokosci - 25) / 25, 0, 1))

    fitness = 1.0 / (1.0 + err_szcyty + 2 * roughness + 2 * penalty_szerokosci)
    return fitness

# === GA ===

def tworzenie_populacji(n, A_bounds, w_bounds, pop_size):
    populacja = []
    for _ in range(pop_size):
        Ai = np.random.uniform(*A_bounds, size=n)
        wi = np.random.uniform(*w_bounds, size=n)
        chrom = np.concatenate([Ai, wi])
        populacja.append(chrom)
    return populacja

def selekcja_turniejowa(populacja, fitnessy, n_rodzicow):
    rodzice = []
    while len(rodzice) < n_rodzicow:
        i1, i2 = np.random.choice(len(populacja), 2, replace=False)
        wygrany = populacja[i1] if fitnessy[i1] >= fitnessy[i2] else populacja[i2]
        rodzice.append(wygrany.copy())
    return rodzice

def crossover(a, b, pcross):
    if np.random.rand() < pcross:
        alpha = np.random.rand()
        return alpha * a + (1 - alpha) * b
    else:
        return a.copy()

def mutacja(chrom, pmut, sigma, A_bounds, w_bounds):
    n = len(chrom) // 2
    for i in range(len(chrom)):
        if np.random.rand() < pmut:
            chrom[i] += np.random.normal(0, sigma)
            if i < n:
                chrom[i] = np.clip(chrom[i], *A_bounds)
            else:
                chrom[i] = np.clip(chrom[i], *w_bounds)
    return chrom

def run_ga_oddech(
    y_fragment,
    pozycje,
    pop_size=100,
    generations=100,
    elite_size=2,
    pcross=0.8,
    pmut=0.05,
    sigma=0.1,
    A_bounds=(0.1, 1.5),
    w_bounds=(5.0, 60.0),
    postep=False
):
    n = len(pozycje)
    populacja = tworzenie_populacji(n, A_bounds, w_bounds, pop_size)
    fitnessy = [fitness_gaussowy(y_fragment, chrom, pozycje) for chrom in populacja]
    best_fitness_per_gen = []

    for gen in range(generations):
        idx = np.argsort(fitnessy)[::-1]
        populacja = [populacja[i] for i in idx]
        fitnessy = [fitnessy[i] for i in idx]

        best_fitness_per_gen.append(fitnessy[0])

        elite = populacja[:elite_size]
        rodzice = selekcja_turniejowa(populacja, fitnessy, pop_size - elite_size)

        dzieci = []
        while len(dzieci) < (pop_size - elite_size):
            i1, i2 = np.random.choice(len(rodzice), 2, replace=False)
            dziecko = crossover(rodzice[i1], rodzice[i2], pcross)
            dziecko = mutacja(dziecko, pmut, sigma, A_bounds, w_bounds)
            dzieci.append(dziecko)

        populacja = elite + dzieci
        fitnessy = [fitness_gaussowy(y_fragment, chrom, pozycje) for chrom in populacja]

        if postep and (gen % 10 == 0 or gen == generations - 1):
            print(f"Gen {gen+1} — Best fitness: {fitnessy[0]:.6f}")

    best = populacja[0]
    best_ampl = best[:n]
    best_width = best[n:]
    x = np.arange(len(y_fragment))
    y_hat = generuj_oddech(x, best_ampl, best_width, pozycje)
    
    # Wykres
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(best_fitness_per_gen, 'g.-')
    plt.xlabel("Pokolenie")
    plt.ylabel("Fitness")
    plt.title("GA: postęp")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(x, y_fragment, 'b--', label="Sygnał wejściowy")
    plt.plot(x, y_hat, 'r-', label="GA (suma Gaussów)")
    plt.plot(pozycje, y_fragment[pozycje], 'ko', label="Pozycje szczytów")
    plt.xlabel("t - próbka")
    plt.ylabel("Wartość")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return best_ampl, best_width, best_fitness_per_gen[-1]
