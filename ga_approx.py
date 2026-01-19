# ga_approx.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def fitness_extrema_sinusoidalny(y, chromosom, M, minima_idx, maksima_idx):
    N = len(y)
    a = chromosom[:M+1]
    b = np.concatenate([[0], chromosom[M+1:]])

    x = np.arange(N)
    y_hat = np.full(N, a[0], dtype=float)
    for k in range(1, M + 1):
        y_hat += a[k] * np.cos(2 * np.pi * k * x / N) + b[k] * np.sin(2 * np.pi * k * x / N)

    # 1. Trafienie w wartości ekstremów
    idx = np.array(minima_idx + maksima_idx)
    err_extrema = np.sqrt(np.mean((y[idx] - y_hat[idx]) ** 2))

    # 2. Gładkość poza ekstremami
    d2 = np.diff(y_hat, n=2)
    maska = np.ones(len(d2), dtype=bool)
    for i in idx:
        maska[max(0, i - 5):min(len(d2), i + 5)] = False
    roughness = np.mean(np.abs(d2[maska])) if np.any(maska) else np.inf

    fitness = 1.0 / (1.0 + 4 * err_extrema + 3 * roughness)
    return fitness

def tworzenie_populacji(pop_size, num_genes, A):
    return [np.random.uniform(-A, A, size=num_genes) for _ in range(pop_size)]

def wybor_rodzicow(population, fitness_list, num_parents, algorytm):
    if algorytm == "turniejowa":
        return selekcja_turniejowa(population, fitness_list, num_parents)

def selekcja_turniejowa(population, fitness_list, num_parents):
    pop_size = len(population)
    parents = []
    while len(parents) < num_parents:
        i1, i2 = np.random.choice(pop_size, size=2, replace=False)
        chosen = population[i1] if fitness_list[i1] >= fitness_list[i2] else population[i2]
        parents.append(chosen.copy())
    return parents

def crossover(parentA, parentB, pcross):
    if np.random.rand() < pcross:
        alpha = np.random.rand()
        child = alpha * parentA + (1.0 - alpha) * parentB
    else:
        child = parentA.copy()
    return child

def gaussian_mutacja(child, pmut, sigma, A):
    for i in range(len(child)):
        if np.random.rand() < pmut:
            child[i] += np.random.normal(0, sigma)
            child[i] = np.clip(child[i], -A, A)
    return child

def run_ga(y_fragment, M, minima_idx, maksima_idx,
    pop_size,
    num_generations,
    elite_size,
    pcross,
    pmut,
    sigma_factor,
    postep,
):
    A = np.max(np.abs(y_fragment)) if len(y_fragment) > 0 else 1.0
    num_genes = 2 * M + 1
    population = tworzenie_populacji(pop_size, num_genes, A)
    fitness_list = [fitness_extrema_sinusoidalny(y_fragment, indiv, M, minima_idx, maksima_idx) for indiv in population]

    best_fitness_per_gen = []

    for gen in range(num_generations):
        idx_sorted = np.argsort(fitness_list)[::-1]
        population = [population[i] for i in idx_sorted]
        fitness_list = [fitness_list[i] for i in idx_sorted]

        current_best_fitness = fitness_list[0]
        best_fitness_per_gen.append(current_best_fitness)

        elite = population[:elite_size]
        num_parents = pop_size - elite_size
        parents = wybor_rodzicow(population, fitness_list, num_parents, "turniejowa")

        children = []
        sigma = sigma_factor * A
        while len(children) < num_parents:
            iA, iB = np.random.choice(len(parents), size=2, replace=False)
            parentA = parents[iA]
            parentB = parents[iB]
            child = crossover(parentA, parentB, pcross)
            child = gaussian_mutacja(child, pmut, sigma, A)
            children.append(child)

        population = elite + children
        fitness_list = [fitness_extrema_sinusoidalny(y_fragment, indiv, M, minima_idx, maksima_idx) for indiv in population]

        if postep and ((gen + 1) % 10 == 0 or gen == 0):
            print(f"Gen {gen+1}/{num_generations} — Best fitness: {current_best_fitness:.6f}")

    best_chromosome = population[0]
    return best_chromosome, best_fitness_per_gen

def rysuj_ga_results(y_fragment, best_chromosome, best_fitness_per_gen, M):
    N = len(y_fragment)
    x = np.arange(N)
    Mx = 4
    x_fine = np.linspace(0, N - 1, Mx * (N - 1) + 1)

    a = best_chromosome[: M + 1]
    b = np.concatenate([[0], best_chromosome[M + 1 :]])
    y_hat_fine = np.full(len(x_fine), a[0], dtype=float)
    for k in range(1, M + 1):
        y_hat_fine += (
            a[k] * np.cos(2 * np.pi * k * x_fine / N)
            + b[k] * np.sin(2 * np.pi * k * x_fine / N)
        )

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(best_fitness_per_gen) + 1), best_fitness_per_gen, "go-")
    plt.grid()
    plt.title("GA: Najlepszy fitness w zależności od generacji")
    plt.xlabel("Pokolenie")
    plt.ylabel("Fitness")

    plt.subplot(2, 1, 2)
    plt.plot(x, y_fragment, "b--", label="Dane filtrowane (fragment)", linewidth=1.5)
    plt.plot(x_fine, y_hat_fine, "r-", label=f"Generowany sygnał GA (M={M})", linewidth=1.5)
    plt.grid()
    plt.xlabel("t - próbka")
    plt.ylabel("Wartość sygnału")
    plt.legend()

    plt.tight_layout()
    plt.show()

def ga_approximate(
    y_fragment,
    M,
    minima_idx,
    maksima_idx,
    pop_size=50,
    num_generations=100,
    elite_size=2,
    pcross=0.8,
    pmut=0.02,
    sigma_factor=0.05,
    postep=False,
):
    best_chromosome, best_fitness_per_gen = run_ga(
        y_fragment,
        M,
        minima_idx,
        maksima_idx,
        pop_size,
        num_generations,
        elite_size,
        pcross,
        pmut,
        sigma_factor,
        postep,
    )

    #rysuj_ga_results(y_fragment, best_chromosome, best_fitness_per_gen, M)
    a = best_chromosome[: M + 1]
    b = np.concatenate([[0], best_chromosome[M + 1 :]])
    return a, b, best_fitness_per_gen[-1]
