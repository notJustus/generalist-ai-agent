import os
import csv
import matplotlib.pyplot as plt

def plot_both():
    pass

def plot_single_run(generations, max_fitness, mean_fitness):
    x_values = list(range(0, generations))

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, max_fitness, label='Max Fitness', color='blue', linestyle='-', marker='o')
    plt.plot(x_values, mean_fitness, label='Mean Fitness', color='orange', linestyle='-', marker='s')

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness Values Over Generations')
    plt.legend()
    plt.grid()

    plt.show()


def parse_single_run(path):
    max_fitness = []
    mean_fitness = []

    with open(os.path.join(path, "max_fitness.csv"), 'r') as max_file:
        reader = csv.reader(max_file)
        next(reader)

        for line in reader:
            max_value = float(line[1])
            max_fitness.append(max_value)

    with open(os.path.join(path, "mean_fitness.csv"), 'r') as mean_file:
        reader = csv.reader(mean_file)
        next(reader)

        for line in reader:
            mean_value = float(line[1])
            mean_fitness.append(mean_value)

    return max_fitness, mean_fitness


if __name__ == "__main__":
    path="results/test01/enemy_group_0/run_0"

    max_fitness, mean_fitness = parse_single_run(path)
    plot_single_run(len(max_fitness), max_fitness, mean_fitness)