import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_both():
    pass


def plot_all_runs(enemy_group, n_generations, max_fitness, max_fitness_std, mean_fitness, mean_fitness_std, path):
    x_values = list(range(0, n_generations))

    plt.figure(figsize=(10, 6))

    plt.plot(x_values, max_fitness, label='Max Fitness', color='b')
    plt.fill_between(x_values, 
                     np.array(max_fitness) - np.array(max_fitness_std), 
                     np.array(max_fitness) + np.array(max_fitness_std), 
                     color='b', alpha=0.1)
    
    plt.plot(x_values, mean_fitness, label='Mean Fitness', color='g')
    plt.fill_between(x_values, 
                     np.array(mean_fitness) - np.array(mean_fitness_std), 
                     np.array(mean_fitness) + np.array(mean_fitness_std), 
                     color='g', alpha=0.1)
    
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title(f"Enemy Group {enemy_group}")
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"eng_{enemy_group}"), dpi=300, bbox_inches='tight')
    #plt.show()


def plot_single_run(n_generations, max_fitness, mean_fitness):
    x_values = list(range(0, n_generations))

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


def parse_n_runs(n, path_enemy_group):
    fitness = {}
    for i in range(n):
        max_fitness, mean_fitness = parse_single_run(os.path.join(path_enemy_group, f"run_{i}"))
        fitness[i] = {
            'max': max_fitness,
            'mean': mean_fitness
        }
    
    return fitness

def to_average_over_n_runs(fitness, n_generations):
    #n_generations = len(fitness[0]['max'])
    n_runs = len(fitness)
    
    max_values = np.zeros((n_runs, n_generations))
    mean_values = np.zeros((n_runs, n_generations))

    for run in range(n_runs):
        max_values[run] = fitness[run]['max']
        mean_values[run] = fitness[run]['mean']
    
    max_fitness = np.mean(max_values, axis=0)
    max_fitness_std = np.std(max_values, axis=0)
    mean_fitness = np.mean(mean_values, axis=0)
    mean_fitness_std = np.std(mean_values, axis=0)

    return max_fitness, max_fitness_std, mean_fitness, mean_fitness_std

def plot_training(args):
    path = os.path.join("plots", args.experiment_name)
    # Enemy Group 1
    fitness_eng1 = parse_n_runs(args.n_runs, os.path.join(args.path, f"enemy_group_{0}"))
    max_fitness_eng1, max_fitness_std_eng1, mean_fitness_eng1, mean_fitness_std_eng1 = to_average_over_n_runs(fitness_eng1, 5)
    plot_all_runs("1,2,3", 5, max_fitness_eng1, max_fitness_std_eng1, mean_fitness_eng1, mean_fitness_std_eng1, path)
    # Enemy Group 2
    fitness_eng2 = parse_n_runs(args.n_runs, os.path.join(args.path, f"enemy_group_{1}"))
    max_fitness_eng2, max_fitness_std_eng2, mean_fitness_eng2, mean_fitness_std_eng2 = to_average_over_n_runs(fitness_eng2, 5)
    plot_all_runs("4,5,6", 5, max_fitness_eng2, max_fitness_std_eng2, mean_fitness_eng2, mean_fitness_std_eng2, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the experiment")
    parser.add_argument("--experiment_name", type=str, default="training")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--n_generations", type=int, required=True)

    args = parser.parse_args()

    if not os.path.exists(os.path.join("plots", args.experiment_name)):
        os.makedirs(os.path.join("plots", args.experiment_name))

    plot_training(args)
