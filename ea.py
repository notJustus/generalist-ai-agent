# 2 EAs
# 2 ememy groups 

# Training per EA per enemy group 
# -> Training runs get saved in train folder
# -> an experiment can be given a name
# -> One file listing the parameters of the run
# -> EA1, EA2
#       -> [ENG1], [ENG2]
#           -> [RUN1,..,RUN10]
#               -> best_individul, best_over_gen, mean_over_gen # perhaps also others

# For conda
# pyyaml, numpy, pandas

import sys
import time
import math
import argparse
import numpy as np
import pandas as pd
from itertools import product
from evoman.environment import Environment
from demo_controller import player_controller
from tqdm import tqdm
import os
import csv
import yaml


def initialize_population(n_vars, population_size):
    """
    Initialize population with 265 random values each, plus a sigma value that is used for mutation.
    """
    population = np.random.uniform(-1, 1, (population_size, n_vars + 1))  # plus sigma
    population[:, -1] = np.random.uniform(2, 4, population_size)  # make sure that sigma is non-negative
    return population

def remove_sigma(individual):
    """
    Remove sigma from the gene pool of an individual.
    """
    return individual[:-1]

def simulation(env, x):
    x_cleaned = remove_sigma(x)
    fitness, _, _, _ = env.play(pcont=x_cleaned)
    return fitness

def mutation(children, population_size, mutation_p):
    """
    Implements uncorrelated mutation with one step size.
    """
    tau = 1 / (np.sqrt(population_size))  # Smaller tau for slower adaptation

    for child in children:
        # Decide whether to mutate this child
        if np.random.uniform() <= mutation_p:
            # Extract the last gene as the standard deviation
            sigma_child = child[-1]

            # Self-adaptive mutation for the standard deviation
            new_sigma = sigma_child * np.exp(tau * np.random.normal(0, 1))
            new_sigma = np.clip(new_sigma, 0.1, 4)  # Clamp sigma to a max value

            child[-1] = new_sigma  # Update the child's standard deviation

            # Apply uncorrelated mutation to other genes
            for i in range(len(child) - 1):
                mutation_value = np.random.normal(0, new_sigma)  # Use the updated standard deviation
                child[i] += mutation_value  # Mutate each gene

            # Clip the gene values to ensure they remain within bounds [-1, 1]
            child[:-1] = np.clip(child[:-1], -1, 1)

    return children

def parent_selection(current_population, current_pop_fitness, tournament_size, number_of_parents=2):
    """
    Implements tournament selection.
    """
    selected_parents = []

    for _ in range(number_of_parents):
        fitnesses = []

        # Select individuals for tournament
        tournament_individuals = np.random.choice(len(current_population), tournament_size, replace=False)

        # Get fitness values for tournament individuals
        for individual_index in tournament_individuals:
            temp_fitness = current_pop_fitness[individual_index]
            fitnesses.append(temp_fitness)

        # Select winner -highest fitness- as parent
        best_individual = tournament_individuals[np.argmax(fitnesses)]
        selected_parents.append(current_population[best_individual])

    return selected_parents

def recombination(parents):
    """
    Implements whole arithmetic recombination.
    """
    offspring = np.mean(parents, axis=0)  # Create one offspring from the average of the parents
    return [offspring]  # Return as a list

def evaluate_population(env, current_population):
    return np.array([simulation(env, individual) for individual in current_population])

# Outdated
def survivor_selection(population, fitness_values, population_size):
    """"
    Implements μ + λ Selection. Orders population by fitness and takes μ fittest individuals.
    """
    sorted_pop = np.argsort(fitness_values)[::-1]
    selected_indivduals = population[sorted_pop[:population_size]]
    selected_fitness = fitness_values[sorted_pop[:population_size]]
    return selected_indivduals, selected_fitness

def survivor_selection2(pop, fit_pop, offspring, fit_offspring, population_size):
    """"
    Implements μ + λ Selection. Orders population by fitness and takes 20% fittest individuals
    and 80% of fittest offspring.
    """
    test = 0.2
    pop_size_survival = int(test * population_size)
    offspring_size_survival = population_size - pop_size_survival

    sorted_pop = np.argsort(fit_pop)[::-1]
    pop_indivduals = pop[sorted_pop[:pop_size_survival]]
    pop_selected_fitness = fit_pop[sorted_pop[:pop_size_survival]]

    sorted_offspring = np.argsort(fit_offspring)[::-1]
    offspring_individuals = offspring[sorted_offspring[:offspring_size_survival]]
    offspring_selected_fitness = fit_offspring[sorted_offspring[:offspring_size_survival]]

    # Combine selected individuals from population and offspring
    combined_individuals = np.vstack((pop_indivduals, offspring_individuals))
    combined_fitness = np.concatenate((pop_selected_fitness, offspring_selected_fitness))

    return combined_individuals, combined_fitness

def get_fitness(pop, fit_pop):
    avg_fitness = np.mean(fit_pop)
    avg_sigma = np.mean(pop[:, -1])
    best_fitness = np.max(fit_pop)
    return best_fitness, avg_fitness, avg_sigma

def log_fitness(path, max_fitness, mean_fitness):
    # Save max fitness
    with open(os.path.join(path, 'max_fitness.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'max_fitness'])
        for i, fitness in enumerate(max_fitness):
            writer.writerow([i + 1, fitness])
    
    # Save mean fitness
    with open(os.path.join(path, 'mean_fitness.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'mean_fitness'])
        for i, fitness in enumerate(mean_fitness):
            writer.writerow([i + 1, fitness])

def log_best_individual(path, best_individual):
    with open(os.path.join(path, 'best_individual.txt'), 'w') as f:
        for value in best_individual:
            f.write(f"{value}\n")

def run_experiment(env, hyper_param, path, n_vars):
    max_fitness = []
    mean_fitness = []
    best_individual = []

    pop = initialize_population(n_vars, hyper_param["pop_size"])
    fit_pop = evaluate_population(env, pop)
    
    # Log the intial random population as GEN 0
    max_fit, mean_fit, avg_sigma = get_fitness(pop, fit_pop)
    max_fitness.append(max_fit)
    mean_fitness.append(mean_fit)

    #log_fitness(experiment_name, 0, best_fitness, avg_fitness, avg_sigma)

    ini_g = 0
    for gen in range(ini_g + 1, hyper_param["n_generations"]):
        # book recommends between 5-7, could also be tested as a hyperparamter
        generational_gap = 5
        offspring_size = generational_gap * hyper_param["pop_size"]
        num_love_making = math.ceil(offspring_size / hyper_param["n_parents"])
        offspring = []
        for parents in range(num_love_making):
            parents = parent_selection(pop, fit_pop, hyper_param["tour_size"], hyper_param["n_parents"])
            children = recombination(parents)
            offspring.extend(children)

        # Mutation
        offspring = mutation(np.array(offspring), hyper_param["pop_size"], hyper_param["muatation_p"])
        fit_offspring = evaluate_population(env, offspring)

        # Survivor Selection
        pop, fit_pop = survivor_selection2(pop, fit_pop, offspring, fit_offspring, hyper_param["pop_size"])

        # Log the fitness AFTER performing all EA steps (GEN starting with 1)
        max_fit, mean_fit, avg_sigma = get_fitness(pop, fit_pop)
        max_fitness.append(max_fit)
        mean_fitness.append(mean_fit)
        #log_fitness(experiment_name, gen, best_fitness, avg_fitness, avg_sigma)

        best_individual = pop[np.argmax(fit_pop)]
        #np.savetxt(f"{experiment_name}/best.txt", pop[best_individual])
    
    log_fitness(path, max_fitness, mean_fitness)
    log_best_individual(path, best_individual)

def mkdir(name, base=None):
    if base:
        name = os.path.join(base, name)
    if not os.path.exists(name):
        os.makedirs(name)
    return name

def init_params(args):
    # Turn off visuals
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Retrieve config
    with open("configs/debug.yaml", "r") as file:
        config = yaml.safe_load(file)

    base_path = mkdir(name=config["env"]["path"], base="results")
    
    envs = [
        Environment(experiment_name=base_path,
                    enemies=args.enemy_group_1,
                    multiplemode=config["env"]["multiplemode"],
                    playermode="ai",
                    player_controller=player_controller(config["env"]["n_hidden_neurons"]),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    randomini=config["env"]["randomini"],
                    visuals=config["env"]["visuals"]),
        
        Environment(experiment_name=base_path,
                    enemies=args.enemy_group_2,
                    multiplemode=config["env"]["multiplemode"],
                    playermode="ai",
                    player_controller=player_controller(config["env"]["n_hidden_neurons"]),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    randomini=config["env"]["randomini"],
                    visuals=config["env"]["visuals"])
    ]
    #base_path = os.path.join("results", config["env"]["path"])

    # To-DO
    for enemy_group, env in enumerate(envs):
        #env_path = os.path.join(base_path, f"enemy_group_{enemy_group}")
        env_path = mkdir(name=f"enemy_group_{enemy_group}", base=base_path)
        n_vars = (env.get_num_sensors() + 1) * config["env"]["n_hidden_neurons"] + (config["env"]["n_hidden_neurons"] + 1) * 5
        
        for run in range(config["env"]["n_runs"]):
            #run_path = os.path.join(env_path, f"run_{run}")
            run_path = mkdir(name=f"run_{run}", base=env_path)

            run_experiment(env, config["hyperparamters"], run_path, n_vars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enemy_group_1", nargs='*', type=int, help="Enemy number: 1 to 8")
    parser.add_argument("--enemy_group_2", nargs='*', type=int, help="Enemy number: 1 to 8")

    args = parser.parse_args()

    init_params(args)
