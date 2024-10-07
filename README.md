# Evolutionary Computing

## Setting up Environment

`conda env create -f environment.yml`

`conda activate evoman`

> This is gonna automatically install all dependencies if you have conda installed

## Framework setup

In the `/configs` folder you can add your own configurations of the framework by creating new .yaml files.

To run an EA you simply pass the config file to ea.py and it will generate all results in the results folder.

`python ea.py --config configs/debug.yaml`

The files are organized as such:

````
experiment_name/
│
├── enemy_group_0/
│   ├── run_0
│   ├── run_1
│   └── run_2
│       └── max_fitness.csv
│       └── mean_fitness.csv
│       └── best_individual.txt
│
├── enemy_group_1/
│   ├── run_0
│   ├── run_1
│   └── run_2
│       └── max_fitness.csv
│       └── mean_fitness.csv
│       └── best_individual.txt
```