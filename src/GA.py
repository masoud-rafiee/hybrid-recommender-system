import json
import random
import joblib
import os
import copy
import numpy as np
from model import build_pipeline, prepare_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# hyperparameter space
SEARCH_SPACE = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'learning_rate_init': [1e-4, 1e-3, 1e-2],
}


# fitness function --> returning validation RMSE for a given conf.
def fitness(config, df, test_size=0.2, random_state=42):
    X, y = prepare_X_y(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # biuld pipeline and set hyperparams
    pipe = build_pipeline()
    pipe.set_params(
        mlp__hidden_layer_sizes=config['hidden_layer_sizes'],
        mlp__learning_rate_init=config['learning_rate_init']
    )
    # fiit & evaluate
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse


def random_config():
    """to pick a random hyperparameter configuration"""
    return {
        'hidden_layer_sizes': random.choice(SEARCH_SPACE['hidden_layer_sizes']),
        'learning_rate_init': random.choice(SEARCH_SPACE['learning_rate_init'])
    }


def crossover(parent1, parent2):
    """swapping one hyperparameter between parents"""
    child = {}
    # randomly pick which param to swap
    if random.random() < 0.5:
        child['hidden_layer_sizes'] = parent1['hidden_layer_sizes']
        child['learning_rate_init'] = parent2['learning_rate_init']
    else:
        child['hidden_layer_sizes'] = parent2['hidden_layer_sizes']
        child['learning_rate_init'] = parent1['learning_rate_init']
    return child


def mutate(config, mutation_rate=0.1):
    """rnndomly mutate one param with probability mutation_rate"""
    cfg = config.copy()
    if random.random() < mutation_rate:
        cfg['hidden_layer_sizes'] = random.choice(SEARCH_SPACE['hidden_layer_sizes'])
    if random.random() < mutation_rate:
        cfg['learning_rate_init'] = random.choice(SEARCH_SPACE['learning_rate_init'])
    return cfg


def run_ga(df, generations=5, population_size=6, elite_size=2, mutation_rate=0.1):
    """
    Runs GA and returns:
      - best_config: the top hyperparameter set found
      - history: list of best RMSE per generation
    """
    # 1) initialize population
    best_config = None
    population = [random_config() for _ in range(population_size)]
    history = []

    for gen in range(generations):
        # 2) evaluate fitness for each
        scores = [(config, fitness(config, df)) for config in population]
        scores.sort(key=lambda x: x[1])  # lowest RMSE first

        # record best
        best_config, best_rmse = scores[0]
        history.append(best_rmse)
        print(f"Gen {gen + 1}: Best RMSE = {best_rmse:.4f} with {best_config}")

        # 3) select elites
        elites = [cfg for cfg, _ in scores[:elite_size]]

        # 4) generate next population
        next_pop = elites.copy()
        while len(next_pop) < population_size:
            # pick two random parents from elites
            p1, p2 = random.sample(elites, 2)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            next_pop.append(child)

        population = next_pop

    return best_config, history


def save_ga_model(df, best_cfg, model_path="../models/ga_mlp.pkl"):
    """so for the train on full train set with best_cfg and save the pipeline, i will do this"""
    X, y = prepare_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe = build_pipeline()
    pipe.set_params(
        mlp__hidden_layer_sizes=best_cfg['hidden_layer_sizes'],
        mlp__learning_rate_init=best_cfg['learning_rate_init']
    )
    pipe.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"Saved GA-optimized MLP to {model_path}")

# example fo rand conf.
if __name__ == "__main__":
    import pandas as pd
    from data_loader import load_and_merge

    df = load_and_merge(sample_n=1000) # I MUST CHANGE THIS TO 100,000 as the assignment wants
    random_conf = {
        'hidden_layer_sizes': random.choice(SEARCH_SPACE['hidden_layer_sizes']),
        'learning_rate_init': random.choice(SEARCH_SPACE['learning_rate_init'])
    }
    print("RMSE for random config:", fitness(random_conf, df))
    best_cfg, hist = run_ga(df, generations=5)
    print("\nGA completed. Best config:", best_cfg)
    #ensure the models directory exists
    os.makedirs(os.path.join('..', 'models'), exist_ok=True)

    #sp dumping  the GA history to JSON
    with open(os.path.join('..', 'models', 'ga_history.json'), 'w') as f:
        #json.dump(hist, f)#ignore pycharm BS :/ or data=json.dumps(hist)
        data = json.dumps(hist)  #returns a str
        f.write(data) #to accept string
    #saving the final GA‐optimized model
    save_ga_model(df, best_cfg)