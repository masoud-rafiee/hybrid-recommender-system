import json
import random
import os
import joblib
import numpy as np
from copy import deepcopy
from model import build_pipeline, prepare_X_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#same search space as GA
SEARCH_SPACE = {
    'hidden_layer_sizes': [(50,), (100,), (50,50)],
    'learning_rate_init': [1e-4, 1e-3, 1e-2],
}

def init_particle():
    """rnndomly initialize one particle"""
    return {
        'position': {
            'hidden_layer_sizes': random.choice(SEARCH_SPACE['hidden_layer_sizes']),
            'learning_rate_init': random.choice(SEARCH_SPACE['learning_rate_init'])
        },
        #starting with zero vlocity (i will swap discrete values so v just flags)
        'velocity': {'hidden_layer_sizes': 0, 'learning_rate_init': 0},
        'best_pos': {},
        'best_score': float('inf')
    }

def evaluate(particle, df):
    """compute RMSE at the particle’s current position"""
    cfg = particle['position']
    X, y = prepare_X_y(df)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline()
    pipe.set_params(
        mlp__hidden_layer_sizes=cfg['hidden_layer_sizes'],
        mlp__learning_rate_init=cfg['learning_rate_init']
    )
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_val)
    rmse  = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

def pso(df, n_particles=6, generations=5, w=0.5, c1=1.0, c2=1.0):
    """
    Runs PSO and returns: best_cfg: the top hyperparameter set found (a dict) adn history: list of best RMSE per generation
    """
    #to initialize swarm and best trackers
    swarm = [init_particle() for _ in range(n_particles)]
    best_cfg: dict = {}
    best_score = float('inf')
    history: list [float] = []

    for gen in range(generations):
        for p in swarm:
            #so to evaluate current position:
            score = evaluate(p, df)

            #i uppdate personal best
            if score < p['best_score']:
                p['best_score'] = score
                p['best_pos']   = deepcopy(p['position'])

            # update global best
            if score < best_score:
                best_score = score
                best_cfg = deepcopy(p['position'])
        #record and print
        history.append(best_score)
        print(f"Gen {gen + 1}: Global best RMSE = {best_score:.4f} at {best_cfg}")

        # velocity/position update also :
        for p in swarm:
            for key in p['position']:
                if random.random() < c1:
                    p['position'][key] = p['best_pos'][key]
                if random.random() < c2:
                    p['position'][key] = best_cfg[key]
                if random.random() < 0.1:
                    p['position'][key] = random.choice(SEARCH_SPACE[key])

    return best_cfg, history

def save_pso_model(df, best_cfg, model_path="../models/pso_mlp.pkl"):
    """train on full train set with best_cfg and save model"""
    X, y = prepare_X_y(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline()
    pipe.set_params(
        mlp__hidden_layer_sizes=best_cfg['hidden_layer_sizes'],
        mlp__learning_rate_init=best_cfg['learning_rate_init']
    )
    pipe.fit(X_tr, y_tr)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"Saved PSO-optimized MLP to {model_path}")

if __name__ == "__main__":
    from data_loader import load_and_merge
    df = load_and_merge(sample_n=1000) #CAN BE CHANGED to 100000 but took so long for my computer so I made it this to be faster
    best_cfg, hist = pso(df, n_particles=6, generations=5)
    print("\nPSO completed. Best config:", best_cfg)
    #to make sure the models directory exists
    os.makedirs(os.path.join('..', 'models'), exist_ok=True)

    #dumping the PSO history
    with open(os.path.join('..', 'models', 'pso_history.json'), 'w') as f:
        #json.dump(hist, f)
        data = json.dumps(hist)  #returns a str
        f.write(data)

    save_pso_model(df, best_cfg)
