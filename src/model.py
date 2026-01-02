import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import numpy as np

def prepare_X_y(df):
    """"
    columns of the df will be --> usedId, movieId, genres, rating
    returns X (features) and y (target)
    i wil treat genres as its first listed genre for simplicity
    """
    #extracting primary genres
    df=df.copy()
    df['primary_genre']=df['genres'].str.split('|').str[0]
    X = df[['userId', 'movieId', 'primary_genre']]
    y=df['rating']
    return X,y

def build_pipeline():
    """"
    building a sckit-learn pipeline--> OneHotEncoder userId, movieId, and primary_genre
    and MLPRegressor
    """
    #column transformer: one-hot on each categorical column
    ct=ColumnTransformer([
        ('user', OneHotEncoder(handle_unknown='ignore'), ['userId']),
        ('movie', OneHotEncoder(handle_unknown='ignore'), ['movieId']),
        ('genre', OneHotEncoder(handle_unknown='ignore'), ['primary_genre']),
    ], remainder='drop')

    mlp = MLPRegressor(
        hidden_layer_sizes=(50,), #baseline: one hidden layer of only 50 neurons (just like my ex)
        learning_rate_init=0.001,
        max_iter=50,
        random_state=42,
    )
    pipe = Pipeline([
        ('encode', ct),
        ('mlp', mlp),
    ])
    return pipe

def train_and_evaluate(df, sample_size=None):
    """
    -> prepare X,y
    -> slipt 80/20
    -> Grid-search over a tiny hyperparameter grid
    -> evaluate on test set (RMSE)
    -> save the best model to models/baseline_mlp.pkl
    """
    X, y =prepare_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    param_grid = {
        'mlp__hidden_layer_sizes': [(50,),(100,)],  # just one architecture
        'mlp__learning_rate_init': [0.001,  0.01],  # just one learning rate
    }
    cv = 3
    n_jobs = -1
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs,
        verbose=1
    )

    gs.fit(X_train, y_train)

    best=gs.best_estimator_
    y_pred=best.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Baseline MLP RMSE: {rmse:.4f}")
    print("Best Params: ", gs.best_params_)

    #saving the model
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
    joblib.dump(best, os.path.join(os.path.dirname(__file__), '..',
                                   'models', 'baseline_mlp.pkl'))
    print("Saved baseline MLP to models/baseline_mlp.pkl")
    return best, X_test, y_test