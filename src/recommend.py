import os
import joblib
import pandas as pd
from data_loader import load_and_merge
from model import prepare_X_y

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))


def load_movies():
    #i only need movieId and title (and genres if you like)
    movies = pd.read_csv(os.path.join(DATA_DIR, 'movie.csv'), usecols=['movieId', 'title'])
    return movies


def get_recommendations(model_path, user_id, top_n=5):
    #load the trained pipeline
    pipe = joblib.load(model_path)

    #load movies list
    movies = load_movies()

    #to build a DataFrame with every movie for this user
    df_all = pd.DataFrame({
        'userId': user_id,
        'movieId': movies['movieId']
    })
    #wif your pipeline expects a 'primary_genre' column:
    #load full merged df just for genres
    full = load_and_merge(sample_n=None)  # None → use all rows
    #map movieId ---> first genre
    genre_map = (full
    .assign(primary_genre=full.genres.str.split('|').str[0])
    .drop_duplicates('movieId')
    .set_index('movieId')['primary_genre'])
    df_all['primary_genre'] = df_all['movieId'].map(genre_map).fillna('Drama')

    #predicting
    df_all['pred_rating'] = pipe.predict(df_all)

    #join titles and sort
    recs = (df_all
    .merge(movies, on='movieId')
    .sort_values('pred_rating', ascending=False)
    .head(top_n)
    [['movieId', 'title', 'pred_rating']])
    return recs


if __name__ == "__main__":
    user = 1
    for label, fname in [
        ('Baseline', 'baseline_mlp.pkl'),
        ('GA', 'ga_mlp.pkl'),
        ('PSO', 'pso_mlp.pkl'),
    ]:
        path = os.path.join(MODEL_DIR, fname)
        print(f"\nTop 5 recommendations for user {user} using {label}:")
        print(get_recommendations(path, user, top_n=5).to_string(index=False))
