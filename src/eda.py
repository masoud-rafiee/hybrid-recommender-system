import os
from itertools import groupby
import pandas as pd
import matplotlib.pyplot as plt

def plot_genre_distribution(df, save_path=None):
    """
    df must have a 'genres' column where genres are pipe separated (like 'ACTION|DRAMA')
    plots and optionally saves a bar chart of movie counts per genre, sorted descending
    """
    #spliting and explode genres
    genre_series=df['genres'].str.split('|').explode()
    #count per genre
    counts=genre_series.value_counts().sort_values(ascending=False)
    #plot
    plt.figure(figsize=(10,6))
    counts.plot.bar()
    plt.title("Movie Counts per Genre")
    plt.ylabel("Number of Movies")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    #save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved genre plot to {save_path}")
    else:
        plt.show()
    plt.clf() #clear for next plot

def count_movies_over_threshold(df, threshold=50):
    """
    reeturns the numb. of unique movies that have more than 'threshold' ratings
    """
    movie_counts=df.groupby('movieId').size()
    return (movie_counts > threshold).sum()

def count_users_over_threshold(df, threshold=50):
    """
    num of unique users that have more than 'threshold' movies rate
    """
    user_counts=df.groupby('userId').size()
    return (user_counts > threshold).sum()

def find_similar_users(df, target_user_id, min_common_genres=2):
    """
    returns userIds sharing >= min_common_genres with target_user_id
    """
    user_genres= (
        df.loc[df.userId==target_user_id, 'genres']
        .str.split('|')
        .explode()
        .unique()
    )

    #exploe all other users' genres
    other=(
        df[df.userId!=target_user_id]
        .assign(genre=df.genres.str.split('|'))
                .explode('genre')
    )
    #count how many disticint of thise genres each user shares
    overlap = (
        other[ other.genre.isin(user_genres)]
        .groupby('userId')['genre']
        .nunique()
    )
    #return users meeting the threshold
    return set(overlap[overlap >= min_common_genres].index)
