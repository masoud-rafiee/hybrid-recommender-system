#***********************************************************************************
# Masoud Rafiee
# CS446 - Final Project: Optimized Hybrid Recommender System Using MLP, GA, and PSO
# 2025, April
# ***********************************************************************************
#The goal of the project is to build a hybrid recommender system that uses:
# 1. MLP for predicting user ratings.
# 2. GA and PSO for optimizing the hyperparameters of the MLP model.
# 3. A Recommender System framework to generate personalized movie recommendations.
# ***********************************************************************************
# LIBRARIES:
from data_loader import load_and_merge
from src.eda import plot_genre_distribution, count_movies_over_threshold
from eda import find_similar_users
from model import train_and_evaluate
# ********************************s***************************************************

def main():
    # load and merging data
    df=load_and_merge()

    #inspecting shape and duplication
    print("Rows x Columns: ",df.shape)
    print("Duplicate rows: ",df.duplicated().sum())

    #plt genre distrib.
    plot_genre_distribution(df, save_path="../figures/genre_distribution.png")

    #count movies rated by > 50 users
    movies_50= count_movies_over_threshold(df, threshold=50)
    print(f"Movies rated by > 50 users: {movies_50}")

    #cout users rated over 50 movies
    users_50= count_movies_over_threshold(df, threshold=50)
    print(f"Users who rated > 50 movies: {users_50}")

    sim=find_similar_users(df, target_user_id=1, min_common_genres=2)
    print(f"Users Sharing >= 2 genres with user 1: {len(sim)}→ {sorted(sim)[:10]}…")

    ######################

    print ("\n--- Training baseline MLP ---")
    #train_and_evaluate(df)
    best, X_test, y_test = train_and_evaluate(df)

    # picking one from (user,movie) from X_test
    sample = X_test.sample(1, random_state=42)
    true_rating = y_test.loc[sample.index[0]]
    #predicting the rating using trained mlp model
    pred_rating = best.predict(sample)[0]

    print("\n--- Sample prediction on test pair ---")
    print(f"User {sample.userId.values[0]}, Movie {sample.movieId.values[0]}")
    print(f"True rating = {true_rating:.1f}, Predicted (accuracy) = {pred_rating:.2f}")
    print(f"Error = {abs(pred_rating - true_rating):.2f}")

if __name__=="__main__":
    main()
