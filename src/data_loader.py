import os
import pandas as pd
#point to my data folder
DATA_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))
def load_and_merge(sample_n=100_000):
    """"loading movie.csv and rating.csv files and merging them on movieId, dropping duplicates"""
    movies=pd.read_csv(os.path.join(DATA_DIR,'movie.csv'))
    rating=pd.read_csv(os.path.join(DATA_DIR,'rating.csv'))
    df=rating.merge(movies,on='movieId').drop_duplicates().reset_index(drop=True)
    # to make sure if toolarge, sample down to n
    if isinstance(sample_n, int) and len(df) > sample_n:
        df=df.sample(sample_n, random_state=42).reset_index(drop=True)
    return df