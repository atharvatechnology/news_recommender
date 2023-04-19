import sys
import numpy as np
import pandas as pd
from os import path


def get_file_path(file_name):
    return path.join(DATA_PATH, file_name)


def save_dataframe_to_csv(df, file_name):
    df.to_csv(get_file_path(file_name), index=False)


def preprocess_data():
    users_df = pd.read_csv(get_file_path("users.csv"))
    news_df = pd.read_csv(get_file_path("news.csv"))
    ratings_df = pd.read_csv(get_file_path("ratings.csv"))
    likes_df = pd.read_csv(get_file_path("likes.csv"))

    # add an id column beginning from 0
    users_df["user_id"] = [i for i in range(users_df.shape[0])]
    news_df["news_id"] = [i for i in range(news_df.shape[0])]

    # Count the ratings by deleted users
    sum = 0
    for i, x in enumerate(ratings_df["User"]):
        if len(users_df[users_df.User == x].index) != 1:
            # print(f"user {x} in of {i}th row")
            sum += 1
    print(f"Total ratings to discard {sum}")

    # remove user ratings associated to deleted users
    ratings_df = ratings_df[ratings_df.User.isin(users_df.User)]

    # map user id to the index of the user in user table
    ratings_df["news_id"] = ratings_df["News"].apply(
        lambda x: news_df[news_df.Id == x].index[0]
    )

    # map news id to the index of the news table
    ratings_df["user_id"] = ratings_df["User"].apply(
        lambda x: users_df[users_df.User == x].index[0]
    )

    # Calculate rating of user
    # TODO: Try different methods to calculate implicit rating
    # currently, we use the number of likes and comments as the rating
    # the weightage of like and comment is 1:1
    # even if a user has multiple comments on a news, we only count it as 1
    equal_weight_rating = lambda x: x.Like + (x.Comment + 1) / (x.Comment + 1)
    ratings_df["rating"] = ratings_df[["Like", "Comment"]].apply(
        equal_weight_rating, axis=1
    )

    # save the processed data into files
    users_df.to_csv(get_file_path("users_processed.csv"), index=False)
    news_df.to_csv(get_file_path("news_processed.csv"), index=False)
    ratings_df.to_csv(get_file_path("ratings_processed.csv"), index=False)


if __name__ == "__main__":
    print("Hello World!")
    DATA_PATH = path.abspath(sys.argv[1])
    # data_root = 'data/'
    preprocess_data()
    print(f"Done! Data saved in {DATA_PATH}/<data>_processed.csv")
