import os
import sys
import pandas as pd
import numpy as np


DOT = "dot"
COSINE = "cosine"


def compute_scores(query_embedding, item_embeddings, measure=DOT):
    """Computes the scores of the candidates given a query.
    Args:
      query_embedding: a vector of shape [k], representing the query embedding.
      item_embeddings: a matrix of shape [N, k], such that row i is the embedding
        of item i.
      measure: a string specifying the similarity measure to be used. Can be
        either DOT or COSINE.
    Returns:
      scores: a vector of shape [N], such that scores[i] is the score of item i.
    """
    scores = np.dot(item_embeddings, query_embedding)
    if measure == COSINE:
        u_norm = np.linalg.norm(query_embedding)
        V_norm = np.linalg.norm(item_embeddings, axis=1)
        norm_product = u_norm * V_norm
        scores = scores * (1.0 / norm_product)
    return scores


def user_recommendations(model, measure=DOT, exclude_rated=False, k=6, user_id=7170):
    # user_id = 7170
    scores = compute_scores(
        model.embeddings["user_embed"][user_id], model.embeddings["item_embed"], measure
    )
    score_key = measure + " score"
    df = pd.DataFrame(
        {
            score_key: list(scores),
            "item_id": news_df["news_id"],
            "titles": news_df["Title"],
            "categories": news_df["Category"],
        }
    )
    if exclude_rated:
        # remove the items that are already rated
        rated_items = ratings_df[ratings_df.user_id == user_id]["news_id"].values
        df = df[df.item_id.apply(lambda item_id: item_id not in rated_items)]
    print(df.sort_values([score_key], ascending=False).head(k))


def item_neighbors(model, title_substring, measure=DOT, k=6, item_id=2399):
    # Search for item ids that match the given substring
    ids = news_df[news_df["Title"].str.contains(title_substring)].index.values
    titles = news_df.iloc[ids]["Title"].values

    if len(titles) == 0:
        # raise ValueError("Found no movies with title %s" % title_substring)
        print("Found no movies with title %s" % title_substring)

    print("Nearest neighbors of : %s." % titles[0])
    if len(titles) > 1:
        print(
            "[Found more than one matching movie. Other candidates: {}]".format(
                ", ".join(titles[1:])
            )
        )

    # item_id = 2399
    scores = compute_scores(
        model.embeddings["item_embed"][item_id], model.embeddings["item_embed"], measure
    )
    score_key = measure + " score"
    df = pd.DataFrame(
        {
            score_key: list(scores),
            "titles": news_df["Title"],
            "categories": news_df["Category"],
        }
    )
    print(df.sort_values([score_key], ascending=False).head(k))


if __name__ == "__main__":
    DATA_PATH = os.path.abspath(sys.argv[1])

    news_df = pd.read_csv(DATA_PATH("news.csv"))
    ratings_df = pd.read_csv(DATA_PATH("ratings.csv"))
