import pandas as pd
import torch
import os
import numpy as np

# from evaluate import (
#     get_file_path,
#     user_recommendations,
#     DOT,
#     COSINE,
# )
# Create model object
model = None

DOT = "dot"
COSINE = "cosine"


def get_file_path(file_name):
    return os.path.join("data", file_name)


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


def user_recommendations(
    embeddings, news_df, ratings_df, measure=DOT, exclude_rated=False, k=6, user_id=7170
):
    # user_id = 7170
    scores = compute_scores(
        embeddings["user_embed"][user_id], embeddings["item_embed"], measure
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


def entry_point_function_name(data, context):
    """
    Works on data and context to create model object or process inference request.
    Following sample demonstrates how model object can be initialized for jit mode.
    Similarly you can do it for eager mode models.
    :param data: Input data for prediction
    :param context: context contains model server system properties
    :return: prediction output
    """
    global model

    if not data:
        manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        # Read model serialize/pt file
        serialized_file = manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        model = torch.load(get_file_path("embeddings.pt"))["embedding_dict"]

    else:
        # infer and return result
        print("***************data: ", data)
        news_df = pd.read_csv(get_file_path("news_processed.csv"))
        ratings_df = pd.read_csv(get_file_path("ratings_processed.csv"))
        return user_recommendations(
            model, news_df, ratings_df, measure=COSINE, user_id=data
        )
