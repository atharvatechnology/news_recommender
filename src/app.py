import json
from os import path
from flask import Flask, request
import pandas as pd
import torch
from .evaluate import (
    user_recommendations,
)

app = Flask(__name__)


# specify response type as json
@app.route("/", methods=["POST"])
def hello():
    payload = request.get_json()
    print(f"*****payload: {payload}*****")
    news_df = pd.read_csv(path.join("../data", "news_processed.csv"))
    ratings_df = pd.read_csv(path.join("../data", "ratings_processed.csv"))
    embeddings = torch.load(path.join("../data", "embeddings.pt"))["embedding_dict"]
    # print(news_df.head())
    recs = user_recommendations(embeddings, news_df, ratings_df, user_id=7170)
    rec_dict = {"recs": recs}

    return json.dumps(rec_dict)
