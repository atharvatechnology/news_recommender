from concurrent import futures
from os import path
import grpc
import pandas as pd
import torch
from proto.recommendation_pb2_grpc import (
    NewsRecommendationServiceServicer,
    add_NewsRecommendationServiceServicer_to_server,
)
from proto.recommendation_pb2 import (
    GetNewsRecommendationRequest,
    GetNewsRecommendationResponse,
)
from evaluate import user_recommendations


class Recommender(NewsRecommendationServiceServicer):
    def __init__(self):
        print("Loading data...")
        # self.news_df = pd.read_csv(path.join("data", "news_processed.csv"))
        # self.ratings_df = pd.read_csv(path.join("data", "ratings_processed.csv"))
        # self.embeddings = torch.load(path.join("data", "embeddings.pt"))[
        #     "embedding_dict"
        # ]

    def GetNewsRecommendation(self, request: GetNewsRecommendationRequest, context):
        user_id = request.userId
        # recs = user_recommendations(
        #     self.embeddings, self.news_df, self.ratings_df, user_id=user_id
        # )
        # TODO: Delete this line later
        recs = [90, 91]
        return GetNewsRecommendationResponse(newsIds=recs)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_NewsRecommendationServiceServicer_to_server(Recommender(), server)
    PORT = 50052
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"Recommendation grpc server started on port {PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
