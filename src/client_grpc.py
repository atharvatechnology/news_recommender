import grpc
from proto.recommendation_pb2_grpc import NewsRecommendationServiceStub
from proto.recommendation_pb2 import GetNewsRecommendationRequest

if __name__ == "__main__":
    channel = grpc.insecure_channel("localhost:50052")
    stub = NewsRecommendationServiceStub(channel)
    req = GetNewsRecommendationRequest(userId=12)
    response = stub.GetNewsRecommendation(req)
    print(response.newsIds)
