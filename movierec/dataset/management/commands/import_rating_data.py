import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "movierec.settings")
from django.core.management.base import BaseCommand
import csv

from dataset.models import Movie, Rating

class Command(BaseCommand):
    help = 'Import data from a CSV file into Django models'

    def handle(self, *args, **kwargs):
        with open('/home/saru/Documents/NewsRecommend/news_recommender/movierec/ratings.csv', 'r') as f:
            reader = csv.DictReader(f)
            datas = []
            for row in reader:
                movieId = row['movieId']
                userId = row['userId']
                rating = row['rating']
                timestamp=row['timestamp']
                
                movie, created = Movie.objects.get_or_create(
                    movieId=movieId
                )
                data = Rating(
                    movieId=movie,
                    userId=userId,
                    rating=rating,
                    timestamp=timestamp
                )
                datas.append(data)
            Rating.objects.bulk_create(datas)