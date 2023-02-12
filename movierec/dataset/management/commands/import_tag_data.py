from django.core.management.base import BaseCommand
import csv

from dataset.models import Movie, Tag

class Command(BaseCommand):
    help = 'Import data from a CSV file into Django models'

    def handle(self, *args, **kwargs):
        with open('/home/saru/Documents/NewsRecommend/news_recommender/movierec/tags.csv', 'r') as f:
            reader = csv.DictReader(f)
            datas = []
            for row in reader:
                movieId = row['movieId']
                userId = row['userId']
                tag = row['tag']
                timestamp=row['timestamp']
                
                movie, created = Movie.objects.get_or_create(
                    movieId=movieId
                )
                data = Tag(
                    movieId=movie,
                    userId=userId,
                    tag=tag,
                    timestamp=timestamp
                )
                datas.append(data)
            Tag.objects.bulk_create(datas)