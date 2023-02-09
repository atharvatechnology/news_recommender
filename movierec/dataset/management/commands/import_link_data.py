from django.core.management.base import BaseCommand
import csv

from dataset.models import Movie, Link

class Command(BaseCommand):
    help = 'Import data from a CSV file into Django models'

    def handle(self, *args, **kwargs):
        with open('/home/saru/Documents/NewsRecommend/news_recommender/movierec/links.csv', 'r') as f:
            reader = csv.DictReader(f)
            datas = []
            for row in reader:
                movieId = row['movieId']
                imdbId = row['imdbId']
                tmdbId = row['tmdbId']
                
                movie, created = Movie.objects.get_or_create(
                    movieId=movieId
                )
                data = Link(
                    movieId=movie,
                    imdbId=imdbId,
                    tmdbId=tmdbId,
                )
                datas.append(data)
            Link.objects.bulk_create(datas)