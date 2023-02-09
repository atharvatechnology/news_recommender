from django.db import models

# Create your models here.
class Movie(models.Model):
    movieId=models.IntegerField(primary_key=True)
    title=models.CharField(max_length=500)
    genres=models.CharField(max_length=500)

    def __str__(self):
        return self.title

class Tag(models.Model):
    userId=models.IntegerField()
    movieId=models.ManyToManyField(Movie,related_name='tags')
    tag=models.CharField(max_length=500)
    timestamp=models.DateTimeField(auto_now_add= True)

    def __str__(self):
        return self.tag

class Link(models.Model):
    movieId=models.ForeignKey(Movie,related_name='links',on_delete=models.CASCADE)
    imdbId=models.IntegerField()
    tmdbId=models.CharField(max_length=100)

class Rating(models.Model):
    userId=models.IntegerField()
    movieId=models.ForeignKey(Movie,related_name='ratings',on_delete=models.CASCADE)
    rating=models.FloatField()
    timestamp=models.DateTimeField(auto_now_add= True)








