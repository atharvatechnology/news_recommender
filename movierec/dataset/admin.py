from django.contrib import admin
from . models import Movie,Link,Tag,Rating
# Register your models here.
@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display=['movieId','title','genres']

@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display=['id','movieId','userId','tag','timestamp']

@admin.register(Link)
class LinkAdmin(admin.ModelAdmin):
    list_display=['id','movieId','imdbId','tmdbId']
@admin.register(Rating)
class RatingAdmin(admin.ModelAdmin):
    list_display=['id','movieId','userId','rating','timestamp']