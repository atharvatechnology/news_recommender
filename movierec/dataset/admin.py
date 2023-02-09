from django.contrib import admin
from . models import Movie
# Register your models here.
@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display=['movieId','title','genres']

# @admin.register(Tag)
# class MovieAdmin(admin.ModelAdmin):
#     list_display=['movieId','userId','tag','timestamp']