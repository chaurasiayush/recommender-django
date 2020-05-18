from django.urls import path
from . import views

app_name = 'players'

urlpatterns = [
    path('allplayers', views.overallPRec, name='allplayerrecords'),
    path('teamplayers', views.teamPlayers, name='teamplayers')
]