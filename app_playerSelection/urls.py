from django.urls import path
from app_playerSelection import views

app_name = 'playerSelection'

urlpatterns = [

    path('selectplayers', views.selectPlayers, name='selectplayers'),
]