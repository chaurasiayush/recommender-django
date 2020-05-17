from django.urls import path
from .views import overallPRec
urlpatterns = [

    path('allplayers', overallPRec, name='allplayerrecords')
]