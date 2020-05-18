from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def selectPlayers(request):
    return HttpResponse("<h1> SELECT PLAYER IS WORKING </h1>")