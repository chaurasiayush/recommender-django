from django.db import connection
from django.shortcuts import render
from django.http import HttpResponse
from .models import *
from app_players.models import OdiPlayersInfo
# Create your views here.

curr = connection.cursor()
teamsQuery = '''select cname, count(pid) as players, img_url from
                    player_data natural join country_data where iscurrent == 1 group by cid'''

def home(request):

    curr.execute(teamsQuery)
    teams = curr.fetchall()

    print(teams)
    teams = CountryData.objects.all()

    return render(request, 'home/index.html', {'teams': teams})
