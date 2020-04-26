from django.shortcuts import render
from django.http import HttpResponse
from.models import Team
# Create your views here.
def home(request):

    team1 = Team()
    team1.name = "INDIA"
    team1.numberOfPlayers = 21
    team1.pic = 'location1.png'

    team2 = Team()
    team2.name = "PAKISTAN"
    team2.numberOfPlayers = 16
    team2.pic = 'location2.png'

    team3 = Team()
    team3.name = "ENGLAND"
    team3.numberOfPlayers = 19
    team3.pic = 'location3.png'

    teams = [team1, team2, team3]

    return render(request, 'index.html', {'teams': teams})