from django.shortcuts import render
from .models import PlayerData, OverallPlayerRecord
# Create your views here.

def overallPRec(request):
    allplayers = OverallPlayerRecord.objects.all().order_by('pid')
    print(allplayers)

    inject = {
        'players' : allplayers
    }

    return render(request, 'players/allPlayerRecord.html', context=inject)