from django.shortcuts import render
from .models import OdiPlayersInfo, OverallPlayerRecord
# Create your views here.

def overallPRec(request):
    allplayers = OverallPlayerRecord.objects.all().order_by('pid')
    print(allplayers)

    inject = {
        'teamval' : "All",
        'players' : allplayers
    }

    return render(request, 'players/PlayersRecord.html', context=inject)

def teamPlayers(request):
    if request.method == "POST":
        cid = str(request.POST['cid'])

        print("getting from form: ", cid)
        # teamplayers = OverallPlayerRecord.objects.raw('select * from overall_player_record where pcid == '+ cid + ' order by pid')

        teamplayers = OverallPlayerRecord.objects.filter(pcid__exact=cid, iscurrent__exact=1).order_by('name')

        inject = {
            'teamval': teamplayers[0].cname + " Team",
            'players': teamplayers
        }

        return render(request, 'players/PlayersRecord.html', context=inject)
