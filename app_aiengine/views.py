from django.shortcuts import render, redirect
from django.http import HttpResponse
from app_home.models import CountryData
from app_players.models import OdiPlayersInfo, OdiClusters, StatePlayerInfo, StateClusters
from app_archive.models import OdiMatchScorecardSerieswise, StateMatchScorecardSerieswise
from app_aiengine.lib.prsystem import *
import pandas as pd

def makeClusters(request):

    qs2 = OdiMatchScorecardSerieswise.objects.all()
    rec2 = qs2.values()
    odimatchserieswise = pd.DataFrame.from_records(rec2)
    # html2 = odimatchserieswise.head(10).to_html()

    qs4 = StateMatchScorecardSerieswise.objects.all()
    rec4 = qs4.values()
    statematchserieswise = pd.DataFrame.from_records(rec4)
    # html4 = statematchserieswise.head(10).to_html()

    odiindexes = clusterOdiPlayers(odimatchserieswise[:], saveLocation="media/aiengine/")
    odiindexes.to_csv('odi.csv', index=False)
    html5 = odiindexes.head(20).to_html()

    OdiClusters.objects.all().delete()
    odiindexes_records = odiindexes.to_dict('records')
    model_instances = [OdiClusters(
        pid=record['pid'],
        series_played=record['series_played'],
        batri=record['batRI'],
        bowlri=record['bowlRI'],
        cluster=record['clusterRI']
    ) for record in odiindexes_records]
    OdiClusters.objects.bulk_create(model_instances)

    stateindexes = clusterStatePlayers(statematchserieswise, odiindexes)
    StateClusters.objects.all().delete()
    stateindexes_records = stateindexes.to_dict('records')
    model_instances = [StateClusters(
        pid=record['pid'],
        series_played=record['series_played'],
        batri=record['batRI'],
        bowlri=record['bowlRI'],
        cluster=record['clusterRI']
    ) for record in stateindexes_records]
    StateClusters.objects.bulk_create(model_instances)

    args = {
        'dflist' : [html5]
    }

    # print(df)
    # return HttpResponse(olst)
    return render(request, 'aiengine/dfprint.html', args)


def recommendTeamPlayers(request):
    #getting request attributes

    forteam = int(request.POST['team1'])
    vteam = int(request.POST['team2'])
    obats = int(request.POST['obats'])
    obowls = int(request.POST['obowls'])
    oall = int(request.POST['oall'])
    selmethod = str(request.POST['state-sel-method'])
    sbats = int(request.POST['sbats'])
    sbowls = int(request.POST['sbowls'])

    if forteam > 0 and vteam > 0 and forteam!= vteam and selmethod != 'dontselect':

        # fetching odi players data
        qs1 = OdiPlayersInfo.objects.all()
        rec1 = qs1.values()
        odiplayersinfo = pd.DataFrame.from_records(rec1)
        # html1 = odiplayersinfo.head(10).to_html()

        qs2 = OdiClusters.objects.all()
        rec2 = qs2.values()
        odiclusters = pd.DataFrame.from_records(rec2)

        # fetching state players data
        qs3 = StateClusters.objects.all()
        rec3 = qs3.values()
        stateclusters = pd.DataFrame.from_records(rec3)

        qs4 = StatePlayerInfo.objects.all()
        rec4 = qs4.values()
        stateplayersinfo = pd.DataFrame.from_records(rec4)
        html7 = stateplayersinfo.head(15).to_html()

        allplayers = buildFinalDataset(odiclusters, odiplayersinfo,
                                        stateclusters, stateplayersinfo,
                                        selMethod=selmethod, numBats=sbats, numBowls=sbowls)
        # html7 = allplayers.head(15).to_html()
        #
        ranking = selectPlayers(allplayers, forteam, nbats=obats, nbowls=obowls, nall=oall)

        print(forteam, vteam, obats, obowls, oall, selmethod, sbats, sbowls, end='\n')

        title =  ['Batting', 'Bowling']
        ranks = zip(ranking, title)
        args = {

            'ranks' : ranks
        }

        return render(request, 'aiengine/selection_result.html', args)

    else:
        return redirect('aiengine:selection_form')

def selectionCriteriaForm(request):

    teamlist = CountryData.objects.all()

    attr = {
        'teams' : teamlist
    }

    return render(request, 'aiengine/selection_form.html', attr)
