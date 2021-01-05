#
# import os
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommender.app_players')
# import django
# django.setup()

# from lib.prsystem import *
# import pandas as pd
from recommender.app_players.models import OdiPlayersInfo, OdiPlayersRankingIndexes, OdiClusters, StatePlayerInfo, StatePlayersRankingIndexes, StateClusters
# from app_home import views



# print('CALCULATING INDEXES.....')
# odiplayers = pd.read_csv("serieswiseMatchRecord.csv")
# ipinfov = pd.read_csv('playerList.csv')
#
# spinfov = pd.read_csv('newPlayerList.csv')
# stateplayers = pd.read_csv("newPlayers.csv")
#
# odiind = clusterOdiPlayers(odiplayers, '')
# stind = clusterStatePlayers(stateplayers, odiind)
#
# allplayers = buildFinalDataset(odiind, ipinfov, stind, spinfov, selMethod='overall', numBats=5, numBowls=5)
# # allplayers.head()
#
# ranking = selectPlayers(allplayers, 4, nbats=5, nbowls=5, nall=5)
#
# print(ranking)
allplayers = OdiPlayersInfo.objects.all()

print(allplayers)
