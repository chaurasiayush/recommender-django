from django.urls import path
from app_aiengine import views

app_name = 'aiengine'

urlpatterns = [
    path('clustering', views.makeClusters, name = 'clustering'),
    path('recommend_players', views.recommendTeamPlayers, name='recommend_players'),
    path('selection_form', views.selectionCriteriaForm, name='selection_form'),


]
