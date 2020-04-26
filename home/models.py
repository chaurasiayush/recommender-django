from django.db import models

# Create your models here.
class Team:
    name : str
    numberOfPlayers : int
    pic : str

    def team(self, name, numberOfPlayers ):
        self.name = name
        self.numberOfPlayers = numberOfPlayers