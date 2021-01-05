from django.db import models

# ODI players models

class OdiPlayersInfo(models.Model):
    pid = models.IntegerField(primary_key=True)
    name = models.TextField()
    yob = models.IntegerField()
    cid = models.IntegerField()
    iscurrent = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'ODI_player_info'


class PlayersWKeepRecords(models.Model):
    pid = models.IntegerField(unique=True, null=False)
    iswk = models.IntegerField(default=0)
    match_played = models.IntegerField(default=0)
    catches = models.IntegerField(default=0)
    stumpings = models.IntegerField(default=0)
    total_dismissals = models.IntegerField(default=0)

    def __str__(self):
        return str(self.pid)

    class Meta:
        db_table = 'players_wicket_keeping_records'



class OverallPlayerRecord(models.Model):
    pcid = models.IntegerField()
    iscurrent = models.IntegerField(null=True)
    cname = models.TextField()
    pid = models.IntegerField(primary_key=True)
    name = models.TextField(blank=True, null=True)  # This field type is a guess.
    series_played = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    matches_played = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    noi = models.IntegerField(db_column='NOI', blank=True, null=True)  # Field name made lowercase. This field type is a guess.
    total_runs = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_ballfaced = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_fours = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_sixes = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_overs = models.FloatField(blank=True, null=True)  # This field type is a guess.
    total_maiden_overs = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_runs_given = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_wickets_taken = models.IntegerField(blank=True, null=True)  # This field type is a guess.

    def __str__(self):
        return self.name

    class Meta:
        managed = False  # Created from a view. Don't remove.
        db_table = 'ODI_overall_players_record'


class OdiPlayersRankingIndexes(models.Model):
    pid = models.IntegerField(primary_key=True)
    batri = models.FloatField()
    bowlri = models.FloatField()

    def __str__(self):
        return self.pid

    class Meta:
        managed = False
        db_table = 'ODI_players_ranking_indexes'


class OdiClusters(models.Model):
    pid = models.IntegerField(primary_key=True)
    clusterRI = models.IntegerField()
    series_played = models.IntegerField()
    batRI = models.FloatField(db_column='batRI')
    bowlRI = models.FloatField(db_column='bowlRI')
    wkRI = models.FloatField(db_column='wkRI')
    def __str__(self):
        return str(self.pid)

    class Meta:
        managed = True
        db_table = 'ODI_clusters'


# models for state level players

class StatePlayerInfo(models.Model):
    pid = models.IntegerField(primary_key=True)
    name = models.TextField()
    yob = models.IntegerField()
    team = models.TextField()
    iscurrent = models.IntegerField()

    def __str__(self):
        return self.name

    class Meta:
        managed = False
        db_table = 'state_player_info'


class StatePlayersRankingIndexes(models.Model):
    pid = models.IntegerField(primary_key=True)
    batri = models.FloatField()
    bowlri = models.FloatField()

    def __str__(self):
        return self.pid

    class Meta:
        managed = False
        db_table = 'state_players_ranking_indexes'


class StateClusters(models.Model):
    pid = models.IntegerField(primary_key=True)
    clusterRI = models.IntegerField()
    series_played = models.IntegerField()
    batRI = models.FloatField(db_column='batRI')
    bowlRI = models.FloatField(db_column='bowlRI')
    wkRI = models.FloatField(db_column='wkRI')

    def __str__(self):
        return str(self.pid)

    class Meta:
        managed = True
        db_table = 'state_clusters'
