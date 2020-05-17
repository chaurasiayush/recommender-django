from django.db import models

# Create your models here.
class PlayerData(models.Model):
    pid = models.AutoField(primary_key=True)
    name = models.TextField()
    yob = models.IntegerField()
    cid = models.IntegerField()
    iscurrent = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'player_data'


class OverallPlayerRecord(models.Model):
    pid = models.IntegerField(primary_key=True)
    name = models.TextField(blank=True, null=True)  # This field type is a guess.
    series_played = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    matches_played = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    noi = models.IntegerField(db_column='NOI', blank=True, null=True)  # Field name made lowercase. This field type is a guess.
    total_runs = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_ballfaced = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_fours = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_sixes = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_oers = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_maiden_overs = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_runs_given = models.IntegerField(blank=True, null=True)  # This field type is a guess.
    total_wickets_taken = models.IntegerField(blank=True, null=True)  # This field type is a guess.

    def __str__(self):
        return self.name

    class Meta:
        managed = False  # Created from a view. Don't remove.
        db_table = 'overall_player_record'
