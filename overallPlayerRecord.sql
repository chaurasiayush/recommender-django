# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class OverallPlayerRecord(models.Model):
    pid = models.IntegerField(blank=True, null=True)
    name = models.TextField(blank=True, null=True)  # This field type is a guess.
    series_played = models.TextField(blank=True, null=True)  # This field type is a guess.
    matches_played = models.TextField(blank=True, null=True)  # This field type is a guess.
    noi = models.TextField(db_column='NOI', blank=True, null=True)  # Field name made lowercase. This field type is a guess.
    total_runs = models.TextField(blank=True, null=True)  # This field type is a guess.
    total_ballfaced = models.TextField(blank=True, null=True)  # This field type is a guess.
    total_fours = models.TextField(blank=True, null=True)  # This field type is a guess.
    total_sixes = models.TextField(blank=True, null=True)  # This field type is a guess.
    total_oers = models.TextField(blank=True, null=True)  # This field type is a guess.
    total_maiden_overs = models.TextField(blank=True, null=True)  # This field type is a guess.
    total_runs_given = models.TextField(blank=True, null=True)  # This field type is a guess.
    total_wickets_taken = models.TextField(blank=True, null=True)  # This field type is a guess.

    class Meta:
        managed = False  # Created from a view. Don't remove.
        db_table = 'overall_player_record'
