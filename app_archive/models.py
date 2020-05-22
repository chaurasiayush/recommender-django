from django.db import models

class OdiSeriesRecord(models.Model):
    series_code = models.IntegerField(primary_key=True)
    cnt1 = models.IntegerField(blank=False, null=False)
    cnt2 = models.IntegerField(blank=False, null=False)
    matches = models.IntegerField(blank=False, null=False)
    cnt1_wins = models.IntegerField(blank=False, null=False)
    cnt2_wins = models.IntegerField(blank=False, null=False)


    def __str__(self):
        return "series_code: " + str(self.series_code)

    class Meta:
        managed = False
        db_table = 'ODI_series_record'


class OdiMatchScorecard(models.Model):
    id = models.IntegerField(primary_key=True)
    match_code = models.IntegerField()
    # match_code = models.ForeignKey(OdiSeriesRecord, on_delete=models.CASCADE)
    pid = models.IntegerField(blank=False, null=False)
    notout = models.IntegerField(blank=True, null=True)
    runs = models.IntegerField(blank=False, null=False)
    ballfaced = models.IntegerField(blank=False, null=False)
    fours = models.IntegerField(blank=False, null=False)
    sixes = models.IntegerField(blank=False, null=False)
    overs = models.FloatField(blank=False, null=False)
    maiden_overs = models.IntegerField(blank=False, null=False)
    runs_given = models.IntegerField(blank=False, null=False)
    wickets_taken = models.IntegerField(blank=False, null=False)


    def __str__(self):
        return "Match code: " + str(self.match_code) + "    Pid: " + str(self.pid)


    class Meta:
        managed = False
        db_table = 'ODI_match_scorecard'
        unique_together = (('match_code', 'pid'),)


class OdiMatchScorecardSerieswise(models.Model):
    series_code = models.IntegerField(primary_key=True)
    match_code = models.IntegerField(blank=True, null=True)
    pcid = models.IntegerField(blank=True, null=True)
    iscurrent = models.IntegerField(blank=True, null=True)
    cname = models.TextField(blank=True, null=True)
    pid = models.IntegerField(blank=True, null=True)
    name = models.TextField(blank=True, null=True)  # This field type is a guess.
    notout = models.IntegerField(blank=True, null=True)
    runs = models.IntegerField(blank=True, null=True)
    ballfaced = models.IntegerField(blank=True, null=True)
    fours = models.IntegerField(blank=True, null=True)
    sixes = models.IntegerField(blank=True, null=True)
    overs = models.IntegerField(blank=True, null=True)
    maiden_overs = models.IntegerField(blank=True, null=True)
    runs_given = models.IntegerField(blank=True, null=True)
    wickets_taken = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False  # Created from a view. Don't remove.
        db_table = 'ODI_match_scorecard_serieswise'

# state level matches


class StateMatchScorecard(models.Model):
    # id = models.IntegerField(primary_key=True)
    match_code = models.IntegerField(blank=False, null=False)
    pid = models.IntegerField(blank=False, null=False)
    outby = models.TextField(blank=False, null=False)
    notout = models.IntegerField(blank=False, null=False)
    runs = models.IntegerField(blank=False, null=False)
    ballfaced = models.IntegerField(blank=False, null=False)
    fours = models.IntegerField(blank=False, null=False)
    sixes = models.IntegerField(blank=False, null=False)
    strikerate = models.FloatField(blank=False, null=False)
    overs = models.FloatField(blank=False, null=False)
    maiden_overs = models.IntegerField(blank=False, null=False)
    runs_given = models.IntegerField(blank=False, null=False)
    wickets_taken = models.IntegerField(blank=False, null=False)
    noballs = models.IntegerField(blank=False, null=False)
    wideballs = models.IntegerField(blank=False, null=False)


    def __str__(self):
        return 'scorecard match code: ' + str(self.match_code)


    class Meta:
        managed = False
        db_table = 'state_match_scorecard'
        unique_together = (('match_code', 'pid'),)


class StateSeriesRecord(models.Model):
    series_code = models.IntegerField(primary_key=True)
    year = models.IntegerField(blank=False, null=False)
    url = models.TextField(blank=False, null=False)


    def __str__(self):
        return 'series code: ' + str(self.series_code)

    class Meta:
        managed = False
        db_table = 'state_series_record'


class StateMatchScorecardSerieswise(models.Model):
    series_code = models.IntegerField(primary_key=True)
    match_code = models.IntegerField(blank=True, null=True)
    team = models.TextField(blank=True, null=True)
    pid = models.IntegerField(blank=True, null=True)
    name = models.TextField(blank=True, null=True)  # This field type is a guess.
    notout = models.IntegerField(blank=True, null=True)
    runs = models.IntegerField(blank=True, null=True)
    ballfaced = models.IntegerField(blank=True, null=True)
    fours = models.IntegerField(blank=True, null=True)
    sixes = models.IntegerField(blank=True, null=True)
    overs = models.FloatField(blank=True, null=True)
    maiden_overs = models.IntegerField(blank=True, null=True)
    runs_given = models.IntegerField(blank=True, null=True)
    wickets_taken = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False  # Created from a view. Don't remove.
        db_table = 'state_match_scorecard_serieswise'
