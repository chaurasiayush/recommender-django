
class CountryData(models.Model):
    cid = models.AutoField(primary_key=True)
    cname = models.TextField()
    img_url = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'country_data'




class MatchScorecard(models.Model):
    match_code = models.AutoField(blank=True, null=True)
    pid = models.IntegerField(blank=True, null=True)
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
        managed = False
        db_table = 'match_scorecard'


class MatchesInSeries(models.Model):
    series_code = models.AutoField(blank=True, null=True)
    match_code = models.IntegerField(blank=True, null=True)
    date = models.TextField(blank=True, null=True)
    result = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'matches_in_series'


class PlayerRi(models.Model):
    sr = models.AutoField(blank=True, null=True)
    series_code = models.TextField(blank=True, null=True)  # This field type is a guess.
    pid = models.FloatField(blank=True, null=True)
    batagr = models.FloatField(db_column='batAGR', blank=True, null=True)  # Field name made lowercase.
    bowlagr = models.FloatField(db_column='bowlAGR', blank=True, null=True)  # Field name made lowercase.
    batri = models.FloatField(db_column='batRI', blank=True, null=True)  # Field name made lowercase.
    bowlri = models.FloatField(db_column='bowlRI', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'player_RI'


class PlayerData(models.Model):
    pid = models.AutoField(primary_key=True)
    name = models.TextField()
    yob = models.IntegerField()
    cid = models.IntegerField()
    iscurrent = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'player_data'


class PlayerRankingindexes(models.Model):
    series_code = models.IntegerField()
    pid = models.IntegerField()
    batagr = models.IntegerField(db_column='batAGR', blank=True, null=True)  # Field name made lowercase.
    bowlagr = models.IntegerField(db_column='bowlAGR', blank=True, null=True)  # Field name made lowercase.
    batri = models.IntegerField(db_column='batRI', blank=True, null=True)  # Field name made lowercase.
    bowlri = models.IntegerField(db_column='bowlRI', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'player_rankingindexes'


class SeriesRecord(models.Model):
    series_code = models.AutoField()
    cnt1 = models.IntegerField(blank=True, null=True)
    cnt2 = models.IntegerField(blank=True, null=True)
    matches = models.IntegerField(blank=True, null=True)
    cnt1_wins = models.IntegerField(blank=True, null=True)
    cnt2_wins = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'series_record'
