# Generated by Django 3.0.5 on 2020-05-16 14:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app_players', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='OverallPlayerRecord',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pid', models.IntegerField(blank=True, null=True)),
                ('name', models.TextField(blank=True, null=True)),
                ('series_played', models.IntegerField(blank=True, null=True)),
                ('matches_played', models.IntegerField(blank=True, null=True)),
                ('noi', models.IntegerField(blank=True, db_column='NOI', null=True)),
                ('total_runs', models.IntegerField(blank=True, null=True)),
                ('total_ballfaced', models.IntegerField(blank=True, null=True)),
                ('total_fours', models.IntegerField(blank=True, null=True)),
                ('total_sixes', models.IntegerField(blank=True, null=True)),
                ('total_oers', models.IntegerField(blank=True, null=True)),
                ('total_maiden_overs', models.IntegerField(blank=True, null=True)),
                ('total_runs_given', models.IntegerField(blank=True, null=True)),
                ('total_wickets_taken', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'overall_player_record',
                'managed': False,
            },
        ),
    ]