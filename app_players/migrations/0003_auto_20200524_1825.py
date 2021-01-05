# Generated by Django 3.0.6 on 2020-05-24 12:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app_players', '0002_overallplayerrecord'),
    ]

    operations = [
        migrations.CreateModel(
            name='OdiPlayersInfo',
            fields=[
                ('pid', models.IntegerField(primary_key=True, serialize=False)),
                ('name', models.TextField()),
                ('yob', models.IntegerField()),
                ('cid', models.IntegerField()),
                ('iscurrent', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'ODI_player_info',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='OdiPlayersRankingIndexes',
            fields=[
                ('pid', models.IntegerField(primary_key=True, serialize=False)),
                ('batri', models.FloatField()),
                ('bowlri', models.FloatField()),
            ],
            options={
                'db_table': 'ODI_players_ranking_indexes',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='StatePlayerInfo',
            fields=[
                ('pid', models.IntegerField(primary_key=True, serialize=False)),
                ('name', models.TextField()),
                ('yob', models.IntegerField()),
                ('team', models.TextField()),
                ('iscurrent', models.IntegerField()),
            ],
            options={
                'db_table': 'state_player_info',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='StatePlayersRankingIndexes',
            fields=[
                ('pid', models.IntegerField(primary_key=True, serialize=False)),
                ('batri', models.FloatField()),
                ('bowlri', models.FloatField()),
            ],
            options={
                'db_table': 'state_players_ranking_indexes',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='OdiClusters',
            fields=[
                ('pid', models.IntegerField(primary_key=True, serialize=False)),
                ('clusterRI', models.IntegerField()),
                ('series_played', models.IntegerField()),
                ('batRI', models.FloatField(db_column='batRI')),
                ('bowlRI', models.FloatField(db_column='bowlRI')),
                ('wkRI', models.FloatField(db_column='wkRI')),
            ],
            options={
                'db_table': 'ODI_clusters',
                'managed': True,
            },
        ),
        migrations.CreateModel(
            name='StateClusters',
            fields=[
                ('pid', models.IntegerField(primary_key=True, serialize=False)),
                ('clusterRI', models.IntegerField()),
                ('series_played', models.IntegerField()),
                ('batRI', models.FloatField(db_column='batRI')),
                ('bowlRI', models.FloatField(db_column='bowlRI')),
                ('wkRI', models.FloatField(db_column='wkRI')),
            ],
            options={
                'db_table': 'state_clusters',
                'managed': True,
            },
        ),
        migrations.DeleteModel(
            name='PlayerData',
        ),
        migrations.AlterModelTable(
            name='overallplayerrecord',
            table='ODI_overall_players_record',
        ),
    ]
