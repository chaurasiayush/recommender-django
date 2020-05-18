from django.db import models

# Create your models here.

class CountryData(models.Model):
    cid = models.AutoField(primary_key=True)
    cname = models.TextField()
    img_url = models.ImageField(upload_to= 'teams', default='teams/India.jpg')

    def __str__(self):
        return self.cname

    class Meta:
        managed = False
        db_table = 'country_data'