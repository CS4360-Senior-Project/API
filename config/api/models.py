from django.db import models

# Create your models here.

class newTable(models.Model):
    image_type = models.CharField(max_length=10)


# class preProcessedImages(models.Model):
#     imageID = models.AutoField(primary_key=True)
#     image_name = models.CharField(max_length=500)
#     added_date = models.DateField()
#     image_type = models.ForeignKey(newTable, on_delete=models.CASCADE, related_name='newtable', null=True)
