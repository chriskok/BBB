# Generated by Django 4.1.7 on 2023-03-23 14:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blocks', '0005_sentencesimilarityrule'),
    ]

    operations = [
        migrations.AddField(
            model_name='sentencesimilarityrule',
            name='method',
            field=models.CharField(blank=True, default='SBert', max_length=200, null=True),
        ),
    ]