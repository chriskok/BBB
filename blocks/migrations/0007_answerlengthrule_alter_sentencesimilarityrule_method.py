# Generated by Django 4.1.7 on 2023-03-23 15:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('blocks', '0006_sentencesimilarityrule_method'),
    ]

    operations = [
        migrations.CreateModel(
            name='AnswerLengthRule',
            fields=[
                ('rule_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='blocks.rule')),
                ('length_type', models.CharField(blank=True, default='word', max_length=200, null=True)),
                ('length', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('blocks.rule',),
        ),
        migrations.AlterField(
            model_name='sentencesimilarityrule',
            name='method',
            field=models.CharField(blank=True, default='sbert', max_length=200, null=True),
        ),
    ]