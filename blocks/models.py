from django.db import models
from django.urls import reverse
import json
from polymorphic.models import PolymorphicModel

class Question(models.Model):
    question_exam_id = models.CharField(max_length = 200, blank=True, null=True)
    question_text = models.TextField(blank=True, null=True)
    question_keywords = models.TextField(blank=True, null=True)
    question_image_url = models.TextField(blank=True, null=True)  # right now, using static/images to hold media files
    
    class Meta:
        ordering = ['question_exam_id']

    def __str__(self):
        return "{}. {}".format(str(self.id), self.question_text)

class Answer(models.Model):
    answer_text = models.TextField()
    student_id = models.IntegerField()
    assigned_grade = models.FloatField(default=0.0)

    # https://docs.djangoproject.com/en/4.0/ref/models/fields/#foreignkey
    question = models.ForeignKey(Question, on_delete=models.CASCADE, default=None, null=True, blank=True)

    def __str__(self):
        return "Q #{}: {}".format(self.question.id, self.answer_text)

class Rule(PolymorphicModel):
    question = models.ForeignKey(Question, on_delete=models.SET_NULL, default=None, null=True, blank=True)
    applied_answers = models.ManyToManyField(Answer)

class KeywordRule(Rule):
    keyword = models.CharField(max_length=200)

    def __str__(self):
        return "Keyword: {}".format(self.keyword)