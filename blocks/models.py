from django.db import models
from django.urls import reverse
import json
from polymorphic.models import PolymorphicModel

class Question(models.Model):
    question_exam_id = models.CharField(max_length = 200, blank=True, null=True)
    question_text = models.TextField(blank=True, null=True)
    question_keywords = models.TextField(blank=True, null=True)
    question_image_url = models.TextField(blank=True, null=True)  # right now, using static/images to hold media files\
    related_concepts = models.TextField(default="", blank=True, null=True)

    class Meta:
        ordering = ['question_exam_id']

    def __str__(self):
        return "{}. {}".format(str(self.id), self.question_text)
    
class Rule(PolymorphicModel):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, default=None, null=True, blank=True)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, default=None, null=True, blank=True)
    polarity = models.CharField(max_length=200, default="positive", null=True, blank=True)
    positive_examples = models.CharField(max_length=1000, default="[]")
    negative_examples = models.CharField(max_length=1000, default="[]")

    def set_positive_examples(self, x):
        self.positive_examples = json.dumps(x)

    def get_positive_examples(self):
        return json.loads(self.positive_examples)

    def set_negative_examples(self, x):
        self.negative_examples = json.dumps(x)

    def get_negative_examples(self):
        return json.loads(self.negative_examples)

def get_polarity_emoji(polarity):
    return "✔️" if polarity == "positive" else "❌"

class KeywordRule(Rule):
    keyword = models.CharField(max_length=200)
    similarity_threshold = models.FloatField(default=1.0, null=True, blank=True)
    relevant_keywords = models.CharField(max_length=1000, default="[]")  \

    # https://stackoverflow.com/questions/22340258/list-field-in-model
    def set_relevant_keywords(self, x):
        self.relevant_keywords = json.dumps(x)

    def get_relevant_keywords(self):
        return json.loads(self.relevant_keywords)

    def __str__(self):
        return "{} Keyword: {}".format(get_polarity_emoji(self.polarity), self.keyword)

class SentenceSimilarityRule(Rule):
    sentence = models.CharField(max_length=2048)
    similarity_threshold = models.FloatField(default=0.8, null=True, blank=True)
    method = models.CharField(max_length=200, default="sbert", null=True, blank=True)

    def __str__(self):
        return "{} Sentence: {:.20s}...".format(get_polarity_emoji(self.polarity), self.sentence) if len(self.sentence) > 20 else "{} Sentence: {}".format(get_polarity_emoji(self.polarity), self.sentence)

class AnswerLengthRule(Rule):
    length_type = models.CharField(max_length=200, default="word", null=True, blank=True)
    length = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return "{} Answer Length: {} {}s".format(get_polarity_emoji(self.polarity), self.length, self.length_type)
    
class Answer(models.Model):
    answer_text = models.TextField()
    student_id = models.IntegerField()
    assigned_grade = models.FloatField(default=0.0)

    # https://docs.djangoproject.com/en/4.0/ref/models/fields/#foreignkey
    question = models.ForeignKey(Question, on_delete=models.CASCADE, default=None, null=True, blank=True)
    applied_rules = models.ManyToManyField(Rule, default=None)

    rule_strings = models.CharField(max_length=1000, default="[]")  
    outlier_score = models.FloatField(default=0.0)

    # https://stackoverflow.com/questions/22340258/list-field-in-model
    def set_rule_strings(self, x):
        self.rule_strings = json.dumps(x)

    def get_rule_strings(self):
        return json.loads(self.rule_strings)

    def __str__(self):
        return "{}. Q #{}: {}".format(self.id, self.question.id, self.answer_text)
    
class ChatGPTGradeAndFeedback(models.Model):
    answer = models.ForeignKey(Answer, on_delete=models.CASCADE, default=None, null=True, blank=True)
    response = models.TextField()
    prompt = models.TextField()
    prompt_type = models.CharField(max_length=200)
    trial_run_number = models.IntegerField()
    openai_model = models.CharField(max_length=200, default="", null=True, blank=True)

    def __str__(self):
        return "{}. Trial: {}, Prompt: {}, Model: {}".format(self.id, self.trial_run_number, self.prompt_type, self.openai_model)

