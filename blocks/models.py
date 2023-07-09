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

class ConceptSimilarityRule(Rule):
    concept = models.CharField(max_length=2048)
    similarity_threshold = models.FloatField(default=0.8, null=True, blank=True)

    def __str__(self):
        return "{} Concept: {:.20s}...".format(get_polarity_emoji(self.polarity), self.concept) if len(self.concept) > 20 else "{} Concept: {}".format(get_polarity_emoji(self.polarity), self.concept)

class AnswerLengthRule(Rule):
    length_type = models.CharField(max_length=200, default="word", null=True, blank=True)
    length = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return "{} Answer Length: {} {}s".format(get_polarity_emoji(self.polarity), self.length, self.length_type)
    
class Cluster(models.Model):
    question = models.ForeignKey(Question, on_delete=models.SET_NULL, default=None, null=True, blank=True)
    cluster_name = models.CharField(max_length = 200, blank=True, null=True)
    cluster_id = models.IntegerField(default=0)

    applied_rules = models.ManyToManyField(Rule, default=None)
    cluster_description = models.TextField(blank=True, null=True)
    cluster_grade = models.FloatField(blank=True, null=True)
    cluster_feedback = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['question__id', 'cluster_id']

class Answer(models.Model):
    answer_text = models.TextField()
    student_id = models.IntegerField()
    assigned_grade = models.FloatField(default=0.0)

    # https://docs.djangoproject.com/en/4.0/ref/models/fields/#foreignkey
    question = models.ForeignKey(Question, on_delete=models.CASCADE, default=None, null=True, blank=True)
    cluster = models.ForeignKey(Cluster, on_delete=models.SET_NULL, default=None, null=True, blank=True)
    applied_rules = models.ManyToManyField(Rule, default=None)

    # answer model features if user edits/inputs individually
    override_grade = models.FloatField(blank=True, null=True)
    override_feedback =  models.TextField(blank=True, null=True)

    rule_strings = models.CharField(max_length=1000, default="[]")  
    outlier_score = models.FloatField(default=0.0)
    concept_scores = models.CharField(max_length=1000, default="[]")  

    # https://stackoverflow.com/questions/22340258/list-field-in-model
    def set_rule_strings(self, x):
        self.rule_strings = json.dumps(x)

    def get_rule_strings(self):
        return json.loads(self.rule_strings)
    
    def set_concept_scores(self, x):
        self.concept_scores = json.dumps(x)

    def get_concept_scores(self):
        return json.loads(self.concept_scores)

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

class Rubric(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, default=None, null=True, blank=True)
    rubric_dict = models.TextField()
    message_history = models.TextField()
    
    def set_rubric_dict(self, x):
        self.rubric_dict = json.dumps(x)

    def get_rubric_dict(self):
        return json.loads(self.rubric_dict)
    
    def set_message_history(self, x):
        self.message_history = json.dumps(x)

    def get_message_history(self):
        return json.loads(self.message_history)

    def __str__(self):
        return "{}. Question: {}".format(self.id, self.question.question_text)
    
class RubricList(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, default=None, null=True, blank=True)
    # format: [{'id': <rubric_id>, 'polarity': <positive or negative>, 'title': <name of rubric>, 'description': <optional: explanation of rubric>, "reasoning_dict": {}}, ...]
    rubric_list = models.TextField()
    
    def set_rubric_list(self, x):
        self.rubric_list = json.dumps(x)

    def get_rubric_list(self):
        return json.loads(self.rubric_list)
    
    def __str__(self):
        return "{}. Question: {}".format(self.id, self.question.question_text)
    
class AnswerTag(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, default=None, null=True, blank=True)
    answer = models.ForeignKey(Answer, on_delete=models.CASCADE, default=None, null=True, blank=True)
    tag = models.CharField(max_length=200, default="", null=True, blank=True)
    reasoning_dict = models.TextField()  # dictionary with highlighted section and reasoning for chosen tags
    feedback = models.TextField()
    
    def set_reasoning_dict(self, x):
        self.reasoning_dict = json.dumps(x)

    def get_reasoning_dict(self):
        return json.loads(self.reasoning_dict)

    def __str__(self):
        return "{}. Answer: {}, Tag: {}".format(self.id, self.answer.id, self.tag)
    

    class Meta:
        ordering = ['question__id', 'answer__id', 'tag']