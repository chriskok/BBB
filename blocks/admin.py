from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(Question)
admin.site.register(Answer)
admin.site.register(Cluster)
admin.site.register(Rule)
admin.site.register(KeywordRule)
admin.site.register(SentenceSimilarityRule)
admin.site.register(ChatGPTGradeAndFeedback)
admin.site.register(Rubric)
admin.site.register(RubricList)
admin.site.register(AnswerTag)
admin.site.register(AnswerFeedback)