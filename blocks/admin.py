from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(Question)
admin.site.register(Answer)
admin.site.register(Rule)
admin.site.register(KeywordRule)
admin.site.register(SentenceSimilarityRule)
admin.site.register(ChatGPTGradeAndFeedback)