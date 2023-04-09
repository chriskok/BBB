import pandas as pd
from blocks.models import *

df = pd.DataFrame(list(ChatGPTGradeAndFeedback.objects.all().values('answer__question__question_text', 'answer__answer_text', 'answer__student_id', 'response', 'prompt_type', 'openai_model').order_by('answer__question__question_text', 'answer__student_id')))

df.to_csv("comb_chatgpt.csv") 
