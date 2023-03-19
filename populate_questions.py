from blocks.models import Answer, Question

import pandas as pd

# ================================== #
#            UI FINAL EXAM           #
# ================================== #

question_text_df = pd.read_excel('data/ui_final/final_B_questions.xlsx', header=0, engine='openpyxl')

df_records = question_text_df.to_dict('records')

for record in df_records:
    obj, created = Question.objects.get_or_create(
        question_text = record['question_text'],
        question_exam_id = record['question_id'],
        question_keywords = "",
    )
    if(created): print('Question: {} created'.format(record['question_id']))
    else: print('Question: {} already exists'.format(record['question_id']))

