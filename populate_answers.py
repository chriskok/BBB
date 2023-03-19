from blocks.models import Answer, Question

import pandas as pd
import re

# ================================== #
#        UI FINAL EXAM - GRADED      #
# ================================== #

df = pd.read_excel('data/ui_final/final_B_merged.xlsx', header=0, engine='openpyxl')

# indexed by "answer text col", "question ID", "score col"
answer_cols = [
    ("Q9 (i)", 9.1, "9: Pick2 (2.0 pts)"), 
    ("Q9 (ii)", 9.2, "9: Pick2 (2.0 pts)"), 
    ("Q10 b(ii)", 10.0, "10.3: OE (1.0 pts)"), 
    ("Q11", 11.0, "11 (2.0 pts)"), 
    ("Q13 a)", 13.1, "13.1: OE (2.0 pts)"), 
    ("Q13 b)", 13.2, "13.2: OE (2.0 pts)"), 
    ("Q13 c)", 13.3, "13.3: OE (2.0 pts)"), 
    ("Q16", 16.0, "16: OE (2.0 pts)"), 
    ("Q17", 17.0, "17: OE (2.0 pts)"), 
    ("Q19 U=Upside, D=downside", 19.0, "19: OE (2.0 pts)"), 
    ("Q21", 21.0, "21: OE (2.0 pts)"), 
    ("Q23", 23.0, "23: OE (2.0 pts)"),
    ("Q25 i, ii", 25.0, "25 (2.0 pts)"), 
    ("Q26 i, ii (H is omitted)", 26.0, "26.2 (2.0 pts)"), 
    ("Q28 b) (B is block-based, T is text based)", 28.0, "28.2 (2.0 pts)"), 
    ("Q29 i)", 29.1, "29.1 (2.0 pts)"), 
    ("Q29 ii)", 29.2, "29.2 (2.0 pts)"), 
]

for index, row in df.iterrows():
    for answer_pair in answer_cols:
        obj, created = Answer.objects.get_or_create(
            answer_text = row[answer_pair[0]],
            student_id = index,
            question = Question.objects.get(question_exam_id=str(answer_pair[1])),
            assigned_grade = row[answer_pair[2]]
        )
        if (created): print('Answer: {}-{} created'.format(answer_pair[1], index))

