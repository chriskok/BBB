from blocks.models import Answer, Question

import pandas as pd
import re
import json

# # ================================== #
# #        UI FINAL EXAM - GRADED      #
# # ================================== #

# df = pd.read_excel('data/ui_final/final_B_merged.xlsx', header=0, engine='openpyxl')

# # indexed by "answer text col", "question ID", "score col"
# answer_cols = [
#     ("Q9 (i)", 9.1, "9: Pick2 (2.0 pts)"), 
#     ("Q9 (ii)", 9.2, "9: Pick2 (2.0 pts)"), 
#     ("Q10 b(ii)", 10.0, "10.3: OE (1.0 pts)"), 
#     ("Q11", 11.0, "11 (2.0 pts)"), 
#     ("Q13 a)", 13.1, "13.1: OE (2.0 pts)"), 
#     ("Q13 b)", 13.2, "13.2: OE (2.0 pts)"), 
#     ("Q13 c)", 13.3, "13.3: OE (2.0 pts)"), 
#     ("Q16", 16.0, "16: OE (2.0 pts)"), 
#     ("Q17", 17.0, "17: OE (2.0 pts)"), 
#     ("Q19 U=Upside, D=downside", 19.0, "19: OE (2.0 pts)"), 
#     ("Q21", 21.0, "21: OE (2.0 pts)"), 
#     ("Q23", 23.0, "23: OE (2.0 pts)"),
#     ("Q25 i, ii", 25.0, "25 (2.0 pts)"), 
#     ("Q26 i, ii (H is omitted)", 26.0, "26.2 (2.0 pts)"), 
#     ("Q28 b) (B is block-based, T is text based)", 28.0, "28.2 (2.0 pts)"), 
#     ("Q29 i)", 29.1, "29.1 (2.0 pts)"), 
#     ("Q29 ii)", 29.2, "29.2 (2.0 pts)"), 
# ]

# for index, row in df.iterrows():
#     for answer_pair in answer_cols:
#         obj, created = Answer.objects.get_or_create(
#             answer_text = row[answer_pair[0]],
#             student_id = index,
#             question = Question.objects.get(question_exam_id=str(answer_pair[1])),
#             assigned_grade = row[answer_pair[2]]
#         )
#         if (created): print('Answer: {}-{} created'.format(answer_pair[1], index))


# ================================== #
#          EECS493 FINAL EXAM        #
# ================================== #

df = pd.read_csv('data/FinalExam.studentresponses.csv', header=0)
df['question_number'] = df['question_number'].astype(str)

grades_df = pd.read_csv('data/Final_exam_-_Tuesday_7pm_-_ALL_scores.csv', header=0)
new_cols = {x: x.split(" ")[0].split(":")[0] for x in grades_df.columns}
grades_df.rename(columns = new_cols, inplace = True)
new_cols = {x: str(float(x)) if x.isnumeric() else x for x in grades_df.columns}
grades_df.rename(columns = new_cols, inplace = True)

for index, row in df.iterrows():
    curr_ques = Question.objects.filter(question_exam_id=row['question_number'])
    if (curr_ques.exists()):
        try:
            grades_index = grades_df.index[grades_df['SID'] == row["student_id"]][0]

            obj, created = Answer.objects.get_or_create(
                answer_text = json.loads(row["answers"])['0'],
                student_id = row["student_id"],
                question = curr_ques.first(),
                assigned_grade = grades_df.iloc[grades_index][row['question_number']]
            )
            if (created): print('Answer #{} created'.format(index))
        except Exception as e:
            print(f"FAILED ON: {index}, {e}")
    # if (index > 130): break

    