from blocks.models import Answer, ChatGPTGradeAndFeedback
from django.db.models import Max

import openai
import time

from my_secrets import my_secrets
openai_key = my_secrets.get('openai_key')

# Set up Open AI
openai.api_key = openai_key



def chatgpt_question_only(question, answer, max_grade):
    prompt = [
                {"role": "system", "content": f"You are an professor for a class called User Interface Development. You are grading and providing feedback for the final exam of the class. You are currently on the question: {question}. Please provide a grade between 0.0 and {str(max_grade)} for the following student's answer."},
                {"role": "student", "content": f"{answer}"},
            ]
    model="gpt-3.5-turbo"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt
        )
        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR", prompt, model
        else:
            return response["choices"][0]["message"]["content"], prompt, model
    except Exception as e: 
        print(e)
        return "ERROR", prompt, model

def gpt3_question_only(question, answer, max_grade):
    prompt=f"You are an professor for a class called User Interface Development. You are grading and providing feedback for the final exam of the class. You are currently on the question: {question}. Please provide a grade between 0.0 and {str(max_grade)} for the following student's answer and feedback/reasoning for the grade so the student can learn from it.\nStudent:{answer}\nTeacher:"
    model="text-davinci-003"
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7,
            stop= ["\nStudent:", "\nTeacher:"]
        )

        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR", prompt, model
        else:
            return response["choices"][0]["text"], prompt, model
    except Exception as e: 
        print(e)
        return "ERROR", prompt, model

# def chatgpt_examples(question, answer, max_grade):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": f"You are an professor for a class called User Interface Development. You are grading and providing feedback for the final exam of the class. You are currently on the question: {question}. Please provide a grade between 0.0 and {str(max_grade)} for the following students' answers"},
#             {"role": "student", "content": f"{answer}"},
#             {"role": "teacher", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#             {"role": "student", "content": "Where was it played?"}
#         ]
#     )
#     if "error" in response:
#         print("OPENAI ERROR: {}".format(response))
#         return "ERROR"
#     else:
#         return response["choices"][0]["message"]["content"]

curr_trial_number = ChatGPTGradeAndFeedback.objects.aggregate(Max('trial_run_number'))['trial_run_number__max'] + 1
print(f"Trial Run #{curr_trial_number}")

starting_index = 80
all_answers = list(Answer.objects.all())
methods = ["question_only", "examples", "rubrics", "rubrics_and_examples"]
max_grades = {'23.0':3.0, '27.0':2.0, '7.0':2.0}

for index, answer in enumerate(all_answers[starting_index:]):
    print(f"Creating chatgpt response: {index+1+starting_index}/{len(all_answers)}")
    chatgpt_response, prompt, model = gpt3_question_only(answer.question.question_text, answer.answer_text, max_grades[answer.question.question_exam_id])
    time.sleep(5)
    if (chatgpt_response[0] == "ERROR"):
        print(f"Retrying: {index+1+starting_index}/{len(all_answers)} in a minute")
        time.sleep(60)
        chatgpt_response, prompt, model = gpt3_question_only(answer.question.question_text, answer.answer_text, max_grades[answer.question.question_exam_id])
    ChatGPTGradeAndFeedback.objects.create(answer=answer,response=chatgpt_response,prompt=prompt,prompt_type="question_only",trial_run_number=curr_trial_number,openai_model=model)

