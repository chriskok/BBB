from blocks.models import Answer, ChatGPTGradeAndFeedback
import openai
from .my_secrets import my_secrets
openai_key = my_secrets.get('openai_key')

all_answers = Answer.objects.all()
methods = ["question_only", "examples", "rubrics", "rubrics_and_examples"]
max_grades = {'23.0':3.0, '27.0':2.0, '7.0':2.0}

def chatgpt_question_only(question, answer):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are an professor for a class called User Interface Development. Please provide a grade"},
            {"role": "user", "content": f"{answer}"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
    )
    if "error" in response:
        print("OPENAI ERROR: {}".format(response))
        return "[system note] The server had an error while processing your request. Sorry about that! Please try again in a minute.", 0
    else:
        return response["choices"][0]["message"]["content"]

for index, answer in enumerate(all_answers):
    print(f"creating chatgpt response for {index+1}/{len(all_answers)}")
    if (answer.question.question_exam_id in max_grades): print(max_grades[answer.question.question_exam_id])

