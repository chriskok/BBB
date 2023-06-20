import openai
from my_secrets import my_secrets
openai_key = my_secrets.get('openai_key')
openai.api_key = openai_key

import random
random.seed(37)
import json

# e.g. "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hello!"}]
def prompt_chatgpt(prompt):
    model="gpt-3.5-turbo-16k"
    # model="gpt-3.5-turbo"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            temperature=0.0,
        )
        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR"
        else:
            return response["choices"][0]["message"]["content"]
    except Exception as e: 
        print(e)
        return "ERROR"

def create_rubrics(question, answers, num_samples=40):
    random_answers = random.sample(list(answers), num_samples)
    max_grade = 2

    answers_str = "\n".join(["{}. {}".format(answer.id, answer.answer_text) for i, answer in enumerate(random_answers)])
    # system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. Given the following random assortment of students' answers, please come up with a set of 10 rubrics that are of high quality: mutually exclusive, easily understood, reflective, and encompasses all kinds of answers (even the unseen). Rubrics can be positive or negative (covering all potential misunderstandings of the concept presented) and multiple rubrics can apply to the same answers. The maximum amount of points/grade for this answer is {max_grade}, please assign points accordingly (and include rubrics with zeroes, showing what it means to be incorrect). \n\n{answers_str}"""
    system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. The following is a random assortment of answers to the question by your students (formatted: <answer ID>. <answer>): \n\n{answers_str}"""
    # user_prompt = """For each rubric generated, please STRICTLY follow the JSON format: {"rubric": "<high-level concept for this rubric>", "answers": "<comma-separated list of answer numbers that fit this rubric>", "points": "<points - from the max to 0>", "explanation": "<elaboration on what really counts for this rubric>"} """
    user_prompt = """Please come up with a set of 5 rubrics that are of high quality: mutually exclusive, easily understood, reflective, and encompasses all kinds of answers (even the unseen). For the output, create a comma-separated list of python dictionaries that STRICTLY follow the JSON format: [{"rubric": "<high-level concept for this rubric>", "answer_ids": "<comma-separated list IDs of top 5 representative answers that fit this rubric>", "explanation": "<elaboration on what really counts for this rubric>"}, ...] """

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    response = prompt_chatgpt(msgs)

    try:
        rubrics = json.loads(response)
    except Exception as e:
        print(e)
        rubrics = []

    return rubrics, msgs


def tag_answers(question, answers, rubrics, num_samples=40):
    random_answers = random.sample(list(answers), num_samples)

    answers_str = "\n".join(["{}. {}".format(answer.id, answer.answer_text) for i, answer in enumerate(random_answers)])
    rubrics_str = "\n".join(["R{}. {} (meaning: {})".format(i+1, rubric["rubric"], rubric['explanation']) for i, rubric in enumerate(rubrics)])
    system_prompt = f"""You are an expert instructor for your given course. You've given the short-answer, open-ended question "{question.question_text}" on a recent final exam. You and your expert instructor partner created the following rubrics for this question (labelled R<rubric number> below): \n\n{rubrics_str}"""
    user_prompt = """Your task is to assign the rubric labels to each of the following students' answers (formatted: <answer ID>. <answer>). Please assign labels to each of the answers provided. Each answer can have multiple rubrics applied too. Treat this as a multi-class classification task. Please provide reasoning for your labels as well. For the output, create a comma-separated list of python dictionaries that STRICTLY follow the JSON format: [{"answer_id": "<id of the answer>", "rubrics": "<comma-separated list of rubrics (labelled R<number>) that apply to this answer>", "reasoning": "<reason you think the rubrics you chose apply to this answer>"}, ...]\n\nStudents' Answers:\n""" + answers_str

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    response = prompt_chatgpt(msgs)

    try:
        tags = json.loads(response)
    except Exception as e:
        print(e)
        tags = []

    return tags, msgs