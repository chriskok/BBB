import pandas as pd
import torch
import time
import os

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
modelPath = 'all-MiniLM-L6-v2'
model = SentenceTransformer(modelPath)
from sklearn.cluster import AgglomerativeClustering
from blocks.models import *

import openai
from my_secrets import my_secrets
openai_key = my_secrets.get('openai_key')
openai.api_key = openai_key

# --------------------------------------------- SAMPLING METHODS ---------------------------------------------
# further AND closest items from mean
def outlier_score(df):
    sentences = df["answer_text"].apply(str).tolist()
    embeddings = model.encode(sentences, convert_to_tensor=True)
    emb_df = pd.DataFrame(embeddings.numpy())
    # adding a row of the mean embedding
    emb_df = pd.concat([emb_df, emb_df.apply(['mean'])])
    # convert back to torch tensor
    embeddings = torch.tensor(emb_df.values)
    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)
    # Get the distance from the mean embedding row
    chosen_scores = cosine_scores[-1]
    return_df = df.copy()
    return_df['score'] = chosen_scores[:-1]  # assign all scores to df, while removing the last sentence that we added
    # TODO: is this valid? We want the most extreme points, so most or least similar
    # TODO: can try without the next line and just sorting from bottom up (or with diff ranges)
    # scale the scores to be between 0 and 1
    return_df['score'] = (return_df['score'] - return_df['score'].min()) / (return_df['score'].max() - return_df['score'].min())
    return_df['score'] = return_df['score'].apply(lambda x: 1-x if x < 0.5 else x)  
    return_df = return_df.sort_values(by=['score'], ascending=False)
    return return_df

# further items from mean
def outlier_score_furthest(df):
    sentences = df["answer_text"].apply(str).tolist()
    embeddings = model.encode(sentences, convert_to_tensor=True)
    emb_df = pd.DataFrame(embeddings.numpy())
    # adding a row of the mean embedding
    emb_df = pd.concat([emb_df, emb_df.apply(['mean'])])
    # convert back to torch tensor
    embeddings = torch.tensor(emb_df.values)
    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)
    # Get the distance from the mean embedding row
    chosen_scores = cosine_scores[-1]
    return_df = df.copy()
    return_df['score'] = chosen_scores[:-1]  # assign all scores to df, while removing the last sentence that we added
    return_df = return_df.sort_values(by=['score'], ascending=True)
    return return_df

# closest items to mean
def outlier_score_closest(df):
    sentences = df["answer_text"].apply(str).tolist()
    embeddings = model.encode(sentences, convert_to_tensor=True)
    emb_df = pd.DataFrame(embeddings.numpy())
    # adding a row of the mean embedding
    emb_df = pd.concat([emb_df, emb_df.apply(['mean'])])
    # convert back to torch tensor
    embeddings = torch.tensor(emb_df.values)
    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, embeddings)
    # Get the distance from the mean embedding row
    chosen_scores = cosine_scores[-1]
    return_df = df.copy()
    return_df['score'] = chosen_scores[:-1]  # assign all scores to df, while removing the last sentence that we added
    return_df = return_df.sort_values(by=['score'], ascending=False)
    return return_df

def cluster(df, n_clusters=10):
    sentences = df["answer_text"].apply(str).tolist()
    # Compute SBERT embeddings
    model = SentenceTransformer(modelPath)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    # Cluster sentences with AgglomerativeClustering
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters) #, affinity='cosine', linkage='average', distance_threshold=threshold)
    clustering_model.fit(embeddings)
    # Assign the cluster labels to the dataframe
    df['cluster'] = clustering_model.labels_
    return df

# --------------------------------------------- FILE WRITING & PROMPTING ---------------------------------------------
def write_to_file(df, filename):
    # write the top N answer_texts and outlier_scores to a file
    with open(filename, 'w') as f:
        for index, row in df.iterrows():
            # remove new lines from answer_text
            new_text = row['answer_text'].replace('\n', ' ')
            f.write(new_text + '\t')
            f.write(str(row['score']) + '\t')
            f.write('\n')

def write_to_file_cluster(df, filename, sample=1):
    # write the answer_texts and clusters to a file
    with open(filename, 'w') as f:
        for index, row in df.iterrows():
            # remove new lines from answer_text
            new_text = row['answer_text'].replace('\n', ' ')
            f.write(new_text + '\t')
            f.write(str(row['cluster']) + '\t')
            f.write('\n')
        f.write('\n\n')
        # get total number of clusters
        num_clusters = df['cluster'].nunique()
        # then get a random sample from each cluster to just print
        for i in range(num_clusters):
            cluster_df = df.loc[df['cluster'] == i]
            cluster_sample = cluster_df.sample(n=sample, replace=True)
            for index, row in cluster_sample.iterrows():
                # remove new lines from answer_text
                new_text = row['answer_text'].replace('\n', ' ')
                f.write(new_text + '\t')
                f.write(str(row['cluster']) + '\t')
                f.write('\n')

def write_responses_to_file(title, response, filename):
    with open(filename, 'a') as file:  # Open the file in append mode
        uppercase_title = title.upper()
        file.write(f"{uppercase_title}\n\n" + response + '\n\n---------------------------------------------------------------------------\n\n')

def prompt_gpt4(prompt):
    model="gpt-4"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            temperature=1.0,
        )
        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR"
        else:
            return response["choices"][0]["message"]["content"]
    except Exception as e: 
        print(e)
        return "ERROR"

# Placeholder function for GPT prompting
def prompt_gpt(system_prompt=None, prompt_text=""):
    if (system_prompt): msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_text}]
    else: msgs = [{"role": "user", "content": prompt_text}]
    response = prompt_gpt4(msgs)
    return response

# --------------------------------------------- PROMPT PHRASING ---------------------------------------------
def prompt_phrasing(df, question_text, filename):
    prompts = [
        "Using the examples provided from a dataset, suggest potential rubric items that would be effective for evaluating students' answers.",
        "Based on the provided answers, identify the main themes or recurrent topics that students emphasized.",
        "Examine the answers and highlight the key insights students have provided.",
        "From the students' responses, deduce the primary concepts or principles they understand.",
        "If you were to summarize the essence of all student responses into a few critical takeaways, what would they be?",
        "Looking at the student responses, what perspectives or viewpoints do they commonly adopt?",
        "Assess the depth of understanding presented in the answers. What are the foundational, intermediate, and advanced concepts students touch upon?",
        "Interpret the collective knowledge demonstrated by the answers. What holistic understanding or overarching message do students convey?"
    ]
    prepended_text = f"You are an expert instructor for your given course. You're in the process of evaluating student answers to the short-answer, open-ended question: '{question_text}' in the recent final exam. "
    # get total number of clusters
    num_clusters = df['cluster'].nunique()
    # then get a random sample from each cluster and add the answer_texts to a list
    sample_answers = []
    for i in range(num_clusters):
        cluster_df = df.loc[df['cluster'] == i]
        cluster_sample = cluster_df.sample(n=1, replace=True)
        for index, row in cluster_sample.iterrows():
            # remove new lines from answer_text
            new_text = row['answer_text'].replace('\n', ' ')
            sample_answers.append(new_text)
    user_prompt = "\n\n".join(sample_answers)
    with open(filename, 'a') as file:  # Open the file in append mode
        for prompt in prompts:
            curr_prompt = prepended_text + prompt
            msgs = [{"role": "system", "content": curr_prompt}, {"role": "user", "content": user_prompt}]
            response = prompt_gpt4(prompt=msgs)
            uppercase_prompt = prompt.upper()
            file.write(f"{uppercase_prompt}\n\n" + response + '\n\n---------------------------------------------------------------------------\n\n')

# --------------------------------------------- SAMPLE SIZE ---------------------------------------------
def sample_answers(df, n_samples=1):
    samples = []
    num_clusters = df['cluster'].nunique()
    for cluster_id in range(num_clusters): 
        sample = df[df['cluster'] == cluster_id].sample(n=n_samples, replace=True)
        for index, row in sample.iterrows():
            # remove new lines from answer_text
            new_text = row['answer_text'].replace('\n', ' ')
            # append to samples
            samples.append(new_text)
    return samples

def sample_size_prompt(df, question_text, file_location):
    rubric_prompt = f"You are an expert instructor for your given course. You're in the process of designing a rubric to evaluate the short-answer, open-ended question: '{question_text}' for a recent final exam. Using the examples provided from a dataset, suggest potential rubric items that would be effective for evaluating students' answers."
    theme_prompt = f"You are an expert instructor for your given course. You're in the process of designing a rubric to evaluate the short-answer, open-ended question: '{question_text}' for a recent final exam. Based on the provided answers, identify the main themes or recurrent topics that students emphasized."
    # sample 1 answer from each cluster - total 20
    samples = sample_answers(df, n_samples=1)
    user_prompt = "\n\n".join(samples)
    response = prompt_gpt(rubric_prompt, user_prompt)
    write_responses_to_file("Rubrics - 20 Answers", response, file_location + 'sample_size_prompts.txt')
    response = prompt_gpt(theme_prompt, user_prompt)
    write_responses_to_file("Themes - 20 Answers", response, file_location + 'sample_size_prompts.txt')
    time.sleep(60)
    # sample 2 answers from each cluster - total 40
    samples = sample_answers(df, n_samples=2)
    user_prompt = "\n\n".join(samples)
    response = prompt_gpt(rubric_prompt, user_prompt)
    write_responses_to_file("Rubrics - 40 Answers", response, file_location + 'sample_size_prompts.txt')
    response = prompt_gpt(theme_prompt, user_prompt)
    write_responses_to_file("Themes - 40 Answers", response, file_location + 'sample_size_prompts.txt')
    time.sleep(60)
    # sample 4 answers from each cluster - total 80
    samples = sample_answers(df, n_samples=4)
    user_prompt = "\n\n".join(samples)
    response = prompt_gpt(rubric_prompt, user_prompt)
    write_responses_to_file("Rubrics - 80 Answers", response, file_location + 'sample_size_prompts.txt')
    response = prompt_gpt(theme_prompt, user_prompt)
    write_responses_to_file("Themes - 80 Answers", response, file_location + 'sample_size_prompts.txt')
    time.sleep(60)

# --------------------------------------------- NUMBER OF RUBRICS/THEMES ---------------------------------------------
def num_rubrics_prompt(df, question_text, file_location, n=3):
    # generate several rounds of 3/5/10/X rubrics or themes to analyze the different outcomes
    rubric_prompt = f"You are an expert instructor for your given course. You're in the process of designing a rubric to evaluate the short-answer, open-ended question: '{question_text}' for a recent final exam. Using the examples provided from a dataset, suggest potential rubric items that would be effective for evaluating students' answers.\n\nCome up with a list of {n} rubric items that effectively encapsulate the types of answers in the dataset."
    theme_prompt = f"You are an expert instructor for your given course. You're in the process of designing a rubric to evaluate the short-answer, open-ended question: '{question_text}' for a recent final exam. Based on the provided answers, identify the main themes or recurrent topics that students emphasized.\n\nCome up with a list of {n} themes that effectively encapsulate the types of answers in the dataset."
    samples = sample_answers(df, n_samples=1)  # total 20 answers
    user_prompt = "\n\n".join(samples)
    response = prompt_gpt(rubric_prompt, user_prompt)
    write_responses_to_file(f"Rubrics - {n} items", response, file_location + 'num_rubrics_prompts.txt')
    response = prompt_gpt(theme_prompt, user_prompt)
    write_responses_to_file(f"Themes - {n} items", response, file_location + 'num_rubrics_prompts.txt')

# --------------------------------------------- SAMPLE SELECTION ---------------------------------------------
def sample_selection_prompt(df, question_text, file_location, method_name, clustered=True):
    rubric_prompt = f"You are an expert instructor for your given course. You're in the process of designing a rubric to evaluate the short-answer, open-ended question: '{question_text}' for a recent final exam. Using the examples provided from a dataset, suggest potential rubric items that would be effective for evaluating students' answers.\n\nCome up with a list of 5 rubric items that effectively encapsulate the types of answers in the dataset."
    theme_prompt = f"You are an expert instructor for your given course. You're in the process of designing a rubric to evaluate the short-answer, open-ended question: '{question_text}' for a recent final exam. Based on the provided answers, identify the main themes or recurrent topics that students emphasized.\n\nCome up with a list of 5 themes that effectively encapsulate the types of answers in the dataset."
    if (clustered): samples = sample_answers(df, n_samples=1)  # total 20 answers
    else: 
        samples = df['answer_text'].tolist()
        samples = samples[:20]  # total 20 answers
    user_prompt = "\n\n".join(samples)
    response = prompt_gpt(rubric_prompt, user_prompt)
    write_responses_to_file(f"Rubrics - {method_name}", response, file_location + 'sample_selection_prompt.txt')
    response = prompt_gpt(theme_prompt, user_prompt)
    write_responses_to_file(f"Themes - {method_name}", response, file_location + 'sample_selection_prompt.txt')

# --------------------------------------------- NUMBER OF CLUSTERS SHOWN ---------------------------------------------
def full_context_prompt(df, question_text, file_location, num_samples=1):
    prompts = []
    num_clusters = df['cluster'].nunique()
    for cluster_id in range(num_clusters): 
        sample = df[df['cluster'] == cluster_id].sample(n=num_samples, replace=True)
        prompts.extend(sample['answer_text'].tolist())
    user_prompt = "\n\n".join(prompts)
    system_prompt = f"You are an expert instructor for your given course. You're in the process of evaluating student answers to the short-answer, open-ended question: '{question_text}' in the recent final exam. Using the examples provided from a dataset, suggest potential rubric items that would be effective for evaluating students' answers."
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = prompt_gpt4(msgs)
    write_responses_to_file("Full Context Prompt", response, file_location + 'cluster_size_prompts.txt')

def x_clusters_prompt(df, x, question_text, file_location, num_samples_per_cluster=5):
    cluster_ids = sorted(df['cluster'].unique())
    system_prompt = f"You are an expert instructor for your given course. Currently, you are evaluating student responses to the question: '{question_text}' from a recent final exam. By examining diverse clusters, we hope to inspire more detailed insights based on the variations observed. Given a selection of answers from different pairings of clusters, please derive and suggest potential rubric items that capture the nuances and differences in students' understanding. What rubric items can best evaluate the diverse perspectives and knowledge levels reflected in these examples? Please output in the following format (one for each cluster): - <rubric title>: <rubric description kept to 15 words maximum>"
    responses = []
    for i in range(0, len(cluster_ids), x):
        selected_clusters = cluster_ids[i:i+x]
        user_prompt = ""
        for cluster_id in selected_clusters:
            sample = df[df['cluster'] == cluster_id].sample(n=num_samples_per_cluster, replace=True)
            cluster_string = f"\n\nCluster {cluster_id}: \n\n" + "\n\n".join(sample['answer_text'].tolist())
            user_prompt += cluster_string
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = prompt_gpt4(msgs)
        responses.append(response)
    summarizing_prompt = f"You will be provided a list of rubric items intended for evaluating student responses on the question: '{question_text}'. Review these rubric items and eliminate those that are either too similar to one another or not directly relevant to the core topic. Which of these rubrics should be retained, and which should be removed due to redundancy or irrelevance?"
    all_rubrics = "\n\n".join(responses)
    msgs = [{"role": "system", "content": summarizing_prompt}, {"role": "user", "content": all_rubrics}]
    response = prompt_gpt4(msgs)
    full_response = all_rubrics + "\n\n" + response
    write_responses_to_file(f"{x} Clusters Prompt", full_response, file_location + 'cluster_size_prompts.txt')

# --------------------------------------------- NEGATIVE/POSITIVE SCENARIOS ---------------------------------------------
def manual_scenario(answers, file_location):
    # GPT generates themes
    themes = prompt_gpt(f"Generate themes based on these answers: \n\n{answers}")
    # Human labels the themes as positive or negative (this step is interactive and manual)
    labeled_themes = input(f"Label the following themes as positive or negative: \n\n{themes}")
    # GPT generates positive and negative rubrics based on human-labeled themes
    rubrics = prompt_gpt(f"Generate 5 positive and 5 negative (covering all potential misunderstandings of the concept presented) rubrics based on these labeled themes: \n\n{labeled_themes}")
    write_responses_to_file("Manual Scenario", rubrics, file_location + 'negative_rubric_generation_scenarios.txt')

def semi_auto_scenario(answers, file_location):
    # GPT generates themes and suggests polarity
    themes_with_polarity = prompt_gpt(f"Generate themes and suggest polarity for these answers: \n\n{answers}")
    # Human checks and edits polarity (this step is interactive)
    checked_themes = input(f"Check and edit the polarity for the following themes: \n\n{themes_with_polarity}")
    # GPT generates positive and negative rubrics based on checked themes
    rubrics = prompt_gpt(f"Generate 5 positive and 5 negative (covering all potential misunderstandings of the concept presented) rubrics based on these checked themes: \n\n{checked_themes}")
    write_responses_to_file("Semi-Auto Scenario", rubrics, file_location + 'negative_rubric_generation_scenarios.txt')

def auto_scenario(answers, question_text, file_location):
    # GPT generates themes and determines polarity
    themes_with_polarity = prompt_gpt(f"You are an expert instructor for your given course. Currently, you are evaluating student responses to the question: '{question_text}' from a recent final exam. Generate 10 themes for the given answers (along with a polarity - whether or not it's a common good answer vs common misunderstanding)", f"{answers}")
    # GPT generates positive and negative rubrics based on themes with polarity
    rubrics = prompt_gpt(f"You are an expert instructor for your given course. Currently, you are evaluating student responses to the question: '{question_text}' from a recent final exam. Generate 5 positive (common good answers) and 5 negative (potential misunderstandings) rubric items based on the given answers and these themes: \n\n{themes_with_polarity} \n\n Please output in the following format: - <rubric title>: <rubric description kept to 15 words maximum> (e.g. <example answer from the dataset provided>)", f"DATASET:\n\n{answers}")
    full_response = themes_with_polarity + "\n\n" + rubrics
    write_responses_to_file("Auto Scenario", full_response, file_location + 'negative_rubric_generation_scenarios.txt')

def auto_scenario_2(answers, question_text, file_location):
    # GPT generates themes and determines polarity
    system_prompt = f"You are an expert instructor for your given course. You're in the process of evaluating student answers to the short-answer, open-ended question: '{question_text}' in the recent final exam."
    theme_prompt = f"Based on the provided answers, identify the main themes or recurrent topics that students emphasized. Keep each theme short and concise and don't overlap them too much with each other. Follow the format: \n- <theme title>: <theme description kept to 15 words maximum> \n\n{answers}"
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": theme_prompt}]
    themes_response = prompt_gpt4(msgs)
    # GPT generates positive and negative rubrics based on themes with polarity
    rubrics_prompt = f"Generate 5 positive (common good answers) and 5 negative (potential misunderstandings) rubric items based on the given answers and the themes generated. Please output in the following format: \n- <rubric title>: <rubric description kept to 15 words maximum> (example: <example answer from the dataset provided>)"
    msgs.append({"role": "assistant", "content": themes_response})
    msgs.append({"role": "user", "content": rubrics_prompt})
    rubrics_response = prompt_gpt4(msgs)
    full_response = themes_response + "\n\n" + rubrics_response
    write_responses_to_file("Auto Scenario 2", full_response, file_location + 'negative_rubric_generation_scenarios.txt')

def auto_no_theme_scenario(answers, question_text, file_location):
    # GPT generates positive and negative rubrics based on themes with polarity
    rubrics = prompt_gpt(f"You are an expert instructor for your given course. Currently, you are evaluating student responses to the question: '{question_text}' from a recent final exam. Generate 5 positive (common good answers) and 5 negative (potential misunderstandings) rubric items based on the given answers. Please output in the following format: - <rubric title>: <rubric description kept to 15 words maximum> (e.g. <example answer from the dataset provided>)", f"DATASET:\n\n{answers}")
    write_responses_to_file("Auto No Theme Scenario", rubrics, file_location + 'negative_rubric_generation_scenarios.txt')

questions = Question.objects.all()
# questions = Question.objects.filter(id=35)
for q in questions:
    question_text = q.question_text
    print(f"{q.question_exam_id}: {question_text}")
    file_location = f"results/chi_evaluations/question_{q.question_exam_id}/" + time.strftime("%Y%m%d-%H%M%S") + "/"
    # create directory if it doesn't exist
    if not os.path.exists(file_location):
        os.makedirs(file_location)
    df = pd.DataFrame(list(q.answer_set.values()))
    # --------------------------------------------- PROMPT PHRASING ---------------------------------------------
    cluster20 = cluster(df, n_clusters=20)
    prompt_phrasing(cluster20, question_text, file_location + 'prompt_evaluations.txt')
    time.sleep(60)
    # --------------------------------------------- SAMPLE SIZE ---------------------------------------------
    cluster20 = cluster(df, n_clusters=20)
    sample_size_prompt(cluster20, question_text, file_location)
    time.sleep(60)
    # --------------------------------------------- NUMBER OF RUBRICS ---------------------------------------------
    cluster20 = cluster(df, n_clusters=20)
    num_rubrics_list = ["3", "5", "10", ""]
    for n in num_rubrics_list:
        num_rubrics_prompt(cluster20, question_text, file_location, n=n)
        time.sleep(60)
    # --------------------------------------------- SAMPLE SELECTION ---------------------------------------------
    outlier_df = outlier_score(df)
    sample_selection_prompt(outlier_df, question_text, file_location, method_name="Outlier Score", clustered=False)
    # write_to_file(outlier_df, file_location + 'outlier_score.txt')
    further_df = outlier_score_furthest(df)
    sample_selection_prompt(further_df, question_text, file_location, method_name="Furthest from Mean", clustered=False)
    # write_to_file(further_df, file_location + 'outlier_score_furthest.txt')
    closer_df = outlier_score_closest(df)
    sample_selection_prompt(closer_df, question_text, file_location, method_name="Closest to Mean", clustered=False)
    time.sleep(60)
    # write_to_file(closer_df, file_location + 'outlier_score_closest.txt')
    cluster10 = cluster(df, n_clusters=10)
    sample_selection_prompt(cluster10, question_text, file_location, method_name="Cluster 10")
    # write_to_file_cluster(cluster10, file_location + 'cluster10.txt', sample=2)
    cluster20 = cluster(df, n_clusters=20)
    sample_selection_prompt(cluster20, question_text, file_location, method_name="Cluster 20")
    # write_to_file_cluster(cluster20, file_location + 'cluster20.txt', sample=1)
    time.sleep(60)
    # --------------------------------------------- # SHOWN CLUSTERS ---------------------------------------------
    cluster20 = cluster(df, n_clusters=20)
    full_context_prompt(cluster20, question_text, file_location, num_samples=1)
    time.sleep(60)
    x_clusters_prompt(cluster20, x=2, question_text=question_text, file_location=file_location, num_samples_per_cluster=5)
    time.sleep(60)
    x_clusters_prompt(cluster20, x=5, question_text=question_text,  file_location=file_location, num_samples_per_cluster=5)
    time.sleep(60)
    x_clusters_prompt(cluster20, x=10, question_text=question_text, file_location=file_location, num_samples_per_cluster=5)
    time.sleep(60)
    # --------------------------------------------- FULL PIPELINE ---------------------------------------------
    cluster20 = cluster(df, n_clusters=20)
    samples = []
    num_clusters = df['cluster'].nunique()
    for cluster_id in range(num_clusters): 
        sample = df[df['cluster'] == cluster_id].sample(n=1, replace=True)
        for index, row in sample.iterrows():
            # remove new lines from answer_text
            new_text = row['answer_text'].replace('\n', ' ')
            # append to samples
            samples.append(new_text)
    samples_string = "\n\n".join(samples)
    # manual_scenario(samples_string, file_location)
    # semi_auto_scenario(samples_string, file_location)
    # auto_scenario(samples_string, file_location)
    auto_scenario_2(samples_string, question_text, file_location)
    auto_no_theme_scenario(samples_string, question_text, file_location)


