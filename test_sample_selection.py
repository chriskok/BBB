import pandas as pd
import torch
import time

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
            cluster_sample = cluster_df.sample(n=sample)
            for index, row in cluster_sample.iterrows():
                # remove new lines from answer_text
                new_text = row['answer_text'].replace('\n', ' ')
                f.write(new_text + '\t')
                f.write(str(row['cluster']) + '\t')
                f.write('\n')

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

def prompt_gpt_and_save(df, filename):
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
    prepended_text = "You are an expert instructor for your given course. You're in the process of evaluating student answers to the short-answer, open-ended question: 'Describe why we want to use asynchronous programming in Javascript?' in the recent final exam. "
    # get total number of clusters
    num_clusters = df['cluster'].nunique()
    # then get a random sample from each cluster and add the answer_texts to a list
    sample_answers = []
    for i in range(num_clusters):
        cluster_df = df.loc[df['cluster'] == i]
        cluster_sample = cluster_df.sample(n=1)
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

def write_responses_to_file(title, response, filename):
    with open(filename, 'a') as file:  # Open the file in append mode
        uppercase_title = title.upper()
        file.write(f"{uppercase_title}\n\n" + response + '\n\n---------------------------------------------------------------------------\n\n')

def full_context_prompt(df, num_samples=1):
    prompts = []
    num_clusters = df['cluster'].nunique()
    for cluster_id in range(num_clusters): 
        sample = df[df['cluster'] == cluster_id].sample(n=num_samples)
        prompts.extend(sample['answer_text'].tolist())
    user_prompt = "\n\n".join(prompts)
    system_prompt = "You are an expert instructor for your given course. You're in the process of evaluating student answers to the short-answer, open-ended question: 'Describe why we want to use asynchronous programming in Javascript?' in the recent final exam. Using the examples provided from a dataset, suggest potential rubric items that would be effective for evaluating students' answers."
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    response = prompt_gpt4(msgs)
    write_responses_to_file("Full Context Prompt", response, 'results/chi_evaluations/cluster_size_prompts.txt')

def x_clusters_prompt(df, x, num_samples_per_cluster=5):
    cluster_ids = sorted(df['cluster'].unique())
    system_prompt = "You are an expert instructor for your given course. Currently, you are evaluating student responses to the question: 'Describe why we want to use asynchronous programming in Javascript?' from a recent final exam. By examining diverse clusters, we hope to inspire more detailed insights based on the variations observed. Given a selection of answers from different pairings of clusters, please derive and suggest potential rubric items that capture the nuances and differences in students' understanding. What rubric items can best evaluate the diverse perspectives and knowledge levels reflected in these examples? Please output in the following format (one for each cluster): - <rubric title>: <rubric description kept to 15 words maximum>"
    responses = []
    for i in range(0, len(cluster_ids), x):
        selected_clusters = cluster_ids[i:i+x]
        user_prompt = ""
        for cluster_id in selected_clusters:
            sample = df[df['cluster'] == cluster_id].sample(n=num_samples_per_cluster)
            cluster_string = f"\n\nCluster {cluster_id}: \n\n" + "\n\n".join(sample['answer_text'].tolist())
            user_prompt += cluster_string
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = prompt_gpt4(msgs)
        responses.append(response)
    summarizing_prompt = "You will be provided a list of rubric items intended for evaluating student responses on the question: 'Describe why we want to use asynchronous programming in Javascript?'. Review these rubric items and eliminate those that are either too similar to one another or not directly relevant to the core topic. Which of these rubrics should be retained, and which should be removed due to redundancy or irrelevance?"
    all_rubrics = "\n\n".join(responses)
    msgs = [{"role": "system", "content": summarizing_prompt}, {"role": "user", "content": all_rubrics}]
    response = prompt_gpt4(msgs)
    full_response = all_rubrics + "\n\n" + response
    write_responses_to_file(f"{x} Clusters Prompt", full_response, 'results/chi_evaluations/cluster_size_prompts.txt')

# Placeholder function for GPT prompting
def prompt_gpt(system_prompt=None, prompt_text=""):
    if (system_prompt): msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_text}]
    else: msgs = [{"role": "user", "content": prompt_text}]
    response = prompt_gpt4(msgs)
    return response

def manual_scenario(answers):
    # GPT generates themes
    themes = prompt_gpt(f"Generate themes based on these answers: \n\n{answers}")
    # Human labels the themes as positive or negative (this step is interactive and manual)
    labeled_themes = input(f"Label the following themes as positive or negative: \n\n{themes}")
    # GPT generates positive and negative rubrics based on human-labeled themes
    rubrics = prompt_gpt(f"Generate 5 positive and 5 negative (covering all potential misunderstandings of the concept presented) rubrics based on these labeled themes: \n\n{labeled_themes}")
    write_responses_to_file("Manual Scenario", rubrics, 'results/chi_evaluations/negative_rubric_generation_scenarios.txt')

def semi_auto_scenario(answers):
    # GPT generates themes and suggests polarity
    themes_with_polarity = prompt_gpt(f"Generate themes and suggest polarity for these answers: \n\n{answers}")
    # Human checks and edits polarity (this step is interactive)
    checked_themes = input(f"Check and edit the polarity for the following themes: \n\n{themes_with_polarity}")
    # GPT generates positive and negative rubrics based on checked themes
    rubrics = prompt_gpt(f"Generate 5 positive and 5 negative (covering all potential misunderstandings of the concept presented) rubrics based on these checked themes: \n\n{checked_themes}")
    write_responses_to_file("Semi-Auto Scenario", rubrics, 'results/chi_evaluations/negative_rubric_generation_scenarios.txt')

def auto_scenario(answers):
    # GPT generates themes and determines polarity
    themes_with_polarity = prompt_gpt(f"You are an expert instructor for your given course. Currently, you are evaluating student responses to the question: 'Describe why we want to use asynchronous programming in Javascript?' from a recent final exam. Generate 10 themes for the given answers (along with a polarity - whether or not it's a common good answer vs common misunderstanding)", f"{answers}")
    # GPT generates positive and negative rubrics based on themes with polarity
    rubrics = prompt_gpt(f"You are an expert instructor for your given course. Currently, you are evaluating student responses to the question: 'Describe why we want to use asynchronous programming in Javascript?' from a recent final exam. Generate 5 positive (common good answers) and 5 negative (potential misunderstandings) rubric items based on the given answers and these themes: \n\n{themes_with_polarity} \n\n Please output in the following format: - <rubric title>: <rubric description kept to 15 words maximum> (e.g. <example answer from the dataset provided>)", f"DATASET:\n\n{answers}")
    full_response = themes_with_polarity + "\n\n" + rubrics
    write_responses_to_file("Auto Scenario", full_response, 'results/chi_evaluations/negative_rubric_generation_scenarios.txt')

def auto_no_theme_scenario(answers):
    # GPT generates positive and negative rubrics based on themes with polarity
    rubrics = prompt_gpt(f"You are an expert instructor for your given course. Currently, you are evaluating student responses to the question: 'Describe why we want to use asynchronous programming in Javascript?' from a recent final exam. Generate 5 positive (common good answers) and 5 negative (potential misunderstandings) rubric items based on the given answers. Please output in the following format: - <rubric title>: <rubric description kept to 15 words maximum> (e.g. <example answer from the dataset provided>)", f"DATASET:\n\n{answers}")
    write_responses_to_file("Auto No Theme Scenario", rubrics, 'results/chi_evaluations/negative_rubric_generation_scenarios.txt')

# questions = Question.objects.all()
questions = Question.objects.filter(id=35)
for q in questions:
    print(q)
    df = pd.DataFrame(list(q.answer_set.values()))
    # ---------------------------------------------
    # outlier_df = outlier_score(df)
    # write_to_file(outlier_df, 'results/chi_evaluations/outlier_score.txt')
    # further_df = outlier_score_furthest(df)
    # write_to_file(further_df, 'results/chi_evaluations/outlier_score_furthest.txt')
    # closer_df = outlier_score_closest(df)
    # write_to_file(closer_df, 'results/chi_evaluations/outlier_score_closest.txt')
    # ---------------------------------------------
    # cluster10 = cluster(df, n_clusters=10)
    # write_to_file_cluster(cluster10, 'results/chi_evaluations/cluster10.txt', sample=2)
    # cluster20 = cluster(df, n_clusters=20)
    # write_to_file_cluster(cluster20, 'results/chi_evaluations/cluster20.txt', sample=1)
    # prompt_gpt_and_save(cluster20, 'results/chi_evaluations/prompt_evaluations.txt')
    # ---------------------------------------------
    # cluster20 = cluster(df, n_clusters=20)
    # full_context_prompt(cluster20, num_samples=1)
    # x_clusters_prompt(cluster20, x=2, num_samples_per_cluster=5)
    # x_clusters_prompt(cluster20, x=5, num_samples_per_cluster=5)
    # x_clusters_prompt(cluster20, x=10, num_samples_per_cluster=5)
    # ---------------------------------------------
    cluster20 = cluster(df, n_clusters=20)
    samples = []
    num_clusters = df['cluster'].nunique()
    for cluster_id in range(num_clusters): 
        sample = df[df['cluster'] == cluster_id].sample(n=1)
        samples.append(sample)
    # manual_scenario(samples)
    # semi_auto_scenario(samples)
    auto_scenario(samples)
    auto_no_theme_scenario(samples)


