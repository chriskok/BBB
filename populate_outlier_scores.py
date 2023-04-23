from blocks.models import *
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
modelPath = 'all-MiniLM-L6-v2'
model = SentenceTransformer(modelPath)

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
    return_df = return_df.sort_values(by=['score'], ascending=False)
    return return_df

questions = Question.objects.all()
for q in questions:
    df = pd.DataFrame(list(q.answer_set.values()))
    df = outlier_score(df)
    print(df.shape)
    # student_id_list = df["student_id"].values.tolist()

