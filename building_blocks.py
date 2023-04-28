
import pandas as pd
import re

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet

import spacy
from spacy import displacy

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

nlp = spacy.load('en_core_web_lg')  # if not downloaded, run: python -m spacy download en_core_web_lg
modelPath = 'all-MiniLM-L6-v2'
model = SentenceTransformer(modelPath)
sentiment_pipeline = pipeline("sentiment-analysis")  # TODO: download locally so we can run the app without Wi-Fi

# ================================== #
#            PREPROCESSING           #
# ================================== #

def load_data():
    named_df = pd.read_excel('data/ui_final/final_B_named.xlsx', header=0, engine='openpyxl')
    score_df = pd.read_excel('data/ui_final/493_Final-B_scores.xlsx', header=0, engine='openpyxl')
    full_df = named_df.merge(score_df, on=['First Name', 'Last Name', 'SID', 'Email'], how='left')

    # full_df = full_df.dropna()

    return full_df

# TODO: include other steps 
# e.g. https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk

# ================================== #
#           BUILDING BLOCKS          #
# ================================== #

# Building Block 1 - Specific Keyword/Phrase
def specific_keyword(df, keyword):

    data = df["answer_text"]

    mask = data.str.contains(keyword, case=False, na=False)  # can use regex too

    return df[mask]

# Building Block 2 - Similar Keyword/Phrase
def get_synonyms(keyword):
    synonyms = []
    for syn in wordnet.synsets(keyword):
        for lm in syn.lemmas():
            synonyms.append(lm.name().replace('_', " "))  # adding into synonyms
    return list(set(synonyms))

def similar_keyword_slow(df, keyword, sim_score_threshold=0.7, n_return_threshold=None):
    synonyms = get_synonyms(keyword) # TODO: use synonyms on top of nlp sim score

    data = df["answer_text"].apply(str).tolist()

    matching_words = []
    for idx, data_row in enumerate(data):
        words = keyword.lower() + " " + str(data_row).lower()

        tokens = nlp(words)
        
        best_word = ""
        best_score = 0.0
        for token in tokens[1:]:
            # Printing the following attributes of each token.
            # text: the word string, has_vector: if it contains
            # a vector representation in the model, 
            # vector_norm: the algebraic norm of the vector,
            # is_oov: if the word is out of vocabulary.
            # print(token.text, token.has_vector, token.vector_norm, token.is_oov)
            curr_sim = tokens[0].similarity(token)
            if (curr_sim > best_score):
                best_score = curr_sim
                best_word = token.text

        # print("Best word: {}, best score: {}".format(best_word, best_score))
        matching_words.append({'index': idx, 'word': best_word, 'score': best_score, 'student_id': df.iloc[idx]['student_id'], 
            "answer_text": df.iloc[idx]["answer_text"], 'assigned_grade': df.iloc[idx]['assigned_grade']})
    
    matching_words = sorted(matching_words, key=lambda x: x['score'], reverse=True)

    if(n_return_threshold): matching_words = matching_words[0:n_return_threshold]

    # for match in matching_words[0:5]:
    #     i = match['index']
    #     print("Score: {:.4f} - Word: {} - Sentence: {} (file: {})\n".format(match['score'], match['word'], match["answer_text"], match['student_id']))
    
    return_df = pd.DataFrame(matching_words)
    if(not return_df.empty): return_df = return_df[return_df['score'] >= sim_score_threshold] 

    return return_df

def similar_keyword(df, keyword, sim_score_threshold=0.7, n_return_threshold=None):
    synonyms = get_synonyms(keyword) # TODO: use synonyms on top of nlp sim score

    cleaned_data = df['answer_text'].str.lower().str.replace('[^\w\s]','')
    data = cleaned_data.apply(str).tolist()

    unique_words = set([keyword])
    cleaned_data.str.split().apply(unique_words.update)
    unique_words_as_str = keyword.lower() + " " + ' '.join(unique_words)
    tokens = nlp(unique_words_as_str)
    word_similarities = {}
    for token in tokens[1:]:
        curr_sim = tokens[0].similarity(token)
        word_similarities[str(token)] = curr_sim

    matching_words = []
    for idx, data_row in enumerate(data):
        words_in_this_row = str(data_row).lower().replace('[^\w\s]','').split()

        # for each word in words_in_this_row, find the most similar word in word_similarities
        best_word = ""
        best_score = 0.0
        for word in words_in_this_row:
            if word in word_similarities:
                if word_similarities[word] > best_score:
                    best_score = word_similarities[word]
                    best_word = word

        matching_words.append({'index': idx, 'word': best_word, 'score': best_score, 'student_id': df.iloc[idx]['student_id'],
            "answer_text": df.iloc[idx]["answer_text"], 'assigned_grade': df.iloc[idx]['assigned_grade']})
    
    matching_words = sorted(matching_words, key=lambda x: x['score'], reverse=True)

    if(n_return_threshold): matching_words = matching_words[0:n_return_threshold]
    
    return_df = pd.DataFrame(matching_words)
    if(not return_df.empty): return_df = return_df[return_df['score'] >= sim_score_threshold] 

    return return_df

# Building Block 3 - Similar Sentence
# methods: sbert, spacy, gensim
def similar_sentence(df, sentence, sim_score_threshold=0.7, n_return_threshold=None, method='sbert'):

    sentences = [sentence] + df["answer_text"].apply(str).tolist()

    if (method == 'sbert'):
        #Compute embeddings
        embeddings = model.encode(sentences, convert_to_tensor=True)

        #Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.cos_sim(embeddings, embeddings)
        chosen_scores = cosine_scores[0]

    elif (method == 'spacy'):
        embeddings = [nlp(x) for x in sentences]
        chosen_scores = [embeddings[0].similarity(x) for x in embeddings]

    elif (method == 'tfidf'):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        chosen_scores = cosine_sim[0]

    # TODO: Add gensim method as well, https://datascience.stackexchange.com/questions/23969/sentence-similarity-prediction

    return_df = df.copy()
    return_df['score'] = chosen_scores[1:]  # assign all scores to df, while removing the first sentence that we added

    return_df = return_df.sort_values(by=['score'], ascending=False)
    if(n_return_threshold): return_df = return_df.head(n_return_threshold)
    if(not return_df.empty): return_df = return_df[return_df['score'] > sim_score_threshold] 

    return return_df

# Building Block 3.5 - Similar Sentence by Index
def similar_sentence_by_index(df, sentence_index, sim_score_threshold=0.7, n_return_threshold=None, method='sbert'):

    sentences = df["answer_text"].apply(str).tolist()

    if (method == 'sbert'):
        #Compute embeddings
        embeddings = model.encode(sentences, convert_to_tensor=True)

        #Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.cos_sim(embeddings, embeddings)
        chosen_scores = cosine_scores[sentence_index]

    elif (method == 'spacy'):
        embeddings = [nlp(x) for x in sentences]
        chosen_scores = [embeddings[sentence_index].similarity(x) for x in embeddings]

    elif (method == 'tfidf'):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        chosen_scores = cosine_sim[sentence_index]

    # TODO: Add gensim method as well, https://datascience.stackexchange.com/questions/23969/sentence-similarity-prediction

    return_df = df.copy()
    return_df['score'] = chosen_scores
    return_df = return_df.drop([sentence_index])

    return_df = return_df.sort_values(by=['score'], ascending=False)
    if(n_return_threshold): return_df = return_df.head(n_return_threshold)
    if(not return_df.empty): return_df = return_df[return_df['score'] > sim_score_threshold] 

    return return_df

# Building Block 4 - Negation Detection
def filter_by_negation(df, is_negative=True):
    # NOTE: Some additional details we could add to negation
    # https://pypi.org/project/negspacy/ - find out which items specifically are negated or negative in text
    # https://medium.com/@MansiKukreja/clinical-text-negation-handling-using-negspacy-and-scispacy-233ce69ab2ac
    # https://stackoverflow.com/questions/54849111/negation-and-dependency-parsing-with-spacy - identify the specific tokens and heads

    def negation_check(x):
        doc = nlp(x)
        for token in doc:
            is_neg = (token.dep_ == 'neg')
            if is_neg:
                break
        
        return is_neg

    df['negation_check'] = df.apply(lambda row : negation_check(row['answer_text']), axis=1)

    return df[df['negation_check'] == is_negative] 

# Building Block 5 - Sentiment Analysis
def sentiment_analysis(df, label = "POSITIVE", score_threshold=0.7):
    sentences = df["answer_text"].apply(str).tolist()

    sentiments = sentiment_pipeline(sentences)
    for idx, sent in enumerate(sentiments):
        for column_name in df.columns:
            if (column_name == 'label'): continue
            sent[column_name] = df.iloc[idx][column_name]

    return_df = pd.DataFrame([x for x in sentiments if x['label'] == label.upper()])
    if(not return_df.empty): return_df = return_df[return_df['score'] > score_threshold] 

    return return_df

# Building Block 6 - Specific Named Entities
def get_named_entities(sentence_index, sentences):
    curr_sentence = nlp(sentences[sentence_index])
    print(curr_sentence)
    for word in curr_sentence.ents:
        print(word.text, word.label_)

# Available NER tags: 
# ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL',
#  'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
def filter_named_entities(df, tag):
    filtered_list = []
    sentences = df["answer_text"].apply(str).tolist()

    for idx, sentence in enumerate(sentences):
        curr_sentence = nlp(sentence)
        label_check = [x for x in curr_sentence.ents if x.label_ == tag.upper()]
        if label_check: filtered_list.append(idx)
    
    return df.iloc[filtered_list]

# Building Block 7 - Order/Structure
# TODO: add logic for breaking down order/structure (multiple sentences, proposed order, etc.). Potentially with OPEN-AI's GPT

# Building Block 8 - Question Detection
# ref: https://stackoverflow.com/questions/49100615/nltk-detecting-whether-a-sentence-is-interrogative-or-not
question_pattern = ["do i", "do you", "what", "who", "is it", "why","would you", "how","is there",
                    "are there", "is it so", "is this true" ,"to know", "is that true", "are we", "am i", 
                   "question is", "tell me more", "can i", "can we", "tell me", "can you explain",
                   "question","answer", "questions", "answers", "ask"]

helping_verbs = ["is","am","can", "are", "do", "does"]

def question_check(question):
    question = question.lower().strip()

    # check if any of pattern exist in sentence
    for pattern in question_pattern:
        is_ques  = pattern in question
        if is_ques:
            break

    # there could be multiple sentences so divide the sentence
    sentence_arr = question.split(".")
    for sentence in sentence_arr:
        if len(sentence.strip()):
            # if question ends with ? or start with any helping verb
            # word_tokenize will strip by default
            first_word = nltk.word_tokenize(sentence)[0]
            if sentence.endswith("?") or first_word in helping_verbs:
                is_ques = True
                break
    return is_ques    

def filter_by_question(df, is_question=True):

    df['question_check'] = df.apply(lambda row : question_check(row['answer_text']), axis=1)

    return df[df['question_check'] == is_question] 

# Building Block 9 - Causation detection
# TODO: add logic

# Building Block 10 - Answer Length
def answer_length(df, length, length_type="word"):
    return_df = df.copy()

    if (length_type == "word"): return_df['length'] = return_df['answer_text'].str.split().str.len()
    else: return_df['length'] = return_df['answer_text'].str.len()

    if(not return_df.empty): return_df = return_df[return_df['length'] <= length] 

    return return_df

# Building Block 11 - Similarity to Question
# TODO: add logic

# Building Block 12 - Number of different keypoints (too scattered?)
# TODO: add logic

# ================================== #
#              EXECUTION             #
# ================================== #

def write_results(df, title, write_mode="a", filename="test_results"):
    f = open("text_files/{}.txt".format(filename), write_mode)

    f.write('\n\n{}\n'.format(title.upper()))

    dfAsString = df.to_string(header=True, index=False)
    f.write(dfAsString)
    # for index, row in df.iterrows():
    #     f.write(str(row["student_id"]) + ".\t" + str(row["answer_text"]) + "\n")

    f.close()

def cluster_answers(df, num_clusters=6):
    sentences = df["answer_text"].to_list()
    embeddings = model.encode(sentences, convert_to_tensor=True)

    clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(embeddings)
    df['cluster'] = clustering.labels_

    return df.sort_values(by=['cluster']).reset_index(drop=True)

def merge_results(left, right, how='inner'):
    # how: One of 'left', 'right', 'outer', 'inner', 'cross'. Defaults to inner.
    merged = pd.merge(left, right, on=["student_id", "answer_text", 'assigned_grade'], how=how)
    return merged[['student_id', "answer_text", "assigned_grade"]]

def invert_results(ori_df, filtered_df):
    inverse_df = pd.merge(ori_df, filtered_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    return inverse_df[['student_id', "answer_text", "assigned_grade"]]

def main():
    df = load_data()
    df = df.rename(columns={"Filename": 'student_id', "Q11":"answer_text", "11 (2.0 pts)":"assigned_grade"})
    df = df[['student_id', "answer_text", "assigned_grade"]]

    df = cluster_answers(df)
    write_results(df, 'unfiltered', write_mode='w')

    spe_key_df = specific_keyword(df, 'developer')
    write_results(spe_key_df, 'filtered_by_specific_keyword - developer')

    sim_key_df = similar_keyword(df, 'developer')
    write_results(sim_key_df, 'filtered_by_similar_keyword - developer')

    sent_idx = 1
    sim_sen_df = similar_sentence(df, sent_idx, method='sbert', sim_score_threshold=0.5)
    write_results(sim_sen_df, 'filtered_by_similar_sentence_sbert - {}'.format(df["answer_text"].iloc[sent_idx]))

    sim_sen_df = similar_sentence(df, sent_idx, method='spacy', sim_score_threshold=0.85)
    write_results(sim_sen_df, 'filtered_by_similar_sentence_spacy - {}'.format(df["answer_text"].iloc[sent_idx]))

    sim_sen_df = similar_sentence(df, sent_idx, method='tfidf', sim_score_threshold=0.2)
    write_results(sim_sen_df, 'filtered_by_similar_sentence_tfidf - {}'.format(df["answer_text"].iloc[sent_idx]))

    filtered_df = filter_by_negation(df)
    write_results(filtered_df, 'filtered_by_negation')

    sen_ana_df = sentiment_analysis(df, "POSITIVE")
    write_results(sen_ana_df, 'filtered_by_positive_sentiment')

    # get_named_entities(0, ["The Indian Space Research Organisation or is the national space agency of India, headquartered in Bengaluru."])
    # get_named_entities(0, data_list)

    filtered_df = filter_named_entities(df, 'CARDINAL')
    write_results(filtered_df, 'filtered_by_cardinal')

    filtered_df = filter_by_question(df)
    write_results(filtered_df, 'filtered_by_question')

if __name__ == "__main__":
    main()