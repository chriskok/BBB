
import pandas as pd
import re

import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('popular')
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk
from pywsd.lesk import simple_lesk
stemmer = nltk.stem.PorterStemmer()

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
# sentiment_pipeline = pipeline("sentiment-analysis")  # TODO: download locally so we can run the app without Wi-Fi

import openai
from my_secrets import my_secrets
openai_key = my_secrets.get('openai_key')
openai.api_key = openai_key

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
#           KEYWORD BLOCKS           #
# ================================== #

# Building Block 1 - Specific Keyword/Phrase
def specific_keyword(df, keyword):

    data = df["answer_text"]

    mask = data.str.contains(keyword, case=False, na=False)  # can use regex too

    return df[mask]

# Building Block 2 - Similar Keyword/Phrase
def get_synonyms(keyword, for_regex=False):
    synonyms = []
    for syn in wn.synsets(keyword):
        for lm in syn.lemmas():
            lemma = f"\\b{lm.name().replace('_', ' ')}\\b" if for_regex else lm.name().replace('_', " ") 
            synonyms.append(lemma)  # adding into synonyms
    return list(set(synonyms))

def get_definitions(keyword):
    definitions = []
    for syn in wn.synsets(keyword):
        definitions.append(syn.definition())
    return list(set(definitions))

def get_synsets(keyword):
    synsets = []
    for syn in wn.synsets(keyword):
        synsets.append(syn)
    return list(set(synsets))

def get_synset_count(cleaned_data, keyword):
    data = cleaned_data.apply(str).tolist()
    synset_count = {}
    for synset in get_synsets(keyword):
        synset_count[synset] = 0
    for idx, data_row in enumerate(data):
        if keyword in data_row:
            curr_synset = simple_lesk(data_row, keyword)
            if curr_synset in synset_count:
                synset_count[curr_synset] += 1
            else:
                synset_count[curr_synset] = 1
    
    # sort synset_count by value
    synset_count = {k: v for k, v in sorted(synset_count.items(), key=lambda item: item[1], reverse=True)}
    return synset_count

def get_word_similarities(cleaned_data, keyword):
    unique_words = set([keyword])
    cleaned_data.str.split().apply(unique_words.update)
    unique_words_as_str = keyword.lower() + " " + ' '.join(unique_words)
    tokens = nlp(unique_words_as_str)
    word_similarities = {}
    for token in tokens[1:]:
        curr_sim = tokens[0].similarity(token)
        word_similarities[str(token)] = curr_sim

    return word_similarities

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

    cleaned_data = df['answer_text'].str.lower().str.replace('[^\w\s]','')
    data = cleaned_data.apply(str).tolist()

    word_similarities = get_word_similarities(cleaned_data, keyword)

    matching_words = []
    for idx, data_row in enumerate(data):
        words_in_this_row = str(data_row).lower().split()

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

# ================================== #
#          SENTENCE BLOCKS           #
# ================================== #

# Building Block 3 - Similar Sentence

stop_words = set(stopwords.words('english'))
def remove_stopwords(sentence):
    return ' '.join([word for word in sentence.split() if word not in stop_words])

def remove_punctuation(sentence):
    return re.sub(r'[^\w\s]', '', sentence.lower())

def clean_sentence(sentence):
    return_sentence = remove_punctuation(sentence)
    return_sentence = remove_stopwords(return_sentence)
    return return_sentence

def sentence_pattern_breakdown(positive_examples, negative_examples, pattern_limit=3):
    def process_sentences(ori_sentence_set, verbose=False):
        # clean sentences (lowercase, remove punctuation) and remove stop words
        sentence_set = [clean_sentence(sentence) for sentence in ori_sentence_set]

        # identify all common words in the sentences
        # common_words = set.intersection(*map(set, map(str.split, sentence_set))) if sentence_set else set()
        common_words = {}
        for sentence in sentence_set:
            curr_words = set(sentence.split())
            for word in curr_words:
                if word in common_words:
                    common_words[word] += 1
                else:
                    common_words[word] = 1
        # remove words that appear only once
        common_words = {k: v for k, v in common_words.items() if v > 1}
        if(verbose): print(f"common_words: {common_words}")

        # get stems of words in common_words
        stemmed_common_words = [stemmer.stem(word) for word in common_words]
        if(verbose): print(f"stemmed_common_words: {stemmed_common_words}")

        # get synonyms of words in common_words
        synonyms = []
        for word in common_words:
            for syn in wn.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())
        synonyms = set(synonyms)
        if(verbose): print(f"synonyms: {synonyms}")

        # get named entities in the sentences
        named_entities = []
        for sentence in ori_sentence_set:
            for chunk in nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence))):
                if hasattr(chunk, 'label'):
                    named_entities.append((' '.join(c[0] for c in chunk), chunk.label()))
        named_entities = set(named_entities)
        named_entities = dict((x, y) for x, y in named_entities)
        if(verbose): print(f"named_entities: {named_entities}")

        return sentence_set, common_words, stemmed_common_words, synonyms, named_entities

    # recursively process the next string in the list
    patterns = {}

    def add_pattern(pattern, depth, negative=False, wildcard=False):
        if (wildcard): depth = 0
        if pattern in patterns:
            patterns[pattern] += 1 * (depth * 0.5) if not negative else -2 * (depth * 0.5)
        else:
            patterns[pattern] = 1 * (depth * 0.5) if not negative else -2 * (depth * 0.5)

    def process_next_string(word, tag, pos_set, ori_i, i, current_pattern):
        
        # stop if we've reached the end of the list
        if i == len(pos_set) - 1:
            return current_pattern
        
        # stop if we've reached the pattern limit
        curr_depth = i - ori_i + 1
        if curr_depth > pattern_limit:
            return current_pattern
        
        word_l = word.lower()
        matched = False
        
        # check if word in common words (or a stem of that)
        if stemmer.stem(word_l) in stemmed_common_words:
            matched = True
            new_pattern_and = f"[{word_l}]" if current_pattern == "" else current_pattern + "+" + f"[{word_l}]"
            new_pattern_or = f"[{word_l}]" if current_pattern == "" else current_pattern + "|" + f"[{word_l}]"
            new_pattern_array = [new_pattern_and, new_pattern_or] if not current_pattern.endswith("*") else [new_pattern_and]
            for pattern in new_pattern_array:
                add_pattern(pattern, curr_depth)
                process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)
        
        if stemmer.stem(word_l) in n_stemmed_common_words:
            matched = True
            new_pattern_and = f"[{word_l}]" if current_pattern == "" else current_pattern + "+" + f"[{word_l}]"
            new_pattern_or = f"[{word_l}]" if current_pattern == "" else current_pattern + "|" + f"[{word_l}]"
            new_pattern_array = [new_pattern_and, new_pattern_or] if not current_pattern.endswith("*") else [new_pattern_and]
            for pattern in new_pattern_array:
                add_pattern(pattern, curr_depth, negative=True)
                process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

        # check if word is a synonym of a common word
        if word_l in synonyms:
            matched = True
            new_pattern_and = f"({word_l})" if current_pattern == "" else current_pattern + "+" + f"({word_l})"
            new_pattern_or = f"({word_l})" if current_pattern == "" else current_pattern + "|" + f"({word_l})"
            new_pattern_array = [new_pattern_and, new_pattern_or] if not current_pattern.endswith("*") else [new_pattern_and]
            for pattern in new_pattern_array:
                add_pattern(pattern, curr_depth)
                process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

        if word_l in n_synonyms:
            matched = True
            new_pattern_and = f"({word_l})" if current_pattern == "" else current_pattern + "+" + f"({word_l})"
            new_pattern_or = f"({word_l})" if current_pattern == "" else current_pattern + "|" + f"({word_l})"
            new_pattern_array = [new_pattern_and, new_pattern_or] if not current_pattern.endswith("*") else [new_pattern_and]
            for pattern in new_pattern_array:
                add_pattern(pattern, curr_depth, negative=True)
                process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

        # check if word is a named entity
        if word in named_entities:
            matched = True
            new_pattern_and = f"({word})" if current_pattern == "" else current_pattern + "+" + f"({word})"
            new_pattern_or = f"({word})" if current_pattern == "" else current_pattern + "|" + f"({word})"
            new_pattern_label_and = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "+" + f"${named_entities[word]}"
            new_pattern_label_or = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "|" + f"${named_entities[word]}"
            new_pattern_array = [new_pattern_and, new_pattern_or, new_pattern_label_and, new_pattern_label_or] if not current_pattern.endswith("*") else [new_pattern_and, new_pattern_label_and]
            for pattern in new_pattern_array:
                add_pattern(pattern, curr_depth)
                process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

        if word in n_named_entities:
            matched = True
            new_pattern_and = f"({word})" if current_pattern == "" else current_pattern + "+" + f"({word})"
            new_pattern_or = f"({word})" if current_pattern == "" else current_pattern + "|" + f"({word})"
            new_pattern_label_and = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "+" + f"${named_entities[word]}"
            new_pattern_label_or = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "|" + f"${named_entities[word]}"
            new_pattern_array = [new_pattern_and, new_pattern_or, new_pattern_label_and, new_pattern_label_or] if not current_pattern.endswith("*") else [new_pattern_and, new_pattern_label_and]
            for pattern in new_pattern_array:
                add_pattern(pattern, curr_depth, negative=True)
                process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

        # if nothing matches, attach wildcard symbol
        if (not matched and current_pattern != ""):
            wildcard_pattern = current_pattern + "+" + "*" if not current_pattern.endswith("*") else current_pattern
            add_pattern(wildcard_pattern, curr_depth, negative=False, wildcard=True)
            process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, wildcard_pattern)

        # check if word
        # if tag in patterns:
        #     patterns[tag].append(word)
        # else:
        #     patterns[tag] = [word]

    # identify all parts of speech in the sentences
    # TODO: only take into account the NEGATIVE parts of speech (for negation detection)
    pos_set = [nltk.pos_tag(word_tokenize(sentence)) for sentence in [remove_punctuation(eg) for eg in positive_examples]]
    sentence_set, common_words, stemmed_common_words, synonyms, named_entities = process_sentences(positive_examples)
    n_sentence_set, n_common_words, n_stemmed_common_words, n_synonyms, n_named_entities = process_sentences(negative_examples)

    # iterate through the pos_set to build patterns for matching
    for pos in pos_set:
        for i, (word, tag) in enumerate(pos):
            process_next_string(word, tag, pos, i, i, "")

    # sort patterns dictionary by value & remove all keys that ends with "*"
    patterns = {k: v for k, v in sorted(patterns.items(), key=lambda item: item[1], reverse=True) if not k.endswith("*") and v > 0.0}
    # remove duplicates of patterns with same value
    temp = []
    res = dict()
    for key, val in patterns.items():
        if val not in temp:
            temp.append(val)
            res[key] = val
    # take top 5 patterns
    patterns = dict(list(res.items())[:5])

    return patterns

# convert patterns to regex
# e.g. "[without]+[reloading]+*+(send)" --> ".*\bwithout\b \breloading\b.*(\bsend\b|\bdispatch\b|\bmail\b).*"
def convert_patterns_to_regex(patterns, chosen_answers):
    return_dict = {}
    def add_synonyms(word):
        word = word.replace('(', '').replace(')', '')
        synonyms = [w for w in get_synonyms(word, for_regex=True) if ' ' not in w]
        return "(" + '_'.join(synonyms) + ")"

    def pattern_to_regex(pattern):
        pattern_str = pattern
        pattern_str = pattern_str.replace('[', r'\b').replace(']', r'\b').replace('+', ' ').replace('*', '.*').replace('[', r'\b')
        bracket_words = re.findall(r"\(.*?\)", pattern_str)
        for word in bracket_words:
            pattern_str = pattern_str.replace(word, add_synonyms(word))

        pattern_str = re.sub(r" (\S*?\|\S*?) ", r" (\1) ", pattern_str)
        pattern_str = pattern_str.replace('_','|')
        pattern_str = ".*" + pattern_str + ".*"
        return pattern_str

    for pattern,value in patterns.items():
        regex_pattern = pattern_to_regex(pattern)
        applies_count = 0 
        applies_to = []
        for ans in chosen_answers.values_list('answer_text', flat=True):
            result = re.match(regex_pattern, remove_punctuation(ans))
            if result: 
                applies_count += 1
                applies_to.append(ans)

        return_dict[pattern] = {'regex': regex_pattern, 'points': value, 'applies_to': applies_to, 'applies_count': applies_count}

    # print(pattern_to_regex("[consistent]+*+(design)|[individual]|(home)"))

    return return_dict

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

# ================================== #
#           OPENAI BLOCKS            #
# ================================== #

def prompt_chatgpt(prompt):
    model="gpt-3.5-turbo"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            temperature=0,
        )
        if "error" in response:
            print("OPENAI ERROR: {}".format(response))
            return "ERROR"
        else:
            return response["choices"][0]["message"]["content"]
    except Exception as e: 
        print(e)
        return "ERROR"

def get_question_concepts(question_text):
    prompt = [
                {"role": "system", "content": f"You are an expert teacher in a class, you have the following question in your final exam: {question_text}"},
                {"role": "user", "content": f"Please list all related concepts to this question in a numbered list along with a brief description after each of the points. Follow the format <number>. <concept>: <description>"},
            ]
    chatgpt_response = prompt_chatgpt(prompt)
    return chatgpt_response

def convert_question_concepts(question_concepts):
    # convert numbered list in string to list of strings
    concepts = [x.strip() for x in question_concepts.split('\n') if x.strip() != '']
    return concepts


def too_general(df, question, score_threshold=0.7, concept_threshold=2):

    # sentences = [sentence] + df["answer_text"].apply(str).tolist()

    # if (method == 'sbert'):
    #     #Compute embeddings
    #     embeddings = model.encode(sentences, convert_to_tensor=True)

    #     #Compute cosine-similarities for each sentence with each other sentence
    #     cosine_scores = util.cos_sim(embeddings, embeddings)
    #     chosen_scores = cosine_scores[0]

    # elif (method == 'spacy'):
    #     embeddings = [nlp(x) for x in sentences]
    #     chosen_scores = [embeddings[0].similarity(x) for x in embeddings]

    # elif (method == 'tfidf'):
    #     tfidf_vectorizer = TfidfVectorizer()
    #     tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    #     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    #     chosen_scores = cosine_sim[0]

    # # TODO: Add gensim method as well, https://datascience.stackexchange.com/questions/23969/sentence-similarity-prediction

    # return_df = df.copy()
    # return_df['score'] = chosen_scores[1:]  # assign all scores to df, while removing the first sentence that we added

    # return_df = return_df.sort_values(by=['score'], ascending=False)
    # if(n_return_threshold): return_df = return_df.head(n_return_threshold)
    # if(not return_df.empty): return_df = return_df[return_df['score'] > sim_score_threshold] 

    return_df = df.copy()

    return return_df

# ================================== #
#            OTHER BLOCKS            #
# ================================== #

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

# # Building Block 5 - Sentiment Analysis
# def sentiment_analysis(df, label = "POSITIVE", score_threshold=0.7):
#     sentences = df["answer_text"].apply(str).tolist()

#     sentiments = sentiment_pipeline(sentences)
#     for idx, sent in enumerate(sentiments):
#         for column_name in df.columns:
#             if (column_name == 'label'): continue
#             sent[column_name] = df.iloc[idx][column_name]

#     return_df = pd.DataFrame([x for x in sentiments if x['label'] == label.upper()])
#     if(not return_df.empty): return_df = return_df[return_df['score'] > score_threshold] 

#     return return_df

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

    # sen_ana_df = sentiment_analysis(df, "POSITIVE")
    # write_results(sen_ana_df, 'filtered_by_positive_sentiment')

    # get_named_entities(0, ["The Indian Space Research Organisation or is the national space agency of India, headquartered in Bengaluru."])
    # get_named_entities(0, data_list)

    filtered_df = filter_named_entities(df, 'CARDINAL')
    write_results(filtered_df, 'filtered_by_cardinal')

    filtered_df = filter_by_question(df)
    write_results(filtered_df, 'filtered_by_question')

if __name__ == "__main__":
    main()