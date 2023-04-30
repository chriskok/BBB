import nltk
from nltk.tokenize import word_tokenize
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from itertools import product
from pywsd.lesk import simple_lesk
from pywsd import disambiguate
from pywsd.similarity import max_similarity as maxsim
import re
stemmer = nltk.stem.PorterStemmer()

# ================================== #
#          KEYWORD MATCHING          #
# ================================== #

# sent1 = "Because then we can them in different areas, rather than hardcoding them to one area."
# sent2 = "They can help create designs that will be consistent across different pages or projects."
# sent3 = "it creates a consistent interface that can be easily changed instead of changing each individual instance manually."

# sent4 = "We want to execute some instructions without blocking other lines of code."
# sent5 = "The bible represents christianity's moral code."
# sent6 = "Here is my secret code!"
# sent7 = "Did you write this Python code?"

# print(lesk(sent1, 'reuse'))
# print(lesk(sent2, 'consistent'))
# print(lesk(sent3, 'manual'))
# print(lesk(sent3, 'manually'))

# print(lesk(sent4, 'code').definition())
# print(lesk(sent5, 'code').definition())
# print(lesk(sent6, 'code').definition())
# print(lesk(sent7, 'code').definition())

# print(simple_lesk(sent4, 'code').definition())
# print(simple_lesk(sent5, 'code').definition())
# print(simple_lesk(sent6, 'code').definition())
# print(simple_lesk(sent7, 'code').definition())

# print(disambiguate(sent4, algorithm=maxsim, similarity_option='wup', keepLemmas=True))
# # print(disambiguate(sent5, algorithm=maxsim, similarity_option='wup', keepLemmas=True))
# print(disambiguate(sent6, algorithm=maxsim, similarity_option='wup', keepLemmas=True))
# print(disambiguate(sent7, algorithm=maxsim, similarity_option='wup', keepLemmas=True))

# # for ss in wn.synsets('consistent'):
# #     print(ss, ss.definition())


# allsyns1 = set(ss for word in sent1.split(' ') for ss in wn.synsets(word))
# allsyns2 = set(ss for word in sent2.split(' ') for ss in wn.synsets(word))
# # allsyns1 = set([lesk(sent1, 'reuse')])
# # allsyns2 = set([lesk(sent2, 'consistent')])
# # best = max((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))
# # print(f"BEST: {best}")
# # print(list((wn.wup_similarity(s1, s2) or 0, s1, s2, s1.definition(), s2.definition()) for s1, s2 in product(allsyns1, allsyns2) if wn.wup_similarity(s1, s2) > 0.7))
# print(list((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2) if wn.wup_similarity(s1, s2) > 0.7))

# ================================== #
#          SENTENCE MATCHING         #
# ================================== #

ori_sentence_set = ['We can change a Google page without reloading it and send, request, and receive data from an AWS server without blocking the rest of your interface.', 
                'We want to be able to make changes to a Google page without reloading it every time. We also want to send, request, and receive data from an AWS server without blocking the interface.', 
                'It changes the Google page without reloading it and can send, request and receive data without blocking the rest of the interface']

negative_sentence_set = ['This is to allow other components when showing the webpage does not get blocked by operations that require long time.',]
# negative_sentence_set = ['This allows for multiple tasks to run at the same time.',
#                          'This is to allow other components when showing the webpage does not get blocked by operations that require long time.',
#                          'This way you dont have to wait for things to happen to update other separate parts of a webpage/app.']

def process_sentences(ori_sentence_set, verbose=False):
    # clean sentences (lowercase, remove punctuation)
    sentence_set = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in ori_sentence_set]

    # remove stopwords with nltk
    stop_words = set(stopwords.words('english'))
    sentence_set = [' '.join([word for word in sentence.split() if word not in stop_words]) for sentence in sentence_set]

    # identify all common words in the sentences
    common_words = set.intersection(*map(set, map(str.split, sentence_set)))
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
pattern_limit = 3

def add_pattern(pattern, depth, negative=False):
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
    
    # check if word in common words (or a stem of that)
    if stemmer.stem(word_l) in stemmed_common_words:
        new_pattern_and = f"[{word_l}]" if current_pattern == "" else current_pattern + "+" + f"[{word_l}]"
        new_pattern_or = f"[{word_l}]" if current_pattern == "" else current_pattern + "|" + f"[{word_l}]"
        for pattern in [new_pattern_and, new_pattern_or]:
            add_pattern(pattern, curr_depth)
            process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)
    
    if stemmer.stem(word_l) in n_stemmed_common_words:
        new_pattern_and = f"[{word_l}]" if current_pattern == "" else current_pattern + "+" + f"[{word_l}]"
        new_pattern_or = f"[{word_l}]" if current_pattern == "" else current_pattern + "|" + f"[{word_l}]"
        for pattern in [new_pattern_and, new_pattern_or]:
            add_pattern(pattern, curr_depth, negative=True)
            process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

    # check if word is a synonym of a common word
    if word_l in synonyms:
        new_pattern_and = f"({word_l})" if current_pattern == "" else current_pattern + "+" + f"({word_l})"
        new_pattern_or = f"({word_l})" if current_pattern == "" else current_pattern + "|" + f"({word_l})"
        for pattern in [new_pattern_and, new_pattern_or]:
            add_pattern(pattern, curr_depth)
            process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

    if word_l in n_synonyms:
        new_pattern_and = f"({word_l})" if current_pattern == "" else current_pattern + "+" + f"({word_l})"
        new_pattern_or = f"({word_l})" if current_pattern == "" else current_pattern + "|" + f"({word_l})"
        for pattern in [new_pattern_and, new_pattern_or]:
            add_pattern(pattern, curr_depth, negative=True)
            process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

    # check if word is a named entity
    if word in named_entities:
        new_pattern_and = f"({word})" if current_pattern == "" else current_pattern + "+" + f"({word})"
        new_pattern_or = f"({word})" if current_pattern == "" else current_pattern + "|" + f"({word})"
        new_pattern_label_and = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "+" + f"${named_entities[word]}"
        new_pattern_label_or = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "|" + f"${named_entities[word]}"
        for pattern in [new_pattern_and, new_pattern_or, new_pattern_label_and, new_pattern_label_or]:
            add_pattern(pattern, curr_depth)
            process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

    if word in n_named_entities:
        new_pattern_and = f"({word})" if current_pattern == "" else current_pattern + "+" + f"({word})"
        new_pattern_or = f"({word})" if current_pattern == "" else current_pattern + "|" + f"({word})"
        new_pattern_label_and = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "+" + f"${named_entities[word]}"
        new_pattern_label_or = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "|" + f"${named_entities[word]}"
        for pattern in [new_pattern_and, new_pattern_or, new_pattern_label_and, new_pattern_label_or]:
            add_pattern(pattern, curr_depth, negative=True)
            process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

    # check if word
    # if tag in patterns:
    #     patterns[tag].append(word)
    # else:
    #     patterns[tag] = [word]

# identify all parts of speech in the sentences
pos_set = [nltk.pos_tag(word_tokenize(sentence)) for sentence in ori_sentence_set]
sentence_set, common_words, stemmed_common_words, synonyms, named_entities = process_sentences(ori_sentence_set)
n_sentence_set, n_common_words, n_stemmed_common_words, n_synonyms, n_named_entities = process_sentences(negative_sentence_set)

# iterate through the pos_set to build patterns for matching
for pos in pos_set:
    for i, (word, tag) in enumerate(pos):
        process_next_string(word, tag, pos, i, i, "")

# sort patterns dictionary by value
patterns = {k: v for k, v in sorted(patterns.items(), key=lambda item: item[1], reverse=True)}
print(patterns)