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

# clean sentences (lowercase, remove punctuation)
sentence_set = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in ori_sentence_set]

# remove stopwords with nltk
stop_words = set(stopwords.words('english'))
sentence_set = [' '.join([word for word in sentence.split() if word not in stop_words]) for sentence in sentence_set]

# identify all common words in the sentences
common_words = set.intersection(*map(set, map(str.split, sentence_set)))
print(common_words)

# identify all parts of speech in the sentences
pos_set = [nltk.pos_tag(word_tokenize(sentence.lower())) for sentence in ori_sentence_set]
print(pos_set)

# get stems of words in common_words
stemmer = nltk.stem.PorterStemmer()
stemmed_common_words = [stemmer.stem(word) for word in common_words]
print(stemmed_common_words)

# get synonyms of words in common_words
synonyms = []
for word in common_words:
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
synonyms = set(synonyms)
print(synonyms)

# get named entities in the sentences
named_entities = []
for sentence in ori_sentence_set:
    for chunk in nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence))):
        if hasattr(chunk, 'label'):
            named_entities.append(' '.join(c[0] for c in chunk))
named_entities = set(named_entities)
print(named_entities)
