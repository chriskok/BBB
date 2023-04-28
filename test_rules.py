from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from itertools import product
from pywsd.lesk import simple_lesk
from pywsd import disambiguate
from pywsd.similarity import max_similarity as maxsim

sent1 = "Because then we can them in different areas, rather than hardcoding them to one area."
sent2 = "They can help create designs that will be consistent across different pages or projects."
sent3 = "it creates a consistent interface that can be easily changed instead of changing each individual instance manually."

sent4 = "We want to execute some instructions without blocking other lines of code."
sent5 = "The bible represents christianity's moral code."
sent6 = "Here is my secret code!"
sent7 = "Did you write this Python code?"

print(lesk(sent1, 'reuse'))
print(lesk(sent2, 'consistent'))
print(lesk(sent3, 'manual'))
print(lesk(sent3, 'manually'))

print(lesk(sent4, 'code').definition())
print(lesk(sent5, 'code').definition())
print(lesk(sent6, 'code').definition())
print(lesk(sent7, 'code').definition())

print(simple_lesk(sent4, 'code').definition())
print(simple_lesk(sent5, 'code').definition())
print(simple_lesk(sent6, 'code').definition())
print(simple_lesk(sent7, 'code').definition())

print(disambiguate(sent4, algorithm=maxsim, similarity_option='wup', keepLemmas=True))
# print(disambiguate(sent5, algorithm=maxsim, similarity_option='wup', keepLemmas=True))
print(disambiguate(sent6, algorithm=maxsim, similarity_option='wup', keepLemmas=True))
print(disambiguate(sent7, algorithm=maxsim, similarity_option='wup', keepLemmas=True))

# for ss in wn.synsets('consistent'):
#     print(ss, ss.definition())


allsyns1 = set(ss for word in sent1.split(' ') for ss in wn.synsets(word))
allsyns2 = set(ss for word in sent2.split(' ') for ss in wn.synsets(word))
# allsyns1 = set([lesk(sent1, 'reuse')])
# allsyns2 = set([lesk(sent2, 'consistent')])
# best = max((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))
# print(f"BEST: {best}")
# print(list((wn.wup_similarity(s1, s2) or 0, s1, s2, s1.definition(), s2.definition()) for s1, s2 in product(allsyns1, allsyns2) if wn.wup_similarity(s1, s2) > 0.7))
print(list((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2) if wn.wup_similarity(s1, s2) > 0.7))
