from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from itertools import product

sent1 = "Because then we can reuse them in different areas, rather than hardcoding them to one area."
sent2 = "They can help create designs that will be consistent across different pages or projects."

print(lesk(sent1, 'reuse'))
print(lesk(sent2, 'consistent'))

for ss in wn.synsets('consistent'):
    print(ss, ss.definition())


allsyns1 = set(ss for word in sent1.split(' ') for ss in wn.synsets(word))
allsyns2 = set(ss for word in sent2.split(' ') for ss in wn.synsets(word))
# allsyns1 = set([lesk(sent1, 'reuse')])
# allsyns2 = set([lesk(sent2, 'consistent')])
# best = max((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))
# print(f"BEST: {best}")
# print(list((wn.wup_similarity(s1, s2) or 0, s1, s2, s1.definition(), s2.definition()) for s1, s2 in product(allsyns1, allsyns2) if wn.wup_similarity(s1, s2) > 0.7))
print(list((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2) if wn.wup_similarity(s1, s2) > 0.7))
