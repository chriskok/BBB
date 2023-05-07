import re

# CONVERSION
# e.g. [consistent]+*+(design)|[individual] --> ".*\bconsistent\b.*"
def add_synonyms(word):
    word = word.replace('(', '').replace(')', '')
    words = [word, word+'s', word]  # TODO: replace with actual synonyms from wordnet
    words = [r"\b" + w + r"\b" for w in words]
    return "(" + '_'.join(words) + ")"

pattern_str = "[consistent]+*+(design)|[individual]|(home)"

pattern_str = pattern_str.replace('[', r'\b').replace(']', r'\b').replace('+', ' ').replace('*', '.*').replace('[', r'\b')
bracket_words = re.findall(r"\(.*?\)", pattern_str)
for word in bracket_words:
    pattern_str = pattern_str.replace(word, add_synonyms(word))
# print(pattern_str)

# print(re.findall(r" (\S*?\|\S*?) ", pattern_str))

pattern_str = re.sub(r" (\S*?\|\S*?) ", r" (\1) ", pattern_str)

pattern_str = pattern_str.replace('_','|')
pattern_str = ".*" + pattern_str + ".*"
print(f'FINAL: {pattern_str}')

# MATCHING
pattern_regex = ".*\\bconsistent\\b.*"
test_strs = ["it creates a consistent interface that can be easily changed instead of changing each individual instance manually.", "It helps us because we don't need to recreate the same designs repeatedly. It helps us create consistent and organized designs and quickly too.", "they allow the designer to create multiple versions of a single element, this keeps the format consistent and allows the design team to work better together", "Makes repeated feature to be reused and allows everything to be organized more efficiently"]

for test_str in test_strs:
    result = re.match(pattern_str, test_str)
    if result: print('FOUND!')
    else: print('NOT FOUND!')