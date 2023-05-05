from django import forms
from .models import Answer

class BuildingBlocksForm(forms.Form):

    rule_type_selection = forms.ChoiceField(choices = (
        ('pick', "-- Pick One --"), 
        ('keyword_rule', "Keyword Similarity"), 
        ('sentence_rule', "Sentence Similarity"), 
        ('length_rule', "Answer Length"),
    ))

    # keyword rule form
    keyword = forms.CharField(required=False, widget=forms.TextInput(attrs={'class':'form-control','placeholder':'keyword','onkeyup':'highlight()'}))
    similarity_input_attrs = {'type':'range', 'id':"keywordSimilarityInput", 'name':"keywordSimilarityInput", 'step': '0.1', 'min': '0.3', 'max': '1.0',
                               'value': '1.0', 'oninput':"keyword_similarity_choice.value=keywordSimilarityInput.value"}
    keyword_similarity = forms.FloatField(widget=forms.NumberInput(attrs=similarity_input_attrs), label='Similarity to Keyword', min_value=0.3, max_value=1.0)

    # sentence similarity form
    sentence = forms.CharField(required=False, widget=forms.Textarea(attrs={'rows': 3}))
    sentence_similarity_input_attrs = {'type':'range', 'id':"sentenceSimilarityInput", 'name':"sentenceSimilarityInput", 'step': '0.1', 'min': '0.3', 'max': '1.0',
                               'value': '0.6', 'oninput':"sentence_similarity_choice.value=sentenceSimilarityInput.value"}
    sentence_similarity = forms.FloatField(widget=forms.NumberInput(attrs=sentence_similarity_input_attrs), label='Similarity to Sentence', min_value=0.3, max_value=1.0)
    sentence_similarity_method = forms.ChoiceField(choices = (('sbert', "SBert"), ('spacy', "Spacy"), ('tfidf', "TF-IDF")))

    # answer length form
    length_type = forms.ChoiceField(choices = (('word', "Word Count"), ('char', "Character Count")))
    answer_length = forms.IntegerField(required=False, )

    # meta
    rule_polarity = forms.ChoiceField(choices = (('positive', "Positive ✔️"), ('negative', "Negative ❌")))
    positive_examples = forms.CharField(required=False)
    negative_examples = forms.CharField(required=False)
