from django import forms
from .models import Answer, Cluster

class BuildingBlocksForm(forms.Form):

    rule_type_selection = forms.ChoiceField(choices = (
        ('pick', "-- Pick One --"), 
        ('keyword_rule', "Keyword Similarity"), 
        ('sentence_rule', "Sentence Similarity"), 
        ('concept_rule', "Concept Similarity"),
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

    # concept similarity form
    concept_similarity_input_attrs = {'type':'range', 'id':"conceptSimilarityInput", 'name':"conceptSimilarityInput", 'step': '0.1', 'min': '0.3', 'max': '1.0',
                               'value': '0.6', 'oninput':"concept_similarity_choice.value=conceptSimilarityInput.value"}
    concept_similarity = forms.FloatField(widget=forms.NumberInput(attrs=concept_similarity_input_attrs), label='Similarity to Concept', min_value=0.3, max_value=1.0)

    # answer length form
    length_type = forms.ChoiceField(choices = (('word', "Word Count"), ('char', "Character Count")))
    answer_length = forms.IntegerField(required=False, )

    # meta
    rule_polarity = forms.ChoiceField(choices = (('positive', "Positive ✔️"), ('negative', "Negative ❌")))
    positive_examples = forms.CharField(required=False)
    negative_examples = forms.CharField(required=False)


class ClusterGradingForm(forms.ModelForm):
    cluster_feedback = forms.CharField(widget=forms.Textarea(attrs={'rows': 3}), required=False)
    cluster_description = forms.CharField(widget=forms.Textarea(attrs={'rows': 3}), required=False)
    cluster_grade_attrs = {'type': 'number', 'id':"cluster_grade", 'name':"cluster_grade", 'step': '0.1', 'min': '-5', 'max': '5'}
    cluster_grade = forms.FloatField(widget=forms.NumberInput(attrs=cluster_grade_attrs), label='Grade (-5 to 5)', min_value=-5, max_value=5, required=False)

    class Meta:
        model = Cluster
        fields  = ['cluster_name','cluster_grade', 'cluster_feedback', 'cluster_description', ]