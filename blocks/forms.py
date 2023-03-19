from django import forms
from .models import Answer

class BuildingBlocksForm(forms.Form):
    keyword = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control','placeholder':'keyword','onkeyup':'highlight()'}))

    similarity_input_attrs = {'type':'range', 'id':"similarityInput", 'name':"similarityInput", 'step': '0.1', 'min': '0.3', 'max': '1.0', 'value': '1.0', 'oninput':"similarity_choice.value=similarityInput.value"}
    similarity = forms.FloatField(widget=forms.NumberInput(attrs=similarity_input_attrs), label='Similarity to Keyword', min_value=0.3, max_value=1.0)