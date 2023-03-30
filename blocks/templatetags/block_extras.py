from django import template

register = template.Library()

@register.filter(name="get_item")
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter(name="filter_chatgpt_by_method")
def filter_chatgpt_by_method(queryset, method):
    return queryset.filter(prompt_type=method).first()

@register.simple_tag(name='filter_chatgpt')
def filter_chatgpt(queryset, method, model):
    return queryset.filter(prompt_type=method, openai_model=model).first()