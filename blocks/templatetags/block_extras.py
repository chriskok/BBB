from django import template
from ..models import *

register = template.Library()

@register.filter(name="get_item")
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter(name="count_parents")
def count_parents(rule_id):
    rule = Rule.objects.get(id=rule_id)
    if (not rule.parent): return ""
    elif (not rule.parent.parent): return "--"
    elif (not rule.parent.parent.parent): return "----"
    elif (not rule.parent.parent.parent.parent): return "------"
    elif (not rule.parent.parent.parent.parent.parent): return "--------"
    else: return "----------"

@register.filter(name="filter_chatgpt_by_method")
def filter_chatgpt_by_method(queryset, method):
    return queryset.filter(prompt_type=method).first()

@register.simple_tag(name='filter_chatgpt')
def filter_chatgpt(queryset, method, model):
    return queryset.filter(prompt_type=method, openai_model=model).last()

@register.filter(name='lookup')
def lookup(value, arg):
    return value[arg]