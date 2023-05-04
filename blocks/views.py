import re
import pandas as pd
import json
import csv

from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.db.models import Avg, Max, Min, Sum
from django.contrib.auth.decorators import login_required
from django.views.generic.edit import UpdateView

from itertools import product
import building_blocks as bb 
import spacy
nlp = spacy.load('en_core_web_lg')  # if not downloaded, run: python -m spacy download en_core_web_lg

from .models import *
from .forms import *
from .colors import colors

import warnings
warnings.filterwarnings("ignore")

#############################################
#               RULE HELPERS                #
#############################################

def find_element(x, lst):
    res = [i for i, curr_list in enumerate(lst) if curr_list[0] == x]
    return res[0] if res else -1

def add_rule_string(answer, new_rule, rule_string):
    answer.applied_rules.add(new_rule)
    curr_rule_strings = answer.get_rule_strings()

    # If this new rule has a parent, append it right after the parent to preserve order
    if (new_rule.parent): 
        parent_id = new_rule.parent.id
        curr_rule_strings.insert(find_element(parent_id, curr_rule_strings)+1, (new_rule.id, rule_string))
    else:
        curr_rule_strings.append((new_rule.id, rule_string))
    answer.set_rule_strings(curr_rule_strings)
    answer.save()

def similar_keyword_filter(chosen_answers, current_question_obj, keyword, similarity):
    # filters answers that have keyword 
    df = pd.DataFrame(list(chosen_answers.values()))
    df = bb.similar_keyword(df, keyword, sim_score_threshold=similarity)
    relevant_keywords = df.word.unique().tolist()
    student_id_list = df["student_id"].values.tolist()
    filtered_answers = Answer.objects.filter(question=current_question_obj, student_id__in=student_id_list)

    return df, filtered_answers, relevant_keywords

def similar_sentence_filter(chosen_answers, current_question_obj, sentence, similarity, method):
    # filters answers 
    df = pd.DataFrame(list(chosen_answers.values()))
    df = bb.similar_sentence(df, sentence, sim_score_threshold=similarity, method=method)
    student_id_list = df["student_id"].values.tolist()
    filtered_answers = Answer.objects.filter(question=current_question_obj, student_id__in=student_id_list)

    return df, filtered_answers

def answer_length_filter(chosen_answers, current_question_obj, length, length_type):
    # filters answers that have keyword 
    df = pd.DataFrame(list(chosen_answers.values()))
    df = bb.answer_length(df, length, length_type=length_type)
    student_id_list = df["student_id"].values.tolist()
    filtered_answers = Answer.objects.filter(question=current_question_obj, student_id__in=student_id_list)

    return df, filtered_answers

def recursive_rule_child_chain(rule):
    children = rule.rule_set.all()
    if not children: return [], []

    child_list = list(children)
    id_list = list(children.values_list('id', flat=True)) 
    for child in children:
        child_child_list, child_id_list = recursive_rule_child_chain(child)
        child_list.extend(child_child_list)
        id_list.extend(child_id_list)

    return child_list, id_list

def recursive_filtering_chosen_answers(rule, all_answers):
    chosen_answers = all_answers
    if (rule.parent):
        parent_rule = rule.parent
        chosen_answers = recursive_filtering_chosen_answers(parent_rule, all_answers)
    
    # check the content type of this rule and apply filtering accordingly
    if rule.polymorphic_ctype.name == "keyword rule":
        _, return_answers, _ = similar_keyword_filter(chosen_answers, rule.question, rule.keyword, rule.similarity_threshold)
    elif rule.polymorphic_ctype.name == "sentence similarity rule":
        _, return_answers = similar_sentence_filter(chosen_answers, rule.question, rule.sentence, rule.similarity_threshold, rule.method)
    elif rule.polymorphic_ctype.name == "answer length rule":
        _, return_answers = answer_length_filter(chosen_answers, rule.question, rule.length, rule.length_type)
    else:
        return_answers = chosen_answers
    
    return return_answers

def handle_rule_input(form, chosen_answers, current_question_obj):

    parent_string = form.data['parent']
    parent_rule = None
    if parent_string != '-- Parent Rule --': 
        parent_rule = Rule.objects.get(pk = int(parent_string))
        # filter chosen answers by parent rule(s)
        chosen_answers = recursive_filtering_chosen_answers(parent_rule, chosen_answers)

    polarity=form.cleaned_data['rule_polarity']
    polarity_emoji = "✔️" if polarity == "positive" else "❌"

    # KEYWORD RULE
    if (form.cleaned_data['rule_type_selection'] == 'keyword_rule'):
        keyword = form.cleaned_data['keyword']
        similarity = form.cleaned_data['keyword_similarity']

        # filters answers that have keyword 
        df, filtered_answers, relevant_keywords = similar_keyword_filter(chosen_answers, current_question_obj, keyword, similarity)

        # handle keyword rule creation
        new_rule,_ = KeywordRule.objects.get_or_create(question=current_question_obj, parent=parent_rule, keyword=keyword, similarity_threshold=similarity, relevant_keywords=relevant_keywords, polarity=form.cleaned_data['rule_polarity']) 

        # go through each filtered answer and assign the rule and rule strings
        for answer in filtered_answers:
            curr_row = df[df['student_id'] == answer.student_id].iloc[0]
            add_rule_string(answer, new_rule, f"{polarity_emoji} Keyword: {keyword} -> Matched: {curr_row['word']}, Similarity: {curr_row['score']:.2f}")
    
    # SENTENCE SIM RULE
    elif (form.cleaned_data['rule_type_selection'] == 'sentence_rule'):
        sentence = form.cleaned_data['sentence']
        similarity = form.cleaned_data['sentence_similarity']
        method = form.cleaned_data['sentence_similarity_method']

        # filters answers 
        df, filtered_answers = similar_sentence_filter(chosen_answers, current_question_obj, sentence, similarity, method)

        # handle sentence sim rule creation
        new_rule,_ = SentenceSimilarityRule.objects.get_or_create(question=current_question_obj, parent=parent_rule, sentence=sentence, similarity_threshold=similarity, method=method, polarity=form.cleaned_data['rule_polarity']) 

        # go through each filtered answer and assign the rule and rule strings
        for answer in filtered_answers:
            curr_row = df[df['student_id'] == answer.student_id].iloc[0]
            add_rule_string(answer, new_rule, f"{polarity_emoji} Sentence: {sentence} -> Similarity: {curr_row['score']:.2f}")

    # LENGTH RULE
    elif (form.cleaned_data['rule_type_selection'] == 'length_rule'):
        length_type = form.cleaned_data['length_type']
        length = form.cleaned_data['answer_length']

        # filters answers 
        df, filtered_answers = answer_length_filter(chosen_answers, current_question_obj, length, length_type)

        # handle rule creation
        new_rule,_ = AnswerLengthRule.objects.get_or_create(question=current_question_obj, parent=parent_rule, length=length, length_type=length_type, polarity=form.cleaned_data['rule_polarity']) 

        # go through each filtered answer and assign the rule and rule strings
        for answer in filtered_answers:
            curr_row = df[df['student_id'] == answer.student_id].iloc[0]
            add_rule_string(answer, new_rule, f"{polarity_emoji} Length: {length} {length_type}s -> Answer Length: {curr_row['length']}")
    else: 
        print(form.cleaned_data)

#############################################
#                PAGE VIEWS                 #
#############################################

# @login_required
def building_blocks_view(request, q_id, filter=None):
    q_list = Question.objects.all()

    # if the question queried does not exist, get the first Question available
    if Question.objects.filter(pk=q_id).exists():
        current_question_obj = Question.objects.get(pk=q_id)
    else:
        current_question_obj = Question.objects.first()
        q_id = current_question_obj.id

    chosen_answers = Answer.objects.filter(question_id=q_id).order_by('outlier_score')
    answer_count = len(chosen_answers)
    # NOTE: maybe instead of AND, OR, and NOT, we can just either apply a sequential filter or allow MERGING (with logic gates)

    # Handle form input
    if request.method == 'POST':
        form = BuildingBlocksForm(request.POST)
        if form.is_valid():
            handle_rule_input(form, chosen_answers, current_question_obj)

            return HttpResponseRedirect(reverse('building_blocks', args=(q_id,)))
    else:
        form = BuildingBlocksForm()
    
    keywords = list(KeywordRule.objects.filter(question=current_question_obj).values_list('keyword', flat=True))
    rules = Rule.objects.filter(question=current_question_obj)
    orphan_rules = Rule.objects.filter(question=current_question_obj, parent=None)
    color_to_rule = { k.id:v for (k,v) in zip(rules, colors[:len(rules)])} 

    context = {
        "question_obj": current_question_obj,
        "question_exam_id": q_id,
        "question_list": q_list,
        "answers": chosen_answers,
        "answer_count": answer_count,
        "form": form,
        "keywords": keywords,
        "rules": rules,
        "orphan_rules": orphan_rules,
        "color_to_rule": color_to_rule,
    }

    return render(request, "building_blocks.html", context)

# methods: question_only, examples, rubrics, rubrics_and_examples
def chatgpt_view(request, q_id, method='question_only'):
    q_list = Question.objects.all()

    # method = request.GET.get('method', 'question_only')
    model = request.GET.get('model', 'text-davinci-003')

    # if the question queried does not exist, get the first Question available
    if Question.objects.filter(pk=q_id).exists():
        current_question_obj = Question.objects.get(pk=q_id)
    else:
        current_question_obj = Question.objects.first()
        q_id = current_question_obj.id

    chosen_answers = Answer.objects.filter(question_id=q_id)
    answer_count = len(chosen_answers)

    context = {
        "question_obj": current_question_obj,
        "question_exam_id": q_id,
        "question_list": q_list,
        "answers": chosen_answers,
        "answer_count": answer_count,
        "method": method,
        "model": model,
    }

    return render(request, "chatgpt_page.html", context)

def system_reset_view(request, question_id=None):

    # get specific answers for the current question
    chosen_answers = Answer.objects.filter(question_id=question_id).all() if question_id else Answer.objects.all()
    chosen_answers.update(rule_strings="[]") 

    # delete all rules for the current question, iteratively - if not, the deletions will have a foreign key constraint error
    rules = Rule.objects.filter(question_id=question_id).all() if question_id else Rule.objects.all()
    while rules:
        non_parent_rules = rules.filter(rule=None)
        for rule in non_parent_rules:
            rule.delete()
        rules = Rule.objects.filter(question_id=question_id).all() if question_id else Rule.objects.all()

    return JsonResponse({'message': "System Reset!"}) 

def index(request):
  return render(request, "landing_page.html", context={})

#############################################
#               GENERIC VIEWS               #
#############################################

def keywordrule_update(rule, keyword, similarity, chosen_answers, current_question_obj):

    df, filtered_answers, _ = similar_keyword_filter(chosen_answers, current_question_obj, keyword, similarity)

    # go through each filtered answer and assign the rule and rule strings
    for answer in filtered_answers:
        curr_row = df[df['student_id'] == answer.student_id].iloc[0]
        add_rule_string(answer, rule, f"Keyword: {keyword} -> Matched: {curr_row['word']}, Similarity: {curr_row['score']:.2f}")

def sentencesimilarityrule_update(rule, sentence, similarity, method, chosen_answers, current_question_obj):
        
    df, filtered_answers = similar_sentence_filter(chosen_answers, current_question_obj, sentence, similarity, method)

    # go through each filtered answer and assign the rule and rule strings
    for answer in filtered_answers:
        curr_row = df[df['student_id'] == answer.student_id].iloc[0]
        add_rule_string(answer, rule, f"Sentence: {sentence} -> Similarity: {curr_row['score']:.2f}")

def answerlengthrule_update(rule, length, length_type, chosen_answers, current_question_obj):

    df, filtered_answers = answer_length_filter(chosen_answers, current_question_obj, length, length_type)

    # go through each filtered answer and assign the rule and rule strings
    for answer in filtered_answers:
        curr_row = df[df['student_id'] == answer.student_id].iloc[0]
        add_rule_string(answer, rule, f"Length: {length} {length_type}s -> Answer Length: {curr_row['length']}")

def remove_rule_strings(id_array, rule_array, chosen_answers):
    for ans in chosen_answers:
        rule_strings = ans.get_rule_strings()
        new_rule_strings = [x for x in rule_strings if x[0] not in id_array]
        ans.set_rule_strings(new_rule_strings)
        for rule in rule_array:
            ans.applied_rules.remove(rule)
        ans.save()

def update_all_children_rules(child_child_list, chosen_answers, question):

    for rule in child_child_list:
        if rule.polymorphic_ctype.name == "keyword rule":
            # TODO: Make these more efficient by saying we don't have to search past current object as parent, can have param in keywordrule_update
            keywordrule_update(rule, rule.keyword, rule.similarity_threshold, recursive_filtering_chosen_answers(rule, chosen_answers), question)
        elif rule.polymorphic_ctype.name == "sentence similarity rule":
            sentencesimilarityrule_update(rule, rule.sentence, rule.similarity_threshold, rule.method, recursive_filtering_chosen_answers(rule, chosen_answers), question)
        elif rule.polymorphic_ctype.name == "answer length rule":
            answerlengthrule_update(rule, rule.length, rule.length_type, recursive_filtering_chosen_answers(rule, chosen_answers), question)

class KeywordRuleUpdateView(UpdateView):
    model = KeywordRule
    fields = ['keyword', 'similarity_threshold']
    template_name = 'generic_views/rule_update.html'


    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        all_answers = Answer.objects.filter(question_id=self.object.question.id).all()
        df = pd.DataFrame(list(all_answers.values()))

        cleaned_data = df['answer_text'].str.lower().str.replace('[^\w\s]','')
        context['word_similarities'] = bb.get_word_similarities(cleaned_data, self.object.keyword) 
        context['synset_count'] = bb.get_synset_count(cleaned_data, self.object.keyword)
        context['synonyms'] = bb.get_synonyms(self.object.keyword)
        context['similarity_threshold'] = self.object.similarity_threshold
        return context

    def get_success_url(self):
        return reverse_lazy('building_blocks', kwargs={'q_id': self.object.question.id})

    def form_valid(self, form):
        all_answers = Answer.objects.filter(question_id=self.object.question.id).all()
        chosen_answers = recursive_filtering_chosen_answers(self.object, all_answers)
        child_child_list, child_id_list = recursive_rule_child_chain(self.object)

        # remove rule strings for this rule and all child rules from all applied answers
        remove_rule_strings(child_id_list + [self.object.id], child_child_list + [self.object], all_answers)  
        keywordrule_update(self.object, form.cleaned_data['keyword'], form.cleaned_data['similarity_threshold'], chosen_answers, self.object.question)  
        update_all_children_rules(child_child_list, chosen_answers, self.object.question)  

        return super().form_valid(form) 

class SentenceSimilarityRuleUpdateView(UpdateView):
    model = SentenceSimilarityRule
    fields = ['sentence', 'similarity_threshold']
    template_name = 'generic_views/rule_update.html'

    def get_success_url(self):
        return reverse_lazy('building_blocks', kwargs={'q_id': self.object.question.id})

    def form_valid(self, form):
        all_answers = Answer.objects.filter(question_id=self.object.question.id).all()
        chosen_answers = recursive_filtering_chosen_answers(self.object, all_answers)
        child_child_list, child_id_list = recursive_rule_child_chain(self.object)

        # remove rule strings for this rule and all child rules from all applied answers
        remove_rule_strings(child_id_list + [self.object.id], child_child_list + [self.object], all_answers)  
        sentencesimilarityrule_update(self.object, form.cleaned_data['sentence'], form.cleaned_data['similarity_threshold'], self.object.method, chosen_answers, self.object.question)  
        update_all_children_rules(child_child_list, chosen_answers, self.object.question)  

        return super().form_valid(form) 


class AnswerLengthRuleUpdateView(UpdateView):
    model = AnswerLengthRule
    fields = ['length', 'length_type']
    template_name = 'generic_views/rule_update.html'

    def get_success_url(self):
        return reverse_lazy('building_blocks', kwargs={'q_id': self.object.question.id})

    def form_valid(self, form):
        all_answers = Answer.objects.filter(question_id=self.object.question.id).all()
        chosen_answers = recursive_filtering_chosen_answers(self.object, all_answers)
        child_child_list, child_id_list = recursive_rule_child_chain(self.object)

        # remove rule strings for this rule and all child rules from all applied answers
        remove_rule_strings(child_id_list + [self.object.id], child_child_list + [self.object], all_answers)  
        answerlengthrule_update(self.object, form.cleaned_data['length'], form.cleaned_data['length_type'], chosen_answers, self.object.question)  
        update_all_children_rules(child_child_list, chosen_answers, self.object.question)  

        return super().form_valid(form) 