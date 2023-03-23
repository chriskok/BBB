import re
import pandas as pd
import json
import csv

from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.db.models import Avg, Max, Min, Sum
from django.contrib.auth.decorators import login_required

from itertools import product
import building_blocks as bb 

from .models import *
from .forms import *

import warnings
warnings.filterwarnings("ignore")

def add_rule_string(answer, new_rule, rule_string):
    answer.applied_rules.add(new_rule)
    curr_rule_strings = answer.get_rule_strings()
    curr_rule_strings.append((new_rule.id, rule_string))
    answer.set_rule_strings(curr_rule_strings)
    answer.save()

def handle_rule_input(form, chosen_answers, current_question_obj):

    if (form.cleaned_data['rule_type_selection'] == 'keyword_rule'):
        keyword = form.cleaned_data['keyword']
        similarity = form.cleaned_data['keyword_similarity']

        # filters answers that have keyword 
        df = pd.DataFrame(list(chosen_answers.values()))
        df = bb.similar_keyword(df, keyword, sim_score_threshold=similarity)
        relevant_keywords = df.word.unique().tolist()

        # handle keyword rule creation
        new_rule,_ = KeywordRule.objects.get_or_create(question=current_question_obj, keyword=keyword, similarity_threshold=similarity, relevant_keywords=relevant_keywords) 
        student_id_list = df["student_id"].values.tolist()
        filtered_answers = Answer.objects.filter(question=current_question_obj, student_id__in=student_id_list)

        # go through each filtered answer and assign the rule and rule strings
        for answer in filtered_answers:
            curr_row = df[df['student_id'] == answer.student_id].iloc[0]
            add_rule_string(answer, new_rule, f"Keyword: {keyword} -> Matched: {curr_row['word']}, Similarity: {curr_row['score']}")
    elif (form.cleaned_data['rule_type_selection'] == 'sentence_rule'):
        sentence = form.cleaned_data['sentence']
        similarity = form.cleaned_data['sentence_similarity']
        method = form.cleaned_data['sentence_similarity_method']

        # filters answers that have keyword 
        df = pd.DataFrame(list(chosen_answers.values()))
        df = bb.similar_sentence(df, sentence, sim_score_threshold=similarity, method=method)

        # handle sentence sim rule creation
        new_rule,_ = SentenceSimilarityRule.objects.get_or_create(question=current_question_obj, sentence=sentence, similarity_threshold=similarity, method=method) 
        student_id_list = df["student_id"].values.tolist()
        filtered_answers = Answer.objects.filter(question=current_question_obj, student_id__in=student_id_list)

        # go through each filtered answer and assign the rule and rule strings
        for answer in filtered_answers:
            curr_row = df[df['student_id'] == answer.student_id].iloc[0]
            add_rule_string(answer, new_rule, f"Sentence: {sentence} -> Similarity: {curr_row['score']}")
    else: 
        print(form.cleaned_data)

# @login_required
def building_blocks_view(request, q_id, filter=None):
    q_list = Question.objects.all()

    # if the question queried does not exist, get the first Question available
    if Question.objects.filter(pk=q_id).exists():
        current_question_obj = Question.objects.get(pk=q_id)
    else:
        current_question_obj = Question.objects.first()
        q_id = current_question_obj.id

    chosen_answers = Answer.objects.filter(question_id=q_id)
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

    context = {
        "question_obj": current_question_obj,
        "question_exam_id": q_id,
        "question_list": q_list,
        "answers": chosen_answers,
        "answer_count": answer_count,
        "form": form,
        "keywords": keywords,
    }

    return render(request, "building_blocks.html", context)

def system_reset_view(request, question_id=None, include_rules=True):

    if(include_rules): 
        KeywordRule.objects.all().delete()
        SentenceSimilarityRule.objects.all().delete()

    # get specific answers for the current question
    if (question_id): chosen_answers = Answer.objects.filter(question_id=question_id).all()
    else: chosen_answers = Answer.objects.all()
    chosen_answers.update(rule_strings="[]") 

    return JsonResponse({'message': "System Reset!"}) 

def index(request):
  return render(request, "index.html", context={})
