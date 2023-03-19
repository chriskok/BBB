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

# ================================== #
#               ICSA V4              #
# ================================== #

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

    # TODO: filter string parsing -> needs to be ordered and interpret logic gates
    # e.g. "sk_ask&ss_104|!sk_what" = similar keyword to "ask" AND sim sentence to 104 OR NOT similar keyword to "what"
    # e.g. http://127.0.0.1:8000/clustering/building_blocks/1/sk_ask_0.4_9&ss_1
    # NOTE: maybe it'd be better in the future to make a model for each applied BB and associated answers, to easily backtrack and reference
    # NOTE: make sure input string doesn't have '&', '|', '_'
    # NOTE: maybe instead of AND, OR, and NOT, we can just either apply a sequential filter or allow MERGING (with logic gates)

    filter_strings = [] 
    keywords = []

    # if (filter):
    #     df = pd.DataFrame(list(chosen_answers.values()))
    #     split_filter = re.split(r'\&|\|', filter)
    #     for f in split_filter:
    #         f_split = f.split('_')
    #         ori_df = df.copy()
    #         bb_string = f_split[0].replace("!", "")
    #         if(bb_string == 'spk'):
    #             df = bb.specific_keyword(df, *[eval(x) if x.replace('.','',1).isnumeric() else x for x in f_split[1:]])
    #             filter_strings.append("Answers with specific keyword: {}".format(f_split[1]))
    #         elif(bb_string == 'sk'):
    #             df = bb.similar_keyword(df, *[eval(x) if x.replace('.','',1).isnumeric() else x for x in f_split[1:]])
    #             filter_strings.append("Answers with similar keyword to: {} | args: {}".format(f_split[1], ", ".join(f_split[2:])))
    #             keywords.append(f_split[1])
    #         elif(bb_string == 'ss'): 
    #             selected_idx = int(f_split[1])
    #             filter_strings.append("Answers with similar sentence to: {} | args: {}".format(df.iloc[selected_idx]['answer_text'], ", ".join(f_split[2:])))
    #             df = bb.similar_sentence(df, *[eval(x) if x.replace('.','',1).isnumeric() else x for x in f_split[1:]])
    #         elif(bb_string == 'neg'): 
    #             df = bb.filter_by_negation(df, *[eval(x) if x == 'True' or x == 'False' else x for x in f_split[1:]])
    #             filter_strings.append("Answers with negation")
    #         elif(bb_string == 'sa'):
    #             df = bb.sentiment_analysis(df, *[eval(x) if x.replace('.','',1).isnumeric() else x for x in f_split[1:]])
    #             filter_strings.append("Answers with sentiment: {}".format(f_split[1]))
    #         elif(bb_string == 'ne'):
    #             df = bb.filter_named_entities(df, *[eval(x) if x.replace('.','',1).isnumeric() else x for x in f_split[1:]])
    #             filter_strings.append("Answers with named entity: {}".format(f_split[1]))
    #         elif(bb_string == 'ques'):
    #             df = bb.filter_by_question(df, *[eval(x) if x == 'True' or x == 'False' else x for x in f_split[1:]])
    #             filter_strings.append("Answers with question")

    #         if (f_split[0].startswith('!')): 
    #             df = bb.invert_results(ori_df, df)
    #             filter_strings[-1] = 'NOT: ' + filter_strings[-1]
                
    #     chosen_answers = df.to_dict('records')
    #     answer_count = df.shape[0]

    # Handle form input
    if request.method == 'POST':
        form = BuildingBlocksForm(request.POST)
        if form.is_valid():
            keyword = form.cleaned_data['keyword']
            similarity = form.cleaned_data['similarity']

            # filters answers that have keyword 
            df = pd.DataFrame(list(chosen_answers.values()))
            df = bb.similar_keyword(df, keyword, sim_score_threshold=similarity)
            student_id_list = df["student_id"].values.tolist()
            filtered_answers = Answer.objects.filter(question=current_question_obj, student_id__in=student_id_list)

            # handle keyword rule creation
            new_rule,_ = KeywordRule.objects.get_or_create(question=current_question_obj, keyword=keyword)
            new_rule.applied_answers.add(*filtered_answers)
            new_rule.save()

            return HttpResponseRedirect(reverse('building_blocks', args=(q_id,)))
            # return HttpResponseRedirect(reverse('building_blocks', args=(q_id, "sk_{}_{}".format(keyword, similarity),)))
    else:
        form = BuildingBlocksForm()
    
    keywords = list(KeywordRule.objects.filter(question=current_question_obj).values_list('keyword', flat=True))

    context = {
        "question_obj": current_question_obj,
        "question_exam_id": q_id,
        "question_list": q_list,
        "answers": chosen_answers,
        "answer_count": answer_count,
        "filter_strings": filter_strings,
        "form": form,
        "keywords": keywords,
    }

    return render(request, "building_blocks.html", context)

def index(request):
  return render(request, "index.html", context={})
