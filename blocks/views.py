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
from django.forms.models import model_to_dict
from django_tables2 import SingleTableView

from itertools import product
import building_blocks as bb 
import llm_helpers as llmh
import spacy
nlp = spacy.load('en_core_web_lg')  # if not downloaded, run: python -m spacy download en_core_web_lg

from .models import *
from .forms import *
from .tables import *
from .colors import colors

import warnings
warnings.filterwarnings("ignore")

#############################################
#                  COMB V5                  #
#############################################

def rubric_creation(request, q_id):
    q_list = Question.objects.extra(select={'sorted_num': 'CAST(question_exam_id AS FLOAT)'}).order_by('sorted_num')

    # if the question queried does not exist, get the first Question available
    if Question.objects.filter(pk=q_id).exists():
        current_question_obj = Question.objects.get(pk=q_id)
    else:
        current_question_obj = q_list.first()
        q_id = current_question_obj.id

    chosen_answers = Answer.objects.filter(question_id=q_id)
    answer_count = len(chosen_answers)

    max_outlier_score = chosen_answers.aggregate(Max('outlier_score'))
    min_outlier_score = chosen_answers.aggregate(Min('outlier_score'))
    number_of_bins = 10
    bin_size = (max_outlier_score['outlier_score__max'] - min_outlier_score['outlier_score__min']) / number_of_bins

    # get one random example from each bin
    outlier_examples = []
    for i in range(number_of_bins):
        curr_bin = chosen_answers.filter(outlier_score__gte=min_outlier_score['outlier_score__min']+i*bin_size, outlier_score__lt=min_outlier_score['outlier_score__min']+(i+1)*bin_size)
        if curr_bin.exists():
            # outlier_examples.append(curr_bin.order_by('?').first())  # randomized
            # outlier_examples.append(model_to_dict(curr_bin.last(), fields=["id", "answer_text", "student_id", "outlier_score"]))
            outlier_examples.append(curr_bin.last())
    
    # create dict of answer id to answer text for each outlier example
    examples_dict = {}
    for answer in outlier_examples:
        examples_dict[answer.id] = answer.answer_text

    # check if rubric object exists for this question
    default_list = [
        {'id': 1, 'polarity': 'positive', 'title': 'Non-blocking Execution', 'description': 'Clearly states the purpose of asynchronous programming: to send, request, and receive data from a server without blocking other parts of the interface.', 'reasoning_dict': {}},
        {'id': 2, 'polarity': 'positive', 'title': 'User Interaction and Background Tasks', 'description': 'Highlights the need to allow user interaction with the page/app while certain operations are in progress.', 'reasoning_dict': {}},
        {'id': 3, 'polarity': 'negative', 'title': 'Reloading the Page', 'description': 'Mentions the need to reload the page when updating it.', 'reasoning_dict': {}},
        {'id': 4, 'polarity': 'negative', 'title': 'Sequential Execution', 'description': 'Describes the execution of instructions in a strictly sequential manner, without considering concurrent execution.', 'reasoning_dict': {}},
        {'id': 0, 'polarity': 'positive', 'title': 'Dynamic Web Pages', 'description': 'Emphasizes the capability to update specific parts of a page while keeping other parts unchanged.', 'reasoning_dict': {}},
        {'id': 0, 'polarity': 'negative', 'title': 'Dependency on Frequent Data Sending', 'description': 'States that asynchronous programming requires frequent data sending for every change.', 'reasoning_dict': {}},
    ]
    if not RubricList.objects.filter(question_id=q_id).exists():
        rubric_obj = RubricList.objects.create(question_id=q_id, rubric_list=json.dumps(default_list))
    else:
        rubric_obj = RubricList.objects.filter(question_id=q_id).first()

    rubric_list = rubric_obj.get_rubric_list()

    context = {
        "question_obj": current_question_obj,
        "question_exam_id": q_id,
        "question_list": q_list,
        "answers": outlier_examples,
        "answer_count": answer_count,
        "examples_dict": examples_dict,
        "rubric_list": rubric_list,
    }

    return render(request, "rubric_creation.html", context)

def update_rubric_list(request, q_id):
    if request.method == 'POST':
        new_rubric_list = json.loads(request.POST.get("rubric_list", None))
        rubric_obj = RubricList.objects.filter(question_id=q_id).first()
        rubric_obj.set_rubric_list(new_rubric_list)
        rubric_obj.save()
        message = 'update successful'
    else:
        message = 'update failed'
    return HttpResponse(message)

def old_rubric_creation(request, q_id):
    q_list = Question.objects.extra(select={'sorted_num': 'CAST(question_exam_id AS FLOAT)'}).order_by('sorted_num')

    # if the question queried does not exist, get the first Question available
    if Question.objects.filter(pk=q_id).exists():
        current_question_obj = Question.objects.get(pk=q_id)
    else:
        current_question_obj = q_list.first()
        q_id = current_question_obj.id

    # on POST request, parse the form 
    if request.method == "POST":
        form = RubricCreationForm(request.POST)
        if form.is_valid():
            suggestions = form.cleaned_data["rubric_suggestions"]
            method = form.cleaned_data["method"]

            # print(f"Rubric suggestions: {suggestions}")
            # print(f"Method: {method}")

            # redirect to the same page
            return HttpResponseRedirect(reverse("rubric_creation", args=(q_id,)))
    else:
        form = RubricCreationForm()

    chosen_answers = Answer.objects.filter(question_id=q_id)
    answer_count = len(chosen_answers)

    # check if rubric object exists for this question
    if not Rubric.objects.filter(question_id=q_id).exists():
        rubrics, msgs = llmh.create_rubrics(current_question_obj, chosen_answers)
        Rubric.objects.create(question_id=q_id, rubric_dict=json.dumps(rubrics), message_history=json.dumps(msgs))
        # print(f"Created new rubrics: {rubrics}")
    else:
        rubric_obj = Rubric.objects.filter(question_id=q_id).first()
        rubrics = rubric_obj.get_rubric_dict()
        # print(f"Found old rubrics: {rubrics}")

    # replace answer ids with answer objects
    for rubric in rubrics:
        answer_ids = rubric["answer_ids"]
        # transform comma-seperated string of IDs into list of ints
        answer_ids = [int(i) for i in answer_ids.split(",")]
        # replace answer IDs with answer objects
        rubric["answers"] = [Answer.objects.get(pk=answer_id) for answer_id in answer_ids]

    context = {
        "question_obj": current_question_obj,
        "question_exam_id": q_id,
        "question_list": q_list,
        "answers": chosen_answers,
        "answer_count": answer_count,
        "rubrics": rubrics,
        "form": form,
    }

    return render(request, "rubric_creation.html", context)

def rubric_refinement(request, q_id):
    q_list = Question.objects.extra(select={'sorted_num': 'CAST(question_exam_id AS FLOAT)'}).order_by('sorted_num')

    # if the question queried does not exist, get the first Question available
    if Question.objects.filter(pk=q_id).exists():
        current_question_obj = Question.objects.get(pk=q_id)
    else:
        current_question_obj = q_list.first()
        q_id = current_question_obj.id
    
    rubric_obj = RubricList.objects.filter(question_id=q_id).first()
    rubric_list = rubric_obj.get_rubric_list()
    chosen_answers = Answer.objects.filter(question_id=q_id)
    answer_count = len(chosen_answers)

    max_outlier_score = chosen_answers.aggregate(Max('outlier_score'))
    min_outlier_score = chosen_answers.aggregate(Min('outlier_score'))
    number_of_bins = 10
    bin_size = (max_outlier_score['outlier_score__max'] - min_outlier_score['outlier_score__min']) / number_of_bins

    # get one random example from each bin
    outlier_examples = []
    for i in range(number_of_bins):
        curr_bin = chosen_answers.filter(outlier_score__gte=min_outlier_score['outlier_score__min']+i*bin_size, outlier_score__lt=min_outlier_score['outlier_score__min']+(i+1)*bin_size)
        if curr_bin.exists():
            # outlier_examples.append(curr_bin.order_by('?').first())
            outlier_examples.append(curr_bin.first())
    
    # check if AnswerTag objects exist for this question
    if not AnswerTag.objects.filter(question_id=q_id).exists():
        tags = llmh.apply_rubrics(current_question_obj, outlier_examples, rubric_list)
        for ans_id in tags:
            for tag_dict in tags[ans_id]:
                AnswerTag.objects.create(question_id=q_id, answer_id=int(ans_id), tag=tag_dict["rubric"], reasoning_dict=json.dumps(tag_dict))
        ans_tags = AnswerTag.objects.filter(question_id=q_id)
    else:
        ans_tags = AnswerTag.objects.filter(question_id=q_id)

    context = {
        "question_obj": current_question_obj,
        "question_exam_id": q_id,
        "question_list": q_list,
        "rubric_list": rubric_list,
        "answers": outlier_examples,
        "answer_count": answer_count,
        "ans_tags": ans_tags,
    }

    return render(request, "rubric_refinement.html", context)

def rubric_feedback(request, q_id):
    q_list = Question.objects.extra(select={'sorted_num': 'CAST(question_exam_id AS FLOAT)'}).order_by('sorted_num')

    # if the question queried does not exist, get the first Question available
    if Question.objects.filter(pk=q_id).exists():
        current_question_obj = Question.objects.get(pk=q_id)
    else:
        current_question_obj = q_list.first()
        q_id = current_question_obj.id

    chosen_answers = Answer.objects.filter(question_id=q_id)
    answer_count = len(chosen_answers)

    max_outlier_score = chosen_answers.aggregate(Max('outlier_score'))
    min_outlier_score = chosen_answers.aggregate(Min('outlier_score'))
    number_of_bins = 10
    bin_size = (max_outlier_score['outlier_score__max'] - min_outlier_score['outlier_score__min']) / number_of_bins

    # get one random example from each bin
    outlier_examples = []
    for i in range(number_of_bins):
        curr_bin = chosen_answers.filter(outlier_score__gte=min_outlier_score['outlier_score__min']+i*bin_size, outlier_score__lt=min_outlier_score['outlier_score__min']+(i+1)*bin_size)
        if curr_bin.exists():
            # outlier_examples.append(curr_bin.order_by('?').first())
            outlier_examples.append(curr_bin.last())

    context = {
        "question_obj": current_question_obj,
        "question_exam_id": q_id,
        "question_list": q_list,
        "answers": outlier_examples,
        "answer_count": answer_count,
    }

    return render(request, "rubric_feedback.html", context)

def rubric_tagging(request, q_id):
    q_list = Question.objects.extra(select={'sorted_num': 'CAST(question_exam_id AS FLOAT)'}).order_by('sorted_num')

    # if the question queried does not exist, get the first Question available
    if Question.objects.filter(pk=q_id).exists():
        current_question_obj = Question.objects.get(pk=q_id)
    else:
        current_question_obj = q_list.first()
        q_id = current_question_obj.id

    chosen_answers = Answer.objects.filter(question_id=q_id)
    answer_count = len(chosen_answers)

    # check if rubric object exists for this question
    if not Rubric.objects.filter(question_id=q_id).exists():
        rubrics, msgs = llmh.create_rubrics(current_question_obj, chosen_answers)
        rubric_obj = Rubric.objects.create(question_id=q_id, rubric_dict=json.dumps(rubrics), message_history=json.dumps(msgs))
    else:
        rubric_obj = Rubric.objects.filter(question_id=q_id).first()
        rubrics = rubric_obj.get_rubric_dict()

    # check if AnswerTag objects exist for this question
    if not AnswerTag.objects.filter(question_id=q_id).exists():
        tags, msgs = llmh.tag_answers(current_question_obj, chosen_answers, rubrics, num_samples=20)
        for tag in tags:
            AnswerTag.objects.create(question_id=q_id, answer_id=int(tag["answer_id"]), tag=tag["rubrics"], reasoning_dict=tag["reasoning"])
        ans_tags = AnswerTag.objects.filter(question_id=q_id)
    else:
        ans_tags = AnswerTag.objects.filter(question_id=q_id)

    # make dictionary of R<number>: <rubric> for each rubric
    rubric_dict = {}
    for i, rubric in enumerate(rubrics):
        rubric_dict[f"R{i+1}"] = rubric["rubric"]

    context = {
        "question_obj": current_question_obj,
        "question_exam_id": q_id,
        "question_list": q_list,
        "answers": chosen_answers,
        "answer_count": answer_count,
        "ans_tags": ans_tags,
        "rubric_dict": rubric_dict,
    }

    return render(request, "rubric_tagging.html", context)

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

def similar_sentence_filter(chosen_answers, current_question_obj, sentence, positive_examples, negative_examples):
    # filters answers 
    df = pd.DataFrame(list(chosen_answers.values()))
    # df = bb.similar_sentence(df, sentence, sim_score_threshold=similarity, method=method)
    df, lowest_positive_score = bb.similar_sentence_by_example(df, sentence, positive_examples, negative_examples)
    student_id_list = df["student_id"].values.tolist()
    filtered_answers = Answer.objects.filter(question=current_question_obj, student_id__in=student_id_list)

    return df, filtered_answers, lowest_positive_score

def similar_concept_filter(chosen_answers, current_question_obj, concept, similarity):

    df = pd.DataFrame(list(chosen_answers.values()))
    df = bb.similar_concept(df, current_question_obj, concept, score_threshold=similarity)
    student_id_list = df["student_id"].values.tolist() if not df.empty else []
    filtered_answers = Answer.objects.filter(question=current_question_obj, student_id__in=student_id_list)

    return df, filtered_answers

def answer_length_filter(chosen_answers, current_question_obj, length, length_type):
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
        _, return_answers, _ = similar_sentence_filter(chosen_answers, rule.question, rule.sentence, rule.get_positive_examples(), rule.get_negative_examples())
    elif rule.polymorphic_ctype.name == "answer length rule":
        _, return_answers = answer_length_filter(chosen_answers, rule.question, rule.length, rule.length_type)
    elif rule.polymorphic_ctype.name == "concept similarity rule":
        _, return_answers = similar_concept_filter(chosen_answers, rule.question, rule.concept, rule.similarity_threshold)
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
    positive_examples = form.cleaned_data['positive_examples'].split(',') if form.cleaned_data['positive_examples'] else []
    negative_examples = form.cleaned_data['negative_examples'].split(',') if form.cleaned_data['negative_examples'] else []

    # KEYWORD RULE
    if (form.cleaned_data['rule_type_selection'] == 'keyword_rule'):
        keyword = form.cleaned_data['keyword']
        similarity = form.cleaned_data['keyword_similarity']

        # filters answers that have keyword 
        df, filtered_answers, relevant_keywords = similar_keyword_filter(chosen_answers, current_question_obj, keyword, similarity)

        # handle keyword rule creation
        new_rule,_ = KeywordRule.objects.get_or_create(question=current_question_obj, parent=parent_rule, keyword=keyword, similarity_threshold=similarity, relevant_keywords=relevant_keywords, polarity=form.cleaned_data['rule_polarity'], positive_examples=json.dumps(positive_examples), negative_examples=json.dumps(negative_examples))

        # go through each filtered answer and assign the rule and rule strings
        for answer in filtered_answers:
            curr_row = df[df['student_id'] == answer.student_id].iloc[0]
            add_rule_string(answer, new_rule, f"{polarity_emoji} Keyword: {keyword} -> Matched: {curr_row['word']}, Similarity: {curr_row['score']:.2f}")
    
    # SENTENCE SIM RULE
    elif (form.cleaned_data['rule_type_selection'] == 'sentence_rule'):
        sentence = form.cleaned_data['sentence']
        # similarity = form.cleaned_data['sentence_similarity']
        # method = form.cleaned_data['sentence_similarity_method']

        # filters answers 
        df, filtered_answers, lowest_positive_score = similar_sentence_filter(chosen_answers, current_question_obj, sentence, positive_examples, negative_examples)

        # handle sentence sim rule creation
        new_rule,_ = SentenceSimilarityRule.objects.get_or_create(question=current_question_obj, parent=parent_rule, sentence=sentence, similarity_threshold=lowest_positive_score, method="sbert", polarity=form.cleaned_data['rule_polarity'], positive_examples=json.dumps(positive_examples), negative_examples=json.dumps(negative_examples))

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

    # CONCEPT RULE
    elif (form.cleaned_data['rule_type_selection'] == 'concept_rule'):

        # # populate concepts for ALL answers of this question
        # # TODO: get a better process to do this through pre-loading instead
        # bb.populate_answer_concepts(current_question_obj, chosen_answers)

        concept_string = form.data['concept']
        if concept_string == '-- Select Concept --': 
            return 
        similarity = form.cleaned_data['concept_similarity']

        # filters answers 
        df, filtered_answers = similar_concept_filter(chosen_answers, current_question_obj, concept_string, similarity)

        # handle rule creation
        new_rule,_ = ConceptSimilarityRule.objects.get_or_create(question=current_question_obj, parent=parent_rule, concept=concept_string, similarity_threshold=similarity, polarity=form.cleaned_data['rule_polarity']) 

        # go through each filtered answer and assign the rule and rule strings
        for answer in filtered_answers:
            curr_row = df[df['student_id'] == answer.student_id].iloc[0]
            add_rule_string(answer, new_rule, f"{polarity_emoji} Concept: {concept_string} -> Similarity: {curr_row['score']:.2f}")
    else: 
        print(form.cleaned_data)

#############################################
#                PAGE VIEWS                 #
#############################################

def rule_suggestions(request, q_id):

    current_question_obj = Question.objects.get(pk=q_id)

    # if this is a POST request we need to process the form data
    if request.method == "POST":
        # create a form instance and populate it with data from the request:
        form = RuleSuggestionForm(request.POST)
        # check whether it's valid:
        if form.is_valid():

            selection = form.cleaned_data["selection"]
            reason = form.cleaned_data["reason"]
            ans = form.cleaned_data["full_ans"]

            # TODO: The output must be in a strict JSON format: {{'rule_type': 'the type of rule you have chosen', 'args': 'the arguments associated to that rule', 'reason': 'your detailed reason for the choice'}}.
            prompt = [
                        {"role": "system", "content": f"You are an expert instructor for your given course. You've given the question {current_question_obj.question_text} on a recent final exam. Now you have to grade all the responses to that open-ended question with our grading system. The goal is to use your expert knowledge to create specific, understandable, rule-based clusters of answers so that we can grade and provide feedback to each cluster all at once. \n\nGiven an annotation that you (or another expert instructor) has made on one of the answers, you are to select one most suitable rule from a list of possible rules that can be used to best define the annotation made (that would apply to most other answers of the same type). \n\nHere is the list of rules for you to choose from:\n- Keyword Similarity: Measures word-level similarity and applies to all answers that have a similar/same keyword (rule_type: keywordsim, args: <one keyword, usually from the answer itself>)\n- Sentence Similarity: Measures sentence-level similarity through TF-IDF, Sentence-BERT, or Spacy and applies to all answers that have a similar/same sentence meaning (rule_type: sentencesim, args: <full sentence or part of sentence, usually from the answer itself>)\n- Concept Similarity: Leverages ChatGPT to classify each answer based on automatically generated concepts derived from the question's related topics. Matches any other answers classified as having the same concept. For this question, the list of concepts are: Asynchronous programming, Callback functions, Promises, Event loop, Non-blocking I/O, Performance. (rule_type: conceptsim, args: <chosen concept from list of concepts>)\n- Answer Length: Word- or character-level length checking to determine if the answer exceeds a specific threshold. (rule_type: answerlength, args: <number of words/characters to set as limit>)\n\nPlease give the following as output: Rule Type, Arguments, and Reasoning."},
                        {"role": "user", "content": f"Answer: {ans.strip()}\nHighlighted section: {selection}\nAnnotation: {reason}"},
                    ]
            chatgpt_response = bb.prompt_chatgpt(prompt)

            return HttpResponse(chatgpt_response)

    return HttpResponse("Incorrectly formatted rule suggestion request")

# @login_required
def building_blocks_view(request, q_id, filter=None):
    q_list = Question.objects.extra(select={'sorted_num': 'CAST(question_exam_id AS FLOAT)'}).order_by('sorted_num')

    # if the question queried does not exist, get the first Question available
    if Question.objects.filter(pk=q_id).exists():
        current_question_obj = Question.objects.get(pk=q_id)
    else:
        current_question_obj = q_list.first()
        q_id = current_question_obj.id

    chosen_answers = Answer.objects.filter(question_id=q_id).order_by('outlier_score')
    answer_count = len(chosen_answers)
    # NOTE: maybe instead of AND, OR, and NOT, we can just either apply a sequential filter or allow MERGING (with logic gates)

    old_form = None
    # Handle form input
    if request.method == 'POST':
        form = BuildingBlocksForm(request.POST)
        if form.is_valid():
            handle_rule_input(form, chosen_answers, current_question_obj)
            return HttpResponseRedirect(reverse('building_blocks', args=(q_id,)))
        else:
            old_form = form
            form = BuildingBlocksForm()
    else:
        form = BuildingBlocksForm()
    
    keywords = list(KeywordRule.objects.filter(question=current_question_obj).values_list('keyword', flat=True))
    rules = Rule.objects.filter(question=current_question_obj)
    orphan_rules = Rule.objects.filter(question=current_question_obj, parent=None)
    color_to_rule = { k.id:v for (k,v) in zip(rules, colors[:len(rules)])} 

    if (current_question_obj.related_concepts == ""): 
        current_question_obj.related_concepts = bb.get_question_concepts(current_question_obj.question_text)
        current_question_obj.save()
    
    answer_concepts = []
    related_concepts = bb.extract_key_question_concepts(current_question_obj.related_concepts)
    for ans in chosen_answers:
        curr_concept_list = ans.get_concept_scores()
        if (not curr_concept_list): continue
        for item in curr_concept_list:
            if item not in answer_concepts and item in related_concepts:
                answer_concepts.append(item)

    context = {
        "question_obj": current_question_obj,
        "question_exam_id": q_id,
        "question_list": q_list,
        "answers": chosen_answers,
        "answer_count": answer_count,
        "suggestions_form": RuleSuggestionForm(),
        "form": form,
        "old_form": old_form,
        "keywords": keywords,
        "rules": rules,
        "orphan_rules": orphan_rules,
        "color_to_rule": color_to_rule,
        "related_concepts": bb.convert_question_concepts(current_question_obj.related_concepts),
        "answer_concepts": answer_concepts
    }

    return render(request, "building_blocks.html", context)

# methods: question_only, examples, rubrics, rubrics_and_examples
def chatgpt_view(request, q_id, method='question_only'):
    q_list = Question.objects.extra(select={'sorted_num': 'CAST(question_exam_id AS FLOAT)'}).order_by('sorted_num')

    # method = request.GET.get('method', 'question_only')
    model = request.GET.get('model', 'text-davinci-003')

    # if the question queried does not exist, get the first Question available
    if Question.objects.filter(pk=q_id).exists():
        current_question_obj = Question.objects.get(pk=q_id)
    else:
        current_question_obj = q_list.first()
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

def bulk_change_topic(chosen_answers, q_id, new_topic):
    cluster_obj = Cluster.objects.get(question_id=q_id, cluster_id=new_topic)

    # Update all answers
    for answer in chosen_answers:
        answer.cluster = cluster_obj
    Answer.objects.bulk_update(chosen_answers, ['cluster']) 

def parse_request(request, q_id):
    to_delete = request.GET.get('delete')
    to_add = request.GET.get('add')
    to_merge = request.GET.get('merge')

    if (to_delete):
        update_list = [int(i) for i in to_delete.split(',')]
        print('Deleting: {}'.format(update_list))
        for cluster_id in update_list:
            chosen_answers = Answer.objects.filter(question_id=q_id, cluster__cluster_id=cluster_id)
            bulk_change_topic(chosen_answers, q_id, -1)
            curr_cluster = Cluster.objects.get(question_id=q_id, cluster_id=cluster_id)
            curr_cluster.delete()
    elif (to_add):
        new_id = int(Cluster.objects.filter(question_id=q_id).aggregate(Max('cluster_id'))['cluster_id__max']) + 1
        print('Adding Cluster: {}'.format(new_id))
        Cluster.objects.create(
            cluster_id = new_id,
            question_id = q_id,
        )
    # elif (to_merge):
    #     update_list = [int(i) for i in to_merge.split(',')]
    #     print('Merging: {}'.format(update_list))
    #     min_in_list = min(update_list)
    #     print(min_in_list)
    #     for cluster_id in update_list:
    #         if (cluster_id == min_in_list): continue
    #         chosen_answers = Answer.objects.filter(question_id=q_id, cluster__cluster_id=cluster_id)
    #         print(chosen_answers)
    #         bulk_change_topic(chosen_answers, q_id, min_in_list)
    #         curr_cluster = Cluster.objects.get(question_id=q_id, cluster_id=cluster_id)
    #         curr_cluster.delete()
    else:
        return False
    
    return True

def change_answer_cluster(request, q_id):
    changed_item = request.POST.getlist('changed_item')[0]
    new_topic = request.POST.getlist('new_topic')[0]
    print('change answer #{} to topic #{}'.format(changed_item, new_topic))

    clus_obj = Cluster.objects.get(question_id=q_id, cluster_id=new_topic)
    ans_obj = Answer.objects.filter(pk=changed_item).update(cluster=clus_obj)

    return JsonResponse({"changed_item":changed_item})

def create_rule_clusters(answers, question):
    Cluster.objects.filter(question = question).delete()

    norule_cluster = Cluster.objects.create(
        question = question,
        cluster_id = -1,
        cluster_name = "No Rules",
    )

    def create_new_cluster(rules):
        new_id = int(Cluster.objects.filter(question = question).aggregate(Max('cluster_id'))['cluster_id__max']) + 1
        new_cluster = Cluster.objects.create(
            question = question,
            cluster_id = new_id,
            cluster_name = "Group {}".format(new_id),
        )
        for rule in rules:
            new_cluster.applied_rules.add(rule)
        new_cluster.save()
        return new_cluster
    
    existing_clusters = {'': norule_cluster}
    for ans in answers:
        curr_rule_ids = ",".join([str(i) for i in ans.applied_rules.all().values_list('id', flat=True)])
        if curr_rule_ids not in existing_clusters:
            existing_clusters[curr_rule_ids] = create_new_cluster(ans.applied_rules.all())
        ans.cluster = existing_clusters[curr_rule_ids]
        ans.save()

def rule_refinement_view(request, q_id):
    question_id_list = list(Question.objects.values_list('pk', flat=True))

    args_found = parse_request(request, q_id)
    if (args_found): return HttpResponseRedirect(reverse('refinement', kwargs={'q_id': q_id}))

    chosen_answers = Answer.objects.filter(question_id=q_id)
    current_question_obj = Question.objects.filter(pk=q_id)[0]
    if not Cluster.objects.filter(question = current_question_obj):
        create_rule_clusters(chosen_answers, current_question_obj)

    cluster_dict = {}
    average_points_dict = {}
    # cluster_rules = {}
    # cluster_ids = {}

    # calculate average points of each cluster
    cluster_list = Cluster.objects.filter(question_id=q_id)
    for cluster in cluster_list:
        if (int(cluster.cluster_id) == -1): continue  # skip to leave till the end of the list
        cluster_dict[cluster.cluster_id] = Answer.objects.filter(question_id=q_id, cluster__cluster_id=cluster.cluster_id)
        point_avg = cluster_dict[cluster.cluster_id].aggregate(Avg('assigned_grade'))['assigned_grade__avg']
        average_points_dict[cluster.cluster_id] = point_avg if point_avg != None else 0
        # if (cluster.cluster_rules): 
        #     curr_cluster_rules = [x.strip() for x in cluster.cluster_rules.split(',')]
        #     cluster_rules[cluster.id] = curr_cluster_rules
        #     cluster_ids[cluster.id] = cluster.cluster_id
    cluster_dict[-1] = Answer.objects.filter(question_id=q_id, cluster__cluster_id=-1)
    point_avg = Answer.objects.filter(question_id=q_id, cluster__cluster_id=-1).aggregate(Avg('assigned_grade'))['assigned_grade__avg']
    average_points_dict[-1] = point_avg if point_avg != None else 0

    # Develop recommendations for changing clusters based on user-inputted keywords
    # TODO: Change rules to keywords - cluster.get_cluster_keywords()
    # query_df = pd.DataFrame(list(chosen_answers.values()))
    # recommended_reclusters = similarity_reclustering(q_id, query_df, cluster_rules, cluster_ids)

    rules = Rule.objects.filter(question=current_question_obj)
    color_to_rule = { k.id:v for (k,v) in zip(rules, colors[:len(rules)])} 

    context = {
        "question_exam_id": q_id,
        "cluster_dict": cluster_dict,
        "cluster_list": json.dumps(list(cluster_list.values())),
        "question_obj": current_question_obj,
        "average_points_dict": average_points_dict,
        "question_id_list": question_id_list,
        "color_to_rule": color_to_rule,
        # "recommended_reclusters": recommended_reclusters,
    }
    return render(request, "rule_refinement.html", context)

def answer_edit_view(request, id):
    obj = get_object_or_404(Answer, id=id)
    form = AnswerEditForm(request.POST or None, instance=obj)
    if form.is_valid():
        form.save()
        form = AnswerEditForm(instance=obj)
        # return HttpResponseRedirect(reverse('grade', kwargs={'q_id': q_id, 'id': c_id}))

    context = {
        'form': form,
        'answer_id': id,
    }
    return render(request, 'answer_edit.html', context)

def cluster_grade_view(request, q_id, id):
    question_id_list = list(Question.objects.values_list('pk', flat=True))

    chosen_answers = Answer.objects.filter(question_id=q_id)
    current_question_obj = Question.objects.filter(pk=q_id)[0]
    if not Cluster.objects.filter(question = current_question_obj):
        create_rule_clusters(chosen_answers, current_question_obj)

    # go through each cluster and check if any aren't referenced, if so, remove them from the list to grade
    cluster_list = Cluster.objects.filter(question_id=q_id)
    empty_cluster_ids = []
    for curr_cluster in cluster_list:
        curr_cluster_answers = curr_cluster.answer_set.all()
        if (curr_cluster_answers.count() == 0):
            empty_cluster_ids.append(curr_cluster.id)
        elif (curr_cluster.cluster_id == -1):  # check if all answers are complete specifically for Unclustered
            if(curr_cluster_answers.filter(override_grade__isnull=True).count() == 0): curr_cluster.cluster_grade = 0.0
            else: curr_cluster.cluster_grade = None
            curr_cluster.save()
    cluster_list = Cluster.objects.filter(question_id=q_id).exclude(id__in=empty_cluster_ids)

    # if the ID is for a cluster that doesn't exist, get the lowest cluster
    if cluster_list.filter(question_id=q_id, pk=id).exists():
        curr_cluster = Cluster.objects.get(question_id=q_id, pk=id)
    else:
        curr_cluster = cluster_list.filter(question_id=q_id, cluster_id__gte=0).first()
        curr_cluster = cluster_list.get(question_id=q_id, cluster_id=-1) if not curr_cluster else curr_cluster
        id = curr_cluster.id
    
    # curr_applied_rubrics = [] if isinstance(curr_cluster.get_cluster_rubrics(), str) else curr_cluster.get_cluster_rubrics()

    rules = Rule.objects.filter(question_id=q_id)
    color_to_rule = { k.id:v for (k,v) in zip(rules, colors[:len(rules)])} 

    # if this is a POST request we need to process the form data]
    # TODO: there's a bug here with saving the form and then applying the rubrics after - reverts back to form input
    if request.method == 'POST':        
        form = ClusterGradingForm(request.POST, instance=curr_cluster)
        # check whether it's valid:
        if form.is_valid():
            form.save()
    else:
        form = ClusterGradingForm(instance=curr_cluster)

    answer_qs = Answer.objects.filter(question_id=q_id, cluster__id=id)
    table = ClusterGradeTable(answer_qs) if curr_cluster.cluster_id != -1 else UnclusteredGradeTable(answer_qs)

    context = {
        "cluster": curr_cluster,
        "cluster_id": id,
        "question_exam_id": q_id,
        "question_id": q_id,
        "table": table,
        "cluster_list": cluster_list,
        "applied_rules": curr_cluster.applied_rules.all(),
        "color_to_rule": color_to_rule,
        "form": form,
        "question_id_list": question_id_list,
    }

    return render(request, "grading.html", context)

def cluster_reset_view(request, question_id):
    chosen_answers = Answer.objects.filter(question_id=question_id).all() 
    create_rule_clusters(chosen_answers, Question.objects.get(pk=question_id))

    return JsonResponse({'message': "Clusters Reset!"}) 

def system_reset_view(request, question_id=None):

    # get specific answers for the current question
    chosen_answers = Answer.objects.filter(question_id=question_id).all() if question_id else Answer.objects.all()
    chosen_answers.update(rule_strings="[]") 
    chosen_answers.update(override_grade=None)  # remove previous overriden grade and feedback
    chosen_answers.update(override_feedback=None)
    chosen_answers.update(cluster=None)

    # delete all clusters
    if (question_id): Cluster.objects.filter(question_id=question_id).all().delete()
    else: Cluster.objects.all().delete()

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
        add_rule_string(answer, rule, f"{'✔️' if rule.polarity == 'positive' else '❌'} Keyword: {keyword} -> Matched: {curr_row['word']}, Similarity: {curr_row['score']:.2f}")

def sentencesimilarityrule_update(rule, sentence, positive_examples, negative_examples, chosen_answers, current_question_obj):
        
    df, filtered_answers, lowest_positive_score = similar_sentence_filter(chosen_answers, current_question_obj, sentence, positive_examples, negative_examples)

    # go through each filtered answer and assign the rule and rule strings
    for answer in filtered_answers:
        curr_row = df[df['student_id'] == answer.student_id].iloc[0]
        add_rule_string(answer, rule, f"{'✔️' if rule.polarity == 'positive' else '❌'} Sentence: {sentence} -> Similarity: {curr_row['score']:.2f}")

    return lowest_positive_score

def conceptsimilarityrule_update(rule, concept, similarity, chosen_answers, current_question_obj):
        
    df, filtered_answers = similar_concept_filter(chosen_answers, current_question_obj, concept, similarity)

    # go through each filtered answer and assign the rule and rule strings
    for answer in filtered_answers:
        curr_row = df[df['student_id'] == answer.student_id].iloc[0]
        add_rule_string(answer, rule, f"{'✔️' if rule.polarity == 'positive' else '❌'} Concept: {concept} -> Similarity: {curr_row['score']:.2f}")

def answerlengthrule_update(rule, length, length_type, chosen_answers, current_question_obj):

    df, filtered_answers = answer_length_filter(chosen_answers, current_question_obj, length, length_type)

    # go through each filtered answer and assign the rule and rule strings
    for answer in filtered_answers:
        curr_row = df[df['student_id'] == answer.student_id].iloc[0]
        add_rule_string(answer, rule, f"{'✔️' if rule.polarity == 'positive' else '❌'} Length: {length} {length_type}s -> Answer Length: {curr_row['length']}")

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
            sentencesimilarityrule_update(rule, rule.sentence, rule.get_positive_examples(), rule.get_negative_examples(), recursive_filtering_chosen_answers(rule, chosen_answers), question)
        elif rule.polymorphic_ctype.name == "concept similarity rule":
            conceptsimilarityrule_update(rule, rule.concept, rule.similarity_threshold, recursive_filtering_chosen_answers(rule, chosen_answers), question)
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
    fields = ['sentence']
    template_name = 'generic_views/rule_update.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        positive_examples = Answer.objects.filter(pk__in=self.object.get_positive_examples()).values_list('answer_text',flat=True)
        negative_examples = Answer.objects.filter(pk__in=self.object.get_negative_examples()).values_list('answer_text',flat=True)
        patterns = bb.sentence_pattern_breakdown(positive_examples, negative_examples, pattern_limit=10) 
        chosen_answers = Answer.objects.filter(question_id=self.object.question.id)

        # context['sentence_pattern_breakdown'] = patterns
        context['regex_patterns'] = bb.convert_patterns_to_regex(patterns, chosen_answers) 
        context['positive_examples'] = positive_examples
        context['negative_examples'] = negative_examples
        return context
    
    def get_success_url(self):
        return reverse_lazy('building_blocks', kwargs={'q_id': self.object.question.id})

    def form_valid(self, form):
        all_answers = Answer.objects.filter(question_id=self.object.question.id).all()
        chosen_answers = recursive_filtering_chosen_answers(self.object, all_answers)
        child_child_list, child_id_list = recursive_rule_child_chain(self.object)

        # remove rule strings for this rule and all child rules from all applied answers
        remove_rule_strings(child_id_list + [self.object.id], child_child_list + [self.object], all_answers)  
        # sentencesimilarityrule_update(self.object, form.cleaned_data['sentence'], form.cleaned_data['similarity_threshold'], self.object.method, chosen_answers, self.object.question)  
        lowest_positive_score = sentencesimilarityrule_update(self.object, form.cleaned_data['sentence'], self.object.get_positive_examples(), self.object.get_negative_examples(), chosen_answers, self.object.question)  
        
        # BUG: fix lowest_positive_score is 0 because it runs twice when updating.
        # print(lowest_positive_score)
        # print(self.object)
        # self.object.similarity_threshold = lowest_positive_score
        # self.object.save()
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