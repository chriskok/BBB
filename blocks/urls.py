from django.urls import register_converter, path
from . import views, converters

register_converter(converters.NegativeIntConverter, 'negint')

urlpatterns = [
    path('', views.index, name='index'),

    # COMB V5
    path('rubric_creation/<int:q_id>', views.rubric_creation, name='rubric_creation'),
    path('update_rubric_list/<int:q_id>', views.update_rubric_list, name='update_rubric_list'),     # non-page
    path('rubric_refinement/<int:q_id>', views.rubric_refinement, name='rubric_refinement'),
    path('update_answer_tag/', views.update_answer_tag, name='update_answer_tag'),                  # non-page
    path('update_feedback/', views.update_feedback, name='update_feedback'),                        # non-page
    path('update_feedback/<int:feedback_id>', views.update_feedback, name='update_feedback'),       # non-page
    path('rubric_feedback/<int:q_id>', views.rubric_feedback, name='rubric_feedback'),
    path('rubric_tagging/<int:q_id>', views.rubric_tagging, name='rubric_tagging'),
    path('combv5_reset/', views.combv5_reset_view, name='combv5_reset'),                            # non-page
    path('combv5_reset/<int:question_id>', views.combv5_reset_view, name='combv5_reset'),           # non-page

    # BUILDING BLOCKS
    path('rule_suggestions/<int:q_id>', views.rule_suggestions, name='rule_suggestions'),            # non-page
    path('building_blocks/<int:q_id>', views.building_blocks_view, name='building_blocks'),
    path('building_blocks/<int:q_id>/<str:filter>', views.building_blocks_view, name='building_blocks'),

    # CHATGPT
    path('chatgpt/<int:q_id>', views.chatgpt_view, name='chatgpt'),
    path('chatgpt/<int:q_id>/<str:method>', views.chatgpt_view, name='chatgpt'),

    # REFINEMENT 
    path('refinement/<int:q_id>', views.rule_refinement_view, name='refinement'),
    path('change_cluster/<int:q_id>', views.change_answer_cluster, name='change_cluster'),          # non-page
    path('cluster_reset/<int:question_id>', views.cluster_reset_view, name='cluster_reset'),        # non-page

    # GRADING
    path('answer_edit/<int:id>', views.answer_edit_view, name='answer_edit'), 
    path('grade/<int:q_id>/<negint:id>', views.cluster_grade_view, name='grade'),

    # SYSTEM
    path('system_reset/', views.system_reset_view, name='system_reset'),                            # non-page
    path('system_reset/<int:question_id>', views.system_reset_view, name='system_reset'),           # non-page

    # GENERIC
    path("keywordrule_update/<pk>", views.KeywordRuleUpdateView.as_view(), name="keywordrule_update"),
    path("sentencesimilarityrule_update/<pk>", views.SentenceSimilarityRuleUpdateView.as_view(), name="sentencesimilarityrule_update"),
    path("answerlengthrule_update/<pk>", views.AnswerLengthRuleUpdateView.as_view(), name="answerlengthrule_update"), 
]