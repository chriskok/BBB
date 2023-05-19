from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('building_blocks/<int:q_id>', views.building_blocks_view, name='building_blocks'),
    path('building_blocks/<int:q_id>/<str:filter>', views.building_blocks_view, name='building_blocks'),
    path('chatgpt/<int:q_id>', views.chatgpt_view, name='chatgpt'),
    path('chatgpt/<int:q_id>/<str:method>', views.chatgpt_view, name='chatgpt'),
    path('refinement/<int:q_id>', views.rule_refinement_view, name='refinement'),
    path('change_cluster/<int:q_id>', views.change_answer_cluster, name='change_cluster'),          # non-page

    # SYSTEM
    path('system_reset/', views.system_reset_view, name='system_reset'),                            # non-page
    path('system_reset/<int:question_id>', views.system_reset_view, name='system_reset'),           # non-page

    # GENERIC
    path("keywordrule_update/<pk>", views.KeywordRuleUpdateView.as_view(), name="keywordrule_update"),
    path("sentencesimilarityrule_update/<pk>", views.SentenceSimilarityRuleUpdateView.as_view(), name="sentencesimilarityrule_update"),
    path("answerlengthrule_update/<pk>", views.AnswerLengthRuleUpdateView.as_view(), name="answerlengthrule_update"), 
]