# tutorial/tables.py
import django_tables2 as tables
import itertools
from django_tables2.columns import TemplateColumn 
from django_tables2.utils import A  # alias for Accessor
from django.utils.html import format_html

from .models import Question, Answer

class QuestionTable(tables.Table):
    class Meta:
        model = Question
        template_name = "django_tables2/bootstrap4.html"
        fields = ("question_id", "question_text", "question_keywords",)

class AnswerTable(tables.Table):
    q_text = tables.Column(accessor='question.question_text')
    q_id = tables.Column(accessor='question.question_id')
    # cluster_id = tables.Column(accessor='cluster.cluster_id')
    # answer_text = tables.Column(linkify=True)
    delete = TemplateColumn(template_name='answer_delete_column.html')
    class Meta:
        ordering = ['q_id', 'student_id']
        model = Answer
        template_name = "django_tables2/bootstrap4.html"
        fields = ("student_id", "q_id", "q_text", "answer_text",)

class ClusterTable(tables.Table):
    # https://django-tables2.readthedocs.io/en/latest/pages/custom-data.html
    # answer_text = tables.Column()

    # def render_answer_text(self, value):
    #     value = value.replace("{}".format('attributes'), "<span class=\"text-primary\">{}</span>".format('attributes'))
    #     # value = value.replace("{}".format('attributes'), "<mark>{}</mark>".format('attributes'))
    #     return format_html("{}".format(value))
    class Meta:
        model = Answer
        template_name = "django_tables2/bootstrap4.html"
        fields = ("student_id", "answer_text", "points" )

class ClusterGradeTable(tables.Table):

    # edit = TemplateColumn(template_name='answer_edit_column.html')
    class Meta:
        model = Answer
        template_name = "django_tables2/bootstrap4.html"
        fields = ("student_id", "answer_text" )

class UnclusteredGradeTable(tables.Table):
    override_grade = tables.Column(verbose_name= 'Grade' )
    edit = TemplateColumn(template_name='answer_edit_column.html')
    class Meta:
        model = Answer
        template_name = "django_tables2/bootstrap4.html"
        fields = ("student_id", "answer_text" )

class ClusterDetailsTable(tables.Table):
    # https://django-tables2.readthedocs.io/en/latest/pages/custom-data.html
    # answer_text = tables.Column()
    # flag = tables.CheckBoxColumn(accessor='pk')
    flag = TemplateColumn('<input type="checkbox" value="{{ record.pk }}" {% if record.flagged  %}checked{% endif %}/>', verbose_name="Flag?")
    class Meta:
        model = Answer
        template_name = "django_tables2/bootstrap4.html"
        fields = ("student_id", "answer_text", "points" )

