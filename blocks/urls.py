from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('building_blocks/<int:q_id>', views.building_blocks_view, name='building_blocks'),
    path('building_blocks/<int:q_id>/<str:filter>', views.building_blocks_view, name='building_blocks'),
]