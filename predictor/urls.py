from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('history/', views.history, name='history'),
    path('history/<int:pk>/', views.history_detail, name='history_detail'),
]
