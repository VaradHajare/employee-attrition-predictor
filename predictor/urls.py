from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('sample-csv/', views.download_sample_csv, name='sample_csv'),
]
