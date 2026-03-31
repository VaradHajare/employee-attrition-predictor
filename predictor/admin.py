from django.contrib import admin
from .models import EmployeePrediction


@admin.register(EmployeePrediction)
class EmployeePredictionAdmin(admin.ModelAdmin):
    list_display = (
        'employee_name',
        'department',
        'job_role',
        'attrition_probability',
        'prediction',
        'created_at',
    )
    list_filter = ('department', 'overtime', 'prediction', 'created_at')
    search_fields = ('employee_name', 'department', 'job_role')
    ordering = ('-created_at',)
