from django.db import models


class EmployeePrediction(models.Model):
    employee_name = models.CharField(max_length=255)
    age = models.PositiveSmallIntegerField()
    gender = models.CharField(max_length=20)
    department = models.CharField(max_length=100)
    job_role = models.CharField(max_length=100)
    business_travel = models.CharField(max_length=30)
    education_field = models.CharField(max_length=100)
    marital_status = models.CharField(max_length=20)
    overtime = models.CharField(max_length=10)
    monthly_income = models.PositiveIntegerField()
    job_satisfaction = models.PositiveSmallIntegerField()
    environment_satisfaction = models.PositiveSmallIntegerField()
    work_life_balance = models.PositiveSmallIntegerField()
    years_at_company = models.PositiveSmallIntegerField()
    years_in_current_role = models.PositiveSmallIntegerField()
    years_since_last_promotion = models.PositiveSmallIntegerField()
    num_companies_worked = models.PositiveSmallIntegerField()
    distance_from_home = models.PositiveSmallIntegerField()
    percent_salary_hike = models.PositiveSmallIntegerField()
    training_times_last_year = models.PositiveSmallIntegerField()
    job_level = models.PositiveSmallIntegerField()
    prediction = models.PositiveSmallIntegerField()
    attrition_probability = models.DecimalField(max_digits=5, decimal_places=2)
    risk_factors = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.employee_name} - {self.attrition_probability}%"

    @property
    def will_leave(self):
        return self.prediction == 1
