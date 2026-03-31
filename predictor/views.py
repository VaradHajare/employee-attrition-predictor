import os
from decimal import Decimal

import joblib
import numpy as np
from django.db.utils import OperationalError, ProgrammingError
from django.shortcuts import get_object_or_404, render

from .models import EmployeePrediction

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL = None
ENCODERS = None
FEATURES = None
MODEL_LOAD_ERROR = None

CHOICES = {
    'Gender': ['Male', 'Female'],
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'JobRole': ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'],
    'BusinessTravel': ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
    'EducationField': ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'],
    'MaritalStatus': ['Divorced', 'Married', 'Single'],
    'OverTime': ['No', 'Yes'],
}


def load_model_artifacts():
    global MODEL, ENCODERS, FEATURES, MODEL_LOAD_ERROR

    if MODEL is not None and ENCODERS is not None and FEATURES is not None:
        return True
    if MODEL_LOAD_ERROR is not None:
        return False

    try:
        MODEL = joblib.load(os.path.join(BASE, 'ml_model', 'attrition_model.pkl'))
        ENCODERS = joblib.load(os.path.join(BASE, 'ml_model', 'label_encoders.pkl'))
        FEATURES = joblib.load(os.path.join(BASE, 'ml_model', 'feature_columns.pkl'))
        return True
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        return False


def build_raw_input(post_data):
    return {
        'Age': int(post_data.get('Age', 30)),
        'Gender': post_data.get('Gender', 'Male'),
        'Department': post_data.get('Department', 'Sales'),
        'JobRole': post_data.get('JobRole', 'Sales Executive'),
        'BusinessTravel': post_data.get('BusinessTravel', 'Travel_Rarely'),
        'EducationField': post_data.get('EducationField', 'Life Sciences'),
        'MaritalStatus': post_data.get('MaritalStatus', 'Single'),
        'OverTime': post_data.get('OverTime', 'No'),
        'MonthlyIncome': int(post_data.get('MonthlyIncome', 5000)),
        'JobSatisfaction': int(post_data.get('JobSatisfaction', 3)),
        'EnvironmentSatisfaction': int(post_data.get('EnvironmentSatisfaction', 3)),
        'WorkLifeBalance': int(post_data.get('WorkLifeBalance', 3)),
        'YearsAtCompany': int(post_data.get('YearsAtCompany', 5)),
        'YearsInCurrentRole': int(post_data.get('YearsInCurrentRole', 3)),
        'YearsSinceLastPromotion': int(post_data.get('YearsSinceLastPromotion', 1)),
        'NumCompaniesWorked': int(post_data.get('NumCompaniesWorked', 2)),
        'DistanceFromHome': int(post_data.get('DistanceFromHome', 10)),
        'PercentSalaryHike': int(post_data.get('PercentSalaryHike', 13)),
        'TrainingTimesLastYear': int(post_data.get('TrainingTimesLastYear', 3)),
        'JobLevel': int(post_data.get('JobLevel', 2)),
    }


def collect_risk_factors(raw):
    risks = []
    if raw['OverTime'] == 'Yes':
        risks.append('Working overtime significantly increases attrition risk.')
    if raw['MonthlyIncome'] < 4000:
        risks.append('Below-average income is a key driver of employee departure.')
    if raw['Age'] < 30:
        risks.append('Younger employees tend to have higher attrition rates.')
    if raw['YearsAtCompany'] < 3:
        risks.append('Employees in their first 2 years are at higher risk.')
    if raw['JobSatisfaction'] <= 2:
        risks.append('Low job satisfaction strongly predicts attrition.')
    if raw['DistanceFromHome'] > 20:
        risks.append('Long commutes increase the likelihood of leaving.')
    return risks[:3]


def run_prediction(raw):
    encoded = dict(raw)
    for col in ['Gender', 'Department', 'JobRole', 'BusinessTravel', 'EducationField', 'MaritalStatus', 'OverTime']:
        encoded[col] = ENCODERS[col].transform([raw[col]])[0]
    feat_vector = np.array([[encoded[f] for f in FEATURES]])
    prediction = int(MODEL.predict(feat_vector)[0])
    probability = round(float(MODEL.predict_proba(feat_vector)[0][1]) * 100, 1)
    return prediction, probability


def save_prediction(employee_name, raw, prediction, probability, risk_factors):
    return EmployeePrediction.objects.create(
        employee_name=employee_name,
        age=raw['Age'],
        gender=raw['Gender'],
        department=raw['Department'],
        job_role=raw['JobRole'],
        business_travel=raw['BusinessTravel'],
        education_field=raw['EducationField'],
        marital_status=raw['MaritalStatus'],
        overtime=raw['OverTime'],
        monthly_income=raw['MonthlyIncome'],
        job_satisfaction=raw['JobSatisfaction'],
        environment_satisfaction=raw['EnvironmentSatisfaction'],
        work_life_balance=raw['WorkLifeBalance'],
        years_at_company=raw['YearsAtCompany'],
        years_in_current_role=raw['YearsInCurrentRole'],
        years_since_last_promotion=raw['YearsSinceLastPromotion'],
        num_companies_worked=raw['NumCompaniesWorked'],
        distance_from_home=raw['DistanceFromHome'],
        percent_salary_hike=raw['PercentSalaryHike'],
        training_times_last_year=raw['TrainingTimesLastYear'],
        job_level=raw['JobLevel'],
        prediction=prediction,
        attrition_probability=Decimal(str(probability)),
        risk_factors=risk_factors,
    )


def build_result_context(record):
    form_data = {
        'Age': record.age,
        'Gender': record.gender,
        'Department': record.department,
        'JobRole': record.job_role,
        'BusinessTravel': record.business_travel,
        'EducationField': record.education_field,
        'MaritalStatus': record.marital_status,
        'OverTime': record.overtime,
        'MonthlyIncome': record.monthly_income,
        'JobSatisfaction': record.job_satisfaction,
        'EnvironmentSatisfaction': record.environment_satisfaction,
        'WorkLifeBalance': record.work_life_balance,
        'YearsAtCompany': record.years_at_company,
        'YearsInCurrentRole': record.years_in_current_role,
        'YearsSinceLastPromotion': record.years_since_last_promotion,
        'NumCompaniesWorked': record.num_companies_worked,
        'DistanceFromHome': record.distance_from_home,
        'PercentSalaryHike': record.percent_salary_hike,
        'TrainingTimesLastYear': record.training_times_last_year,
        'JobLevel': record.job_level,
    }
    probability = float(record.attrition_probability)
    return {
        'prediction': record.prediction,
        'probability': probability,
        'will_leave': record.will_leave,
        'risk_factors': record.risk_factors,
        'employee_name': record.employee_name,
        'form_data': form_data,
        'prediction_record': record,
    }


def safe_recent_predictions(limit=5):
    try:
        return list(EmployeePrediction.objects.all()[:limit]), None
    except (OperationalError, ProgrammingError):
        return [], 'Prediction history is temporarily unavailable until the database migration finishes.'


def index(request):
    if request.method == 'POST':
        if not load_model_artifacts():
            recent_predictions, db_notice = safe_recent_predictions()
            return render(request, 'predictor/index.html', {
                'choices': CHOICES,
                'recent_predictions': recent_predictions,
                'db_notice': db_notice,
                'error': 'Model files are missing or could not be loaded. Add the files in ml_model/ and ensure all dependencies are installed before running predictions.',
            })

        employee_name = request.POST.get('EmployeeName', 'Employee').strip() or 'Employee'
        raw = build_raw_input(request.POST)
        prediction, probability = run_prediction(raw)
        risk_factors = collect_risk_factors(raw)
        record = save_prediction(employee_name, raw, prediction, probability, risk_factors)
        return render(request, 'predictor/result.html', build_result_context(record))

    recent_predictions, db_notice = safe_recent_predictions()
    return render(request, 'predictor/index.html', {
        'choices': CHOICES,
        'recent_predictions': recent_predictions,
        'db_notice': db_notice,
    })


def history(request):
    try:
        predictions = list(EmployeePrediction.objects.all())
        summary = {
            'total': len(predictions),
            'high_risk': sum(1 for item in predictions if item.prediction == 1),
            'low_risk': sum(1 for item in predictions if item.prediction == 0),
        }
        db_notice = None
    except (OperationalError, ProgrammingError):
        predictions = []
        summary = {
            'total': 0,
            'high_risk': 0,
            'low_risk': 0,
        }
        db_notice = 'History is unavailable because the database schema has not been applied yet.'
    return render(request, 'predictor/history.html', {
        'predictions': predictions,
        'summary': summary,
        'db_notice': db_notice,
    })


def history_detail(request, pk):
    record = get_object_or_404(EmployeePrediction, pk=pk)
    return render(request, 'predictor/result.html', build_result_context(record))
