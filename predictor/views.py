import csv
import os

import joblib
import pandas as pd
from django.http import FileResponse, Http404
from django.shortcuts import render

try:
    import shap
except ImportError:
    shap = None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = joblib.load(os.path.join(BASE, 'ml_model', 'attrition_model.pkl'))
ENCODERS = joblib.load(os.path.join(BASE, 'ml_model', 'label_encoders.pkl'))
FEATURES = joblib.load(os.path.join(BASE, 'ml_model', 'feature_columns.pkl'))
SAMPLE_CSV_PATH = os.path.join(BASE, 'sample_data', 'employee_attrition_batch_sample.csv')

CHOICES = {
    'Gender': ['Male', 'Female'],
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'JobRole': ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'],
    'BusinessTravel': ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
    'EducationField': ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'],
    'MaritalStatus': ['Divorced', 'Married', 'Single'],
    'OverTime': ['No', 'Yes'],
}

CSV_COLUMNS = [
    'EmployeeName',
    'Age',
    'Gender',
    'Department',
    'JobRole',
    'BusinessTravel',
    'EducationField',
    'MaritalStatus',
    'OverTime',
    'MonthlyIncome',
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'WorkLifeBalance',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'NumCompaniesWorked',
    'DistanceFromHome',
    'PercentSalaryHike',
    'TrainingTimesLastYear',
    'JobLevel',
]

FEATURE_LABELS = {
    'Age': 'Age',
    'Gender': 'Gender',
    'Department': 'Department',
    'JobRole': 'Job Role',
    'BusinessTravel': 'Business Travel',
    'EducationField': 'Education Field',
    'MaritalStatus': 'Marital Status',
    'OverTime': 'Overtime',
    'MonthlyIncome': 'Monthly Income',
    'JobSatisfaction': 'Job Satisfaction',
    'EnvironmentSatisfaction': 'Environment Satisfaction',
    'WorkLifeBalance': 'Work-Life Balance',
    'YearsAtCompany': 'Years at Company',
    'YearsInCurrentRole': 'Years in Current Role',
    'YearsSinceLastPromotion': 'Years Since Last Promotion',
    'NumCompaniesWorked': 'Companies Worked',
    'DistanceFromHome': 'Distance From Home',
    'PercentSalaryHike': 'Percent Salary Hike',
    'TrainingTimesLastYear': 'Training Sessions Last Year',
    'JobLevel': 'Job Level',
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


def priority_details(probability):
    if probability >= 65:
        return 'Critical', 'Immediate outreach recommended within 24 hours.'
    if probability >= 45:
        return 'High', 'Manager and HR review should be scheduled this week.'
    if probability >= 25:
        return 'Medium', 'Monitor closely and review workload, growth, and engagement.'
    return 'Low', 'No urgent intervention required. Continue routine engagement.'


def risk_band(probability):
    if probability >= 65:
        return 'Severe', 'high'
    if probability >= 45:
        return 'Elevated', 'warn'
    if probability >= 25:
        return 'Watchlist', 'mid'
    return 'Stable', 'low'


def build_raw_input(source):
    return {
        'Age': int(source.get('Age', 30)),
        'Gender': source.get('Gender', 'Male'),
        'Department': source.get('Department', 'Sales'),
        'JobRole': source.get('JobRole', 'Sales Executive'),
        'BusinessTravel': source.get('BusinessTravel', 'Travel_Rarely'),
        'EducationField': source.get('EducationField', 'Technical Degree'),
        'MaritalStatus': source.get('MaritalStatus', 'Single'),
        'OverTime': source.get('OverTime', 'No'),
        'MonthlyIncome': int(source.get('MonthlyIncome', 5000)),
        'JobSatisfaction': int(source.get('JobSatisfaction', 3)),
        'EnvironmentSatisfaction': int(source.get('EnvironmentSatisfaction', 3)),
        'WorkLifeBalance': int(source.get('WorkLifeBalance', 3)),
        'YearsAtCompany': int(source.get('YearsAtCompany', 5)),
        'YearsInCurrentRole': int(source.get('YearsInCurrentRole', 3)),
        'YearsSinceLastPromotion': int(source.get('YearsSinceLastPromotion', 1)),
        'NumCompaniesWorked': int(source.get('NumCompaniesWorked', 2)),
        'DistanceFromHome': int(source.get('DistanceFromHome', 10)),
        'PercentSalaryHike': int(source.get('PercentSalaryHike', 13)),
        'TrainingTimesLastYear': int(source.get('TrainingTimesLastYear', 3)),
        'JobLevel': int(source.get('JobLevel', 2)),
    }


def predict_attrition(raw):
    encoded = dict(raw)
    for col in ['Gender', 'Department', 'JobRole', 'BusinessTravel', 'EducationField', 'MaritalStatus', 'OverTime']:
        encoded[col] = ENCODERS[col].transform([raw[col]])[0]
    feat_vector = pd.DataFrame([[encoded[f] for f in FEATURES]], columns=FEATURES)
    prediction = int(MODEL.predict(feat_vector)[0])
    probability = round(float(MODEL.predict_proba(feat_vector)[0][1]) * 100, 1)
    return prediction, probability, feat_vector


def build_shap_explanations(raw, feat_vector):
    if shap is None:
        return [], 'Install the SHAP package to render per-employee model explanations.'

    try:
        explainer = shap.TreeExplainer(MODEL)
        shap_values = explainer.shap_values(feat_vector)
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        elif len(getattr(shap_values, 'shape', [])) == 3:
            values = shap_values[0, :, 1]
        else:
            values = shap_values[0]

        explanations = []
        for feature_name, shap_value in zip(FEATURES, values):
            magnitude = abs(float(shap_value))
            if magnitude == 0:
                continue

            direction = 'increases' if shap_value > 0 else 'reduces'
            explanations.append({
                'feature': FEATURE_LABELS.get(feature_name, feature_name),
                'raw_value': raw[feature_name],
                'direction': direction,
                'impact': round(magnitude, 4),
            })

        explanations.sort(key=lambda item: item['impact'], reverse=True)
        return explanations[:5], None
    except Exception as exc:
        return [], f'SHAP explanation could not be generated: {exc}'


def validate_csv_columns(fieldnames):
    if not fieldnames:
        raise ValueError('The uploaded CSV is empty.')

    missing_columns = [column for column in CSV_COLUMNS if column not in fieldnames]
    if missing_columns:
        raise ValueError('Missing CSV columns: ' + ', '.join(missing_columns))


def process_batch_upload(uploaded_file):
    decoded_lines = uploaded_file.read().decode('utf-8-sig').splitlines()
    reader = csv.DictReader(decoded_lines)
    validate_csv_columns(reader.fieldnames)

    results = []
    for index, row in enumerate(reader, start=1):
        employee_name = (row.get('EmployeeName') or '').strip() or f'Employee {index}'
        try:
            raw = build_raw_input(row)
            prediction, probability, feat_vector = predict_attrition(raw)
            shap_highlights, shap_notice = build_shap_explanations(raw, feat_vector)
        except Exception as exc:
            raise ValueError(f'Row {index} for {employee_name}: {exc}') from exc

        priority_label, priority_note = priority_details(probability)

        results.append({
            'priority_rank': 0,
            'priority_label': priority_label,
            'priority_note': priority_note,
            'risk_band_label': risk_band(probability)[0],
            'risk_band_tone': risk_band(probability)[1],
            'employee_name': employee_name,
            'department': raw['Department'],
            'job_role': raw['JobRole'],
            'prediction': prediction,
            'probability': probability,
            'will_leave': prediction == 1,
            'risk_factors': collect_risk_factors(raw),
            'shap_highlights': shap_highlights[:3],
            'shap_notice': shap_notice,
        })

    if not results:
        raise ValueError('The uploaded CSV does not contain any employee rows.')

    results.sort(key=lambda item: item['probability'], reverse=True)
    for rank, item in enumerate(results, start=1):
        item['priority_rank'] = rank

    high_risk_count = sum(1 for result in results if result['will_leave'])
    avg_probability = round(sum(result['probability'] for result in results) / len(results), 1)
    total_employees = len(results)
    high_risk_pct = round((high_risk_count / total_employees) * 100, 1)
    low_risk_pct = round(100 - high_risk_pct, 1)

    return {
        'top_priority': results[:5],
        'results': results,
        'summary': {
            'total_employees': total_employees,
            'high_risk_count': high_risk_count,
            'low_risk_count': total_employees - high_risk_count,
            'high_risk_pct': high_risk_pct,
            'low_risk_pct': low_risk_pct,
            'avg_probability': avg_probability,
        },
    }


def download_sample_csv(request):
    if not os.path.exists(SAMPLE_CSV_PATH):
        raise Http404('Sample CSV not found.')

    return FileResponse(
        open(SAMPLE_CSV_PATH, 'rb'),
        as_attachment=True,
        filename='employee_attrition_batch_sample.csv',
        content_type='text/csv',
    )


def index(request):
    if request.method == 'POST':
        if request.FILES.get('csv_file'):
            try:
                batch_context = process_batch_upload(request.FILES['csv_file'])
            except ValueError as exc:
                return render(request, 'predictor/index.html', {
                    'choices': CHOICES,
                    'error': str(exc),
                })
            return render(request, 'predictor/batch_result.html', batch_context)

        raw = build_raw_input(request.POST)
        prediction, probability, feat_vector = predict_attrition(raw)
        risks = collect_risk_factors(raw)
        shap_explanations, shap_notice = build_shap_explanations(raw, feat_vector)
        priority_label, priority_note = priority_details(probability)
        return render(request, 'predictor/result.html', {
            'prediction': prediction,
            'probability': probability,
            'retention_probability': round(100 - probability, 1),
            'will_leave': prediction == 1,
            'risk_factors': risks,
            'priority_label': priority_label,
            'priority_note': priority_note,
            'risk_band_label': risk_band(probability)[0],
            'risk_band_tone': risk_band(probability)[1],
            'shap_explanations': shap_explanations,
            'shap_notice': shap_notice,
            'employee_name': request.POST.get('EmployeeName', 'Employee'),
            'form_data': raw,
            'choices': CHOICES,
        })

    return render(request, 'predictor/index.html', {'choices': CHOICES})
