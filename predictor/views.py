import joblib, numpy as np, os
from django.shortcuts import render

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = joblib.load(os.path.join(BASE, 'ml_model', 'attrition_model.pkl'))
ENCODERS = joblib.load(os.path.join(BASE, 'ml_model', 'label_encoders.pkl'))
FEATURES = joblib.load(os.path.join(BASE, 'ml_model', 'feature_columns.pkl'))

CHOICES = {
    'Gender': ['Male', 'Female'],
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'JobRole': ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'],
    'BusinessTravel': ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
    'EducationField': ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'],
    'MaritalStatus': ['Divorced', 'Married', 'Single'],
    'OverTime': ['No', 'Yes'],
}

RISK_MAP = {
    'OverTime': ('Yes', 'Working overtime significantly increases attrition risk.'),
    'MonthlyIncome': (4000, 'Lower income is a key driver of employee departure.'),
    'Age': (30, 'Younger employees tend to have higher attrition rates.'),
    'YearsAtCompany': (3, 'Employees in first 2 years are at higher risk.'),
    'JobSatisfaction': (2, 'Low job satisfaction strongly predicts attrition.'),
    'DistanceFromHome': (20, 'Long commutes increase the likelihood of leaving.'),
}

def index(request):
    if request.method == 'POST':
        raw = {
            'Age': int(request.POST.get('Age', 30)),
            'Gender': request.POST.get('Gender', 'Male'),
            'Department': request.POST.get('Department', 'Sales'),
            'JobRole': request.POST.get('JobRole', 'Sales Executive'),
            'BusinessTravel': request.POST.get('BusinessTravel', 'Travel_Rarely'),
            'EducationField': request.POST.get('EducationField', 'Life Sciences'),
            'MaritalStatus': request.POST.get('MaritalStatus', 'Single'),
            'OverTime': request.POST.get('OverTime', 'No'),
            'MonthlyIncome': int(request.POST.get('MonthlyIncome', 5000)),
            'JobSatisfaction': int(request.POST.get('JobSatisfaction', 3)),
            'EnvironmentSatisfaction': int(request.POST.get('EnvironmentSatisfaction', 3)),
            'WorkLifeBalance': int(request.POST.get('WorkLifeBalance', 3)),
            'YearsAtCompany': int(request.POST.get('YearsAtCompany', 5)),
            'YearsInCurrentRole': int(request.POST.get('YearsInCurrentRole', 3)),
            'YearsSinceLastPromotion': int(request.POST.get('YearsSinceLastPromotion', 1)),
            'NumCompaniesWorked': int(request.POST.get('NumCompaniesWorked', 2)),
            'DistanceFromHome': int(request.POST.get('DistanceFromHome', 10)),
            'PercentSalaryHike': int(request.POST.get('PercentSalaryHike', 13)),
            'TrainingTimesLastYear': int(request.POST.get('TrainingTimesLastYear', 3)),
            'JobLevel': int(request.POST.get('JobLevel', 2)),
        }
        encoded = dict(raw)
        for col in ['Gender','Department','JobRole','BusinessTravel','EducationField','MaritalStatus','OverTime']:
            encoded[col] = ENCODERS[col].transform([raw[col]])[0]
        feat_vector = np.array([[encoded[f] for f in FEATURES]])
        prediction = MODEL.predict(feat_vector)[0]
        probability = MODEL.predict_proba(feat_vector)[0][1]
        risks = []
        if raw['OverTime'] == 'Yes': risks.append('Working overtime significantly increases attrition risk.')
        if raw['MonthlyIncome'] < 4000: risks.append('Below-average income is a key driver of employee departure.')
        if raw['Age'] < 30: risks.append('Younger employees tend to have higher attrition rates.')
        if raw['YearsAtCompany'] < 3: risks.append('Employees in their first 2 years are at higher risk.')
        if raw['JobSatisfaction'] <= 2: risks.append('Low job satisfaction strongly predicts attrition.')
        if raw['DistanceFromHome'] > 20: risks.append('Long commutes increase the likelihood of leaving.')
        return render(request, 'predictor/result.html', {
            'prediction': prediction, 'probability': round(probability*100,1),
            'will_leave': prediction == 1, 'risk_factors': risks[:3],
            'employee_name': request.POST.get('EmployeeName', 'Employee'),
            'form_data': raw, 'choices': CHOICES,
        })
    return render(request, 'predictor/index.html', {'choices': CHOICES})
