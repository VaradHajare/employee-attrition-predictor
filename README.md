# Employee Attrition Predictor

A Django-based web application that predicts whether an employee is likely to leave the company based on HR and workplace attributes.

This project was built as an HR analytics internship project and provides:

- a form-driven interface for entering employee details
- an ML-powered attrition prediction
- a probability score for attrition risk
- a short list of key risk factors influencing the result

## Tech Stack

- Python
- Django
- scikit-learn
- NumPy
- pandas
- joblib
- SQLite

## Features

- Predicts employee attrition using a trained machine learning model
- Accepts employee demographic, role, compensation, and satisfaction inputs
- Shows attrition probability as a percentage
- Highlights top risk signals such as overtime, low income, low job satisfaction, and short tenure
- Includes deployment files for Railway and Gunicorn

## Project Structure

```text
attrition_app/
|-- attrition_project/      Django project settings and URLs
|-- predictor/              App logic and prediction view
|-- templates/              HTML templates
|-- static/                 Static assets
|-- ml_model/               Trained model and encoders
|-- manage.py
|-- requirements.txt
|-- Procfile
|-- railway.toml
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/VaradHajare/employee-attrition-predictor.git
cd employee-attrition-predictor
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run database migrations:

```bash
python manage.py migrate
```

5. Start the development server:

```bash
python manage.py runserver
```

6. Open the app in your browser:

```text
http://127.0.0.1:8000/
```

## Model Files

The application expects trained model artifacts inside the `ml_model/` folder:

- `attrition_model.pkl`
- `label_encoders.pkl`
- `feature_columns.pkl`

These `.pkl` files are currently ignored by Git, so they must exist locally or be added through a secure deployment workflow before the app can run predictions.

## Dependencies

Current Python dependencies are listed in `requirements.txt`:

- Django 4.2+
- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.24+
- joblib 1.3+
- gunicorn 21.0+
- whitenoise 6.6+

## Deployment

This repository includes:

- `Procfile` for process startup
- `railway.toml` for Railway deployment configuration

### Deploying on Railway

1. Push the latest code to GitHub.
2. In Railway, create a new project and choose `Deploy from GitHub repo`.
3. Select this repository.
4. Add a PostgreSQL service to the Railway project.
5. In your app service variables, set:

```text
SECRET_KEY=your-secure-secret-key
DEBUG=False
ALLOWED_HOSTS=.railway.app
CSRF_TRUSTED_ORIGINS=https://<your-app-domain>.railway.app
PGDATABASE=${{Postgres.PGDATABASE}}
PGUSER=${{Postgres.PGUSER}}
PGPASSWORD=${{Postgres.PGPASSWORD}}
PGHOST=${{Postgres.PGHOST}}
PGPORT=${{Postgres.PGPORT}}
```

6. Deploy the service.
7. In Railway Networking, generate a public domain for the app.

Before deploying predictions, make sure the model files are available in the deployment environment.

### Important Model File Note

The prediction view depends on these files:

- `ml_model/attrition_model.pkl`
- `ml_model/label_encoders.pkl`
- `ml_model/feature_columns.pkl`

These files are ignored by Git right now, so Railway will not receive them from GitHub unless you change that workflow.

You have two practical options:

- commit the model files to the repository if you are okay storing them in GitHub
- provide them another way in production, such as a mounted volume or startup download step

## Notes

- The default database is SQLite.
- `db.sqlite3` is ignored by Git and is not committed.
- The current Django settings file has `DEBUG = True` and `ALLOWED_HOSTS = ['*']`, which should be tightened before production use.

## Author

Varad Vijay Hajare  
AISSMS College of Engineering, Pune
