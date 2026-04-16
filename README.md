# Employee Attrition Predictor
<img width="575" height="807" alt="image" src="https://github.com/user-attachments/assets/c7ce89cc-3c61-420e-9d2f-5b4cb80f33a4" />

A Django-based web application that predicts whether an employee is likely to leave the company based on HR and workplace attributes made for my internship at MazeQube Software Technologies Pvt. Ltd.

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

- `vercel.json` for Vercel routing and function configuration
- `api/index.py` as the Vercel Python entrypoint

### Deploying on Vercel

1. Push the latest code to GitHub.
2. In Vercel, create a new project and import this repository.
3. Keep the detected framework as `Other` if prompted.
4. Set these environment variables in Vercel:

```text
SECRET_KEY=your-secure-secret-key
DEBUG=False
ALLOWED_HOSTS=.vercel.app
CSRF_TRUSTED_ORIGINS=https://<your-project>.vercel.app
```

5. Deploy the project.

Before deploying predictions, make sure the model files are available in the deployment bundle.

### Important Model File Note

The prediction view depends on these files:

- `ml_model/attrition_model.pkl`
- `ml_model/label_encoders.pkl`
- `ml_model/feature_columns.pkl`

These files are committed in this repository so Vercel can bundle them with the Python function.

### Persistence Note

This app is currently configured with SQLite only.

Inference from Vercel's Functions docs: Vercel recommends persisting writes to object storage or another external database rather than relying on local function files, so SQLite should not be treated as a persistent production datastore on Vercel.

## Notes

- The default database is SQLite.
- `db.sqlite3` is ignored by Git and is not committed.
- The current Django settings file has `DEBUG = True` and `ALLOWED_HOSTS = ['*']`, which should be tightened before production use.
