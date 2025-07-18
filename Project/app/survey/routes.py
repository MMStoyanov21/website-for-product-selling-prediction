from flask import Blueprint, render_template, request, redirect, flash, current_app
from flask_login import login_required
from app.survey.forms import UploadCSVForm
from app.ai_model.regression import LinearRegression

import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

survey = Blueprint('survey', __name__)

@survey.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    upload_form = UploadCSVForm()
    predicted = None

    if request.method == "POST":
        file = request.files.get("csv_file")
        selected_month = int(request.form.get("month", 0))

        if not file or selected_month == 0:
            flash("CSV file and valid month are required.", "danger")
            return redirect(request.url)

        filename = file.filename
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash(f"Could not read CSV: {str(e)}", "danger")
            return redirect(request.url)

        required_cols = ['color', 'type', 'size', 'price', 'sells']
        if not all(col in df.columns for col in required_cols):
            flash(f"CSV must contain columns: {', '.join(required_cols)}", "danger")
            return redirect(request.url)

        # Clean product types
        df['type'] = df['type'].str.strip().replace({'running sho': 'running shoe'})

        df = df.dropna(subset=required_cols)
        df = df[required_cols]
        df = df[df['sells'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]

        next_month = selected_month % 12 + 1
        SEASON_MAP = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        prediction_season = SEASON_MAP[next_month]
        df['season'] = prediction_season

        # Season preference map
        SEASON_PREF = {
            'pants': ['Fall', 'Winter', 'Spring'],
            'skirt': ['Spring', 'Summer'],
            'running shoe': ['Fall', 'Winter', 'Spring', 'Summer']
        }

        # Boost flag if product is in season
        def is_preferred(row):
            return row['season'] in SEASON_PREF.get(row['type'].strip().lower(), [])

        df['in_season'] = df.apply(is_preferred, axis=1).astype(int)

        # Features and target
        features = ['color', 'type', 'size', 'price', 'season', 'in_season']
        target = 'sells'
        X = df[features]
        y = df[target].astype(float)

        # Encode categorical
        categorical_features = ['color', 'type', 'season']
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')

        X_encoded = preprocessor.fit_transform(X).astype(float)

        # Train
        model = LinearRegression(lr=0.001, epochs=5000)
        model.fit(X_encoded, y)
        predictions = model.predict(X_encoded)

        # Results
        df['predicted_sells'] = predictions.round(2)
        season_mean = y.mean()
        df['season_boosted'] = df['predicted_sells'] > season_mean
        predicted = round(float(predictions.mean()), 2)

        return render_template("predict_result.html", score=predicted, df=df.to_dict(orient='records'))

    return render_template("predict.html", upload_form=upload_form)
