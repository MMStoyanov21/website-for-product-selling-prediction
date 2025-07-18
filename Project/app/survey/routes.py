from flask import Blueprint, render_template, request, redirect, flash, current_app
from flask_login import login_required
from app.survey.forms import UploadCSVForm
from app.ai_model.regression import LinearRegression

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

        # Normalize 'type' to lowercase and strip spaces
        df['type'] = df['type'].str.strip().str.lower().replace({'running sho': 'running shoe'})

        # Convert numeric columns safely
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['sells'] = pd.to_numeric(df['sells'], errors='coerce')
        df['size'] = pd.to_numeric(df['size'], errors='coerce')

        # Drop rows with missing or invalid values in required columns
        df = df.dropna(subset=required_cols)

        # Filter 'sells' to ensure numeric values (already converted)
        df = df[df['sells'].apply(lambda x: np.isfinite(x))]

        # Map next month to season (lowercase strings)
        next_month = selected_month % 12 + 1
        SEASON_MAP = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        }
        prediction_season = SEASON_MAP[next_month]
        df['season'] = prediction_season

        # Season preference map (all lowercase)
        SEASON_PREF = {
            'pants': ['fall', 'winter', 'spring'],
            'skirt': ['spring', 'summer'],
            'running shoe': ['fall', 'winter', 'spring', 'summer']
        }

        # Boost flag if product is in season
        def is_preferred(row):
            product_type = row['type']
            season = row['season']
            preferred_seasons = SEASON_PREF.get(product_type, [])
            return season in preferred_seasons

        df['in_season'] = df.apply(is_preferred, axis=1).astype(int)

        # DEBUG PRINT: Check the types, seasons, and in_season flags
        print(df[['type', 'season', 'in_season']])

        # Features and target
        categorical_features = ['color', 'type', 'season']
        numeric_features = ['size', 'price', 'in_season']
        target = 'sells'

        X_cat = df[categorical_features]
        X_num = df[numeric_features]
        y = df[target].astype(float)

        # Encode categorical features
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='drop')

        X_cat_encoded = preprocessor.fit_transform(X_cat)
        if hasattr(X_cat_encoded, "toarray"):
            X_cat_encoded = X_cat_encoded.toarray()

        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)

        X_encoded = np.hstack([X_cat_encoded, X_num_scaled])

        if np.isnan(X_encoded).any() or np.isinf(X_encoded).any():
            flash("Input features contain NaN or infinite values after processing.", "danger")
            return redirect(request.url)

        if y.isna().any() or np.isinf(y).any():
            flash("Target 'sells' contains NaN or infinite values.", "danger")
            return redirect(request.url)

        # Train model
        model = LinearRegression(lr=0.001, epochs=5000)
        model.fit(X_encoded, y)
        predictions = model.predict(X_encoded)

        df['predicted_sells'] = predictions.round(2)

        # Calculate overall mean sells (original target)
        season_mean = y.mean()

        # Boost flag: True only if product is in season AND predicted sells > mean sells
        df['season_boosted'] = (df['in_season'] == 1) & (df['predicted_sells'] > season_mean)

        predicted = int(round(predictions.mean()))

        return render_template("predict_result.html", score=predicted, df=df.to_dict(orient='records'))

    return render_template("predict.html", upload_form=upload_form)
