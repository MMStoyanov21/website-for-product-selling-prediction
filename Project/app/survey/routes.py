import requests
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
        selected_month = int(request.form.get("month", 0))
        selected_type = request.form.get("product_type")
        file = request.files.get("csv_file")
        session_file = request.form.get("csv_file_path")

        # Step 1: Handle CSV upload if it's a new file
        if file:
            filename = file.filename
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        elif session_file:
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], session_file)
        else:
            flash("CSV file is required.", "danger")
            return redirect(request.url)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash(f"Could not read CSV: {str(e)}", "danger")
            return redirect(request.url)

        required_cols = ['color', 'type', 'size', 'price', 'sells', 'preferance']
        if not all(col in df.columns for col in required_cols):
            flash(f"CSV must contain columns: {', '.join(required_cols)}", "danger")
            return redirect(request.url)

        # Clean and preprocess data
        df['type'] = df['type'].str.strip().str.lower().replace({'running sho': 'running shoe'})
        df['preferance'] = df['preferance'].str.strip().str.lower()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['sells'] = pd.to_numeric(df['sells'], errors='coerce')
        df['size'] = pd.to_numeric(df['size'], errors='coerce')

        df = df.dropna(subset=required_cols)
        df = df[df['sells'].apply(lambda x: np.isfinite(x))]

        # Add season column
        next_month = selected_month % 12 + 1
        SEASON_MAP = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        }
        prediction_season = SEASON_MAP[next_month]
        df['season'] = prediction_season

        # Mark if the product is in season
        df['in_season'] = df['preferance'].apply(
            lambda pref: prediction_season in pref or pref == 'all'
        ).astype(int)

        # If user hasn't selected a product type yet, show the selection page
        if not selected_type:
            product_types = sorted(df['type'].unique())
            return render_template("predict_choose_product.html",
                                   product_names=product_types,
                                   month=selected_month,
                                   csv_file_path=os.path.basename(filepath))

        # Filter data by selected product type
        df = df[df['type'] == selected_type]

        categorical_features = ['color', 'type', 'season']
        numeric_features = ['size', 'price', 'in_season']
        target = 'sells'

        X_cat = df[categorical_features]
        X_num = df[numeric_features]
        y = df[target].astype(float)

        # One-hot encode categorical features
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='drop')

        X_cat_encoded = preprocessor.fit_transform(X_cat)
        if hasattr(X_cat_encoded, "toarray"):
            X_cat_encoded = X_cat_encoded.toarray()

        # Scale numeric features
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)

        # Combine features
        X_encoded = np.hstack([X_cat_encoded, X_num_scaled])

        # Sanity check
        if np.isnan(X_encoded).any() or np.isinf(X_encoded).any():
            flash("Processed data contains NaN or infinite values.", "danger")
            return redirect(request.url)

        # Train model and predict
        model = LinearRegression(lr=0.001, epochs=5000)
        model.fit(X_encoded, y)
        predictions = model.predict(X_encoded)

        df['predicted_sells'] = predictions.round(2)
        season_mean = y.mean()
        df['season_boosted'] = (df['in_season'] == 1) & (df['predicted_sells'] > season_mean)

        predicted = int(round(predictions.mean()))

        return render_template("predict_result.html",
                               score=predicted,
                               df=df.to_dict(orient='records'),
                               product=selected_type)

    return render_template("predict.html", upload_form=upload_form)
@survey.route("/predict-url", methods=["GET", "POST"])
def predict_url():
    predicted = None
    data_table = None
    error = None

    if request.method == "POST":
        csv_url = request.form.get("csv_url")
        selected_month = int(request.form.get("month", 0))
        selected_type = request.form.get("product_type")

        try:
            # Fetch the CSV
            response = requests.get(csv_url)
            response.raise_for_status()

            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            required_cols = ['color', 'type', 'size', 'price', 'sells', 'preferance']
            if not all(col in df.columns for col in required_cols):
                error = f"CSV must contain columns: {', '.join(required_cols)}"
                return render_template("predict_url.html", error=error)

            # Preprocessing
            df['type'] = df['type'].str.strip().str.lower().replace({'running sho': 'running shoe'})
            df['preferance'] = df['preferance'].str.strip().str.lower()
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['sells'] = pd.to_numeric(df['sells'], errors='coerce')
            df['size'] = pd.to_numeric(df['size'], errors='coerce')

            df = df.dropna(subset=required_cols)
            df = df[df['sells'].apply(lambda x: np.isfinite(x))]

            # Add season column
            next_month = selected_month % 12 + 1
            SEASON_MAP = {
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall'
            }
            prediction_season = SEASON_MAP[next_month]
            df['season'] = prediction_season

            df['in_season'] = df['preferance'].apply(
                lambda pref: prediction_season in pref or pref == 'all'
            ).astype(int)

            # Step 1: Ask user to pick product type if not already selected
            if not selected_type:
                product_types = sorted(df['type'].unique())
                return render_template("predict_choose_product.html",
                                       product_names=product_types,
                                       month=selected_month,
                                       csv_url=csv_url)

            # Filter by selected product type
            df = df[df['type'] == selected_type]

            categorical_features = ['color', 'type', 'season']
            numeric_features = ['size', 'price', 'in_season']
            target = 'sells'

            X_cat = df[categorical_features]
            X_num = df[numeric_features]
            y = df[target].astype(float)

            # One-hot encode categorical features
            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ], remainder='drop')

            X_cat_encoded = preprocessor.fit_transform(X_cat)
            if hasattr(X_cat_encoded, "toarray"):
                X_cat_encoded = X_cat_encoded.toarray()

            # Scale numeric features
            scaler = StandardScaler()
            X_num_scaled = scaler.fit_transform(X_num)

            X_encoded = np.hstack([X_cat_encoded, X_num_scaled])

            if np.isnan(X_encoded).any() or np.isinf(X_encoded).any():
                error = "Processed data contains NaN or infinite values."
                return render_template("predict_url.html", error=error)

            model = LinearRegression(lr=0.001, epochs=5000)
            model.fit(X_encoded, y)
            predictions = model.predict(X_encoded)

            df['predicted_sells'] = predictions.round(2)
            season_mean = y.mean()
            df['season_boosted'] = (df['in_season'] == 1) & (df['predicted_sells'] > season_mean)

            predicted = int(round(predictions.mean()))

            return render_template("predict_result.html",
                                   score=predicted,
                                   df=df.to_dict(orient='records'),
                                   product=selected_type)

        except Exception as e:
            error = f"Failed to process CSV: {str(e)}"

    return render_template("predict_url.html", error=error, data_table=data_table)
