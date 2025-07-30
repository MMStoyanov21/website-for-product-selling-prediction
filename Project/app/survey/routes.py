import joblib
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

        # Handle CSV file upload or reuse
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

        required_cols = ['type', 'color', 'price', 'sells']
        if not all(col in df.columns for col in required_cols):
            flash(f"CSV must contain columns: {', '.join(required_cols)}", "danger")
            return redirect(request.url)

        # Clean and preprocess
        df['type'] = df['type'].str.strip().str.lower()
        df['color'] = df['color'].str.strip().str.lower()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['sells'] = pd.to_numeric(df['sells'], errors='coerce')
        df = df.dropna(subset=required_cols)

        # ADD MONTH COLUMN HERE (before feature engineering)
        df['month'] = selected_month

        df['preferance'] = 'all'
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

        if not selected_type:
            product_types = sorted(df['type'].unique())
            return render_template("predict_choose_product.html",
                                   product_names=product_types,
                                   month=selected_month,
                                   csv_file_path=os.path.basename(filepath))

        df = df[df['type'] == selected_type]

        if df.empty:
            flash("No data found for selected product type.", "warning")
            return redirect(request.url)

        # Load model
        model_path = os.path.join(current_app.root_path, "ai_model", "custom_linear_model.pkl")
        try:
            model_data = joblib.load(model_path)
        except Exception as e:
            flash(f"Failed to load model: {str(e)}", "danger")
            return redirect(request.url)

        # Select features for prediction, now month is guaranteed to exist
        X_user = df[['type', 'month', 'color', 'price', 'in_season']].copy()

        # One-hot encode categorical variables exactly as in training
        X_user_encoded = pd.get_dummies(X_user, columns=['type', 'month', 'color'])

        # Align columns to model columns, add missing columns with zeros
        for col in model_data['columns']:
            if col not in X_user_encoded.columns:
                X_user_encoded[col] = 0
        # Reorder columns to match the model
        X_user_encoded = X_user_encoded[model_data['columns']]

        # Normalize features
        X_mean = model_data['mean']
        X_std = model_data['std']
        X = (X_user_encoded - X_mean) / X_std

        # Predict
        weights = model_data['weights']
        bias = model_data['bias']
        predictions = np.dot(X, weights) + bias

        df['predicted_sells'] = predictions.round(2)
        predicted = int(round(predictions.mean()))

        return render_template("predict_result.html",
                               score=predicted,
                               df=df.to_dict(orient='records'),
                               product=selected_type)

    return render_template("predict.html", upload_form=upload_form)


"""@survey.route("/predict-url", methods=["GET", "POST"])
def predict_url():
    predicted = None
    error = None

    if request.method == "POST":
        # Default to Novistoki URL if none is provided
        csv_url = request.form.get("csv_url") or "https://www.novistoki.com/site/check-products.php?type=csv"
        selected_month = int(request.form.get("month", 0))
        selected_type = request.form.get("product_type")

        try:
            # Fetch the file from the URL
            response = requests.get(csv_url)
            response.raise_for_status()

            from io import StringIO
            if csv_url.lower().endswith('.json'):
                df = pd.read_json(StringIO(response.text))
            elif csv_url.lower().endswith('.csv'):
                df = pd.read_csv(StringIO(response.text))
            else:
                error = "URL must point to a .csv or .json file."
                return render_template("predict_url.html", error=error)

            # Required columns
            required_cols = [
                'product_type', 'product_price', 'product_price_supply',
                'product_price_profit', 'option', 'date', 'date_month',
                'date_year', 'date_month_year'
            ]
            if not all(col in df.columns for col in required_cols):
                error = f"File must contain columns: {', '.join(required_cols)}"
                return render_template("predict_url.html", error=error)

            # Preprocessing
            df['product_type'] = df['product_type'].str.strip().str.lower().replace({'running sho': 'running shoe'})
            df['option'] = df['option'].str.strip().str.lower()
            for col in ['product_price', 'product_price_supply', 'product_price_profit', 'date_month']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=required_cols)
            df = df[np.isfinite(df['product_price_profit'])]

            # Season determination
            SEASON_MAP = {
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall'
            }
            prediction_season = SEASON_MAP.get(selected_month % 12 + 1, 'unknown')
            df['season'] = prediction_season
            df['in_season'] = df['option'].apply(lambda x: prediction_season in x or x == 'all').astype(int)

            # Ask user to choose a product type if not selected yet
            if not selected_type:
                product_types = sorted(df['product_type'].unique())
                return render_template("predict_choose_product.html",
                                       product_names=product_types,
                                       month=selected_month,
                                       csv_url=csv_url)

            # Filter for the selected product type
            df = df[df['product_type'] == selected_type]

            # Set up features
            categorical_features = ['product_type', 'season']
            numeric_features = ['product_price', 'product_price_supply', 'product_price_profit', 'in_season']
            target = 'product_price_profit'

            X_cat = df[categorical_features]
            X_num = df[numeric_features]
            y = df[target].astype(float)

            # Encode and scale
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
                error = "Processed data contains NaN or infinite values."
                return render_template("predict_url.html", error=error)

            # Train and predict
            model = LinearRegression(lr=0.001, epochs=5000)
            model.fit(X_encoded, y)
            predictions = model.predict(X_encoded)

            df['predicted_profit'] = df['product_price_profit']
            df.loc[df['in_season'] == 1, 'predicted_profit'] = predictions[df['in_season'] == 1].round(2)

            season_mean = y.mean()
            df['season_boosted'] = (df['in_season'] == 1) & (df['predicted_profit'] > season_mean)

            in_season_df = df[df['in_season'] == 1]
            predicted = int(round(in_season_df['predicted_profit'].mean())) if not in_season_df.empty else 0

            return render_template("predict_result.html",
                                   score=predicted,
                                   df=df.to_dict(orient='records'),
                                   product=selected_type)

        except Exception as e:
            error = f"Failed to process file: {str(e)}"

    # Default GET render
    return render_template("predict_url.html", error=error)
"""