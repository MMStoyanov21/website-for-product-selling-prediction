from flask import Blueprint, render_template, request, redirect, flash, url_for, current_app
from flask_login import login_required, current_user
from app.survey.forms import UploadCSVForm
from app.models import Survey, db
from app.ai_model.regression import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

survey = Blueprint('survey', __name__)

@survey.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    upload_form = UploadCSVForm()
    plot_url = None
    predicted = None

    if upload_form.validate_on_submit():
        print("✅ Form validated")
        file = upload_form.csv_file.data
        filename = file.filename
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash(f"Could not read CSV: {str(e)}", "danger")
            return redirect(request.url)

        if 'hours' not in df.columns:
            flash("CSV must contain a 'hours' column", "danger")
            return redirect(request.url)

        hours_data = df['hours'].dropna().values
        if len(hours_data) == 0:
            flash("CSV file is empty or has no valid 'hours' data", "danger")
            return redirect(request.url)

        days = np.arange(1, len(hours_data) + 1)
        model = LinearRegression()
        model.fit(days, hours_data)
        predicted_raw = model.predict(len(hours_data) + 1)
        predicted = round(predicted_raw * 30, 2)
        if predicted > 100:
            predicted = 100

        avg_hours = float(np.mean(hours_data))
        entry = Survey(hours_studied=avg_hours, predicted_score=predicted, author=current_user)
        db.session.add(entry)
        db.session.commit()

        plt.figure()
        plt.scatter(days, hours_data, color='blue', label='Observed')
        plt.plot(days, model.predict(days), color='red', label='Regression line')
        plt.xlabel("Day")
        plt.ylabel("Hours Studied")
        plt.title("Linear Regression from CSV")
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plot_url = f"data:image/png;base64,{plot_data}"
        plt.close()

        return render_template("predict_result.html", score=predicted, plot_url=plot_url)

    if request.method == "POST":
        print("❌ Form not valid")
        print(upload_form.errors)

    return render_template("predict.html", upload_form=upload_form)
