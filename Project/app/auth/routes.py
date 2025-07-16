"""
Module: auth/routes.py
Description: Blueprint routes for user authentication: register, login, logout, and admin delete.
"""

import os
from flask import Blueprint, render_template, redirect, url_for, flash, session, request
from app import db, bcrypt
from flask_login import login_user, logout_user, current_user
from app.models import User
from app.auth.forms import RegistrationForm, LoginForm

auth = Blueprint('auth', __name__)



@auth.route("/register", methods=['GET', 'POST'])
@auth.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))

    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=hashed_pw,
            confirmed=True
        )
        db.session.add(user)
        db.session.commit()

        login_user(user)  # ✅ Automatically log in the new user
        session['is_admin'] = False  # Optional: set admin flag if needed
        flash('Account created and you are now logged in.', 'success')
        return redirect(url_for('main.home'))

    return render_template('register.html', form=form)

@auth.route("/login", methods=['GET', 'POST'])
def login():

    if current_user.is_authenticated:
        return redirect(url_for('main.home'))

    form = LoginForm()
    if form.validate_on_submit():


        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            session['is_admin'] = False
            flash('Logged in successfully.', 'success')
            return redirect(url_for('main.home'))
        else:
            flash('Login unsuccessful. Check credentials.', 'danger')

    return render_template('login.html', form=form)

@auth.route("/logout")
def logout():

    logout_user()
    session.pop('is_admin', None)
    return redirect(url_for('main.home'))

