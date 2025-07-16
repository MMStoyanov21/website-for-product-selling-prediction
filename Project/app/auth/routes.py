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

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin01")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin_acc")

@auth.route("/register", methods=['GET', 'POST'])
def register():
    """
    Handle new user registration.
    """
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))

    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data,
                    password=hashed_pw, confirmed=True)
        db.session.add(user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register.html', form=form)

@auth.route("/login", methods=['GET', 'POST'])
def login():
    """
    Handle user login and admin session assignment.
    """
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))

    form = LoginForm()
    if form.validate_on_submit():
        if form.username.data == ADMIN_USERNAME and form.password.data == ADMIN_PASSWORD:
            session['is_admin'] = True
            flash('Admin logged in successfully.', 'success')
            return redirect(url_for('main.home'))

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
    """
    Log out current user and clear admin session.
    """
    logout_user()
    session.pop('is_admin', None)
    return redirect(url_for('main.home'))

@auth.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    """
    Admin-only route to delete a user and cascade related data.
    """
    if session.get('is_admin'):
        user = User.query.get_or_404(user_id)
        db.session.delete(user)
        db.session.commit()
        flash('User and associated data deleted.', 'info')
    else:
        flash('Unauthorized access.', 'danger')
    return redirect(url_for('main.home'))
