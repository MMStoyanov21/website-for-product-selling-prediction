from flask_login import UserMixin
from app import db, login_manager

@login_manager.user_loader
def load_user(user_id):
    """
    Load a user by ID for Flask-Login.
    """
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    """
    User model with authentication credentials and relationships.
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='student')
    confirmed = db.Column(db.Boolean, default=False)

    surveys = db.relationship('Survey', backref='author', lazy=True, cascade='all, delete-orphan')
    uploads = db.relationship('Upload', backref='user', lazy=True, cascade='all, delete-orphan')
    estimations = db.relationship('Estimation', backref='user', lazy=True, cascade='all, delete-orphan')
    feedbacks = db.relationship('Feedback', backref='author', lazy=True, cascade='all, delete-orphan')