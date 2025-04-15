from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from my_models import db

class User(UserMixin, db.Model):
    table_name = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))


    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)