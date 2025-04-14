from app import db
from flask_login import UserMixin
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password