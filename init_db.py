from app import app
from app import db
from my_models import User

with app.app_context():
    db.create_all()
