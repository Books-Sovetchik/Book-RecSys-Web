from my_models import db


class Rating(db.Model):
    table_name = 'Rating'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    source_book = db.Column(db.String)
    recommended_book = db.Column(db.String)
    rate = db.Column(db.Integer)
