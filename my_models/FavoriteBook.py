from my_models import db


class FavoriteBook(db.Model):
    table_name = 'FavoriteBook'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    book_title = db.Column(db.String)
