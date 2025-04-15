from my_models import db

class TappedBook(db.Model):
    table_name = 'tapped_book'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    book_title = db.Column(db.String)
