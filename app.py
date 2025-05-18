import sys
import os

import numpy as np
from flask import Flask, jsonify, redirect, flash
import pandas as pd
from flask import request, render_template, url_for
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from jmespath import search

from my_models import db
from my_models.Rating import Rating
from my_models.User import User
from my_models.FavoriteBook import FavoriteBook
from my_models.TappedBook import TappedBook
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Book-RecSys', 'src'))
sys.path.append(SRC_PATH)

from models.modules import EmbeddingsProducer
from models.modules import SearchBooksByTitle
from models.modules import Llm
# from models.model import Bibliotekar


load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DB_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

fs_books = np.load("./data/embeddings/fs_embds.npz", allow_pickle=True)

search_by_title = SearchBooksByTitle("./data/raw_data/LEHABOOKS.csv")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, user_id)

# recsys = Bibliotekar("./data/models/fs_embds.npz",  "./data/models/model.pth")
recsys = Llm()
df = pd.read_csv('./data/raw_data/LEHABOOKS.csv')
second_dataset = pd.read_csv('./data/raw_data/kaggle_second_sem/books_data.csv')
embedding_producer = EmbeddingsProducer()

df = df[df['description'].notna()]

main_model = recsys
extra_model = Llm()

@app.route('/', methods=['GET'])
@app.route('/page/<int:page>', methods=['GET'])
def home(page=1):
    per_page = 10
    total_pages = (len(df) // per_page) + 1
    start = (page - 1) * per_page
    end = start + per_page

    books = df[start:end].to_dict(orient='records')

    for book in books:
        book['info'] = url_for('book_info', title=book['Title'])
        book['recommendations_url'] = url_for('book_recommendations', title=book['Title'])
        book['metric_url'] = url_for('book_metric', title=book['Title'])

    return render_template('index.html', books=books, page=page, total_pages=total_pages)


@app.route('/favorite/page/<int:page>', methods=['GET'])
@app.route('/favorite/page/<int:page>', methods=['GET'])
def favorite_books(page=1):
    favorites = FavoriteBook.query.filter_by(user_id=current_user.id).all()
    favorite_titles = [f.book_title for f in favorites]
    filtered_df = df[df['Title'].isin(favorite_titles)]
    per_page = 10
    total_pages = (len(filtered_df) - 1) // per_page + 1
    start = (page - 1) * per_page
    end = start + per_page
    page_books = filtered_df.iloc[start:end]
    books = []

    for _, row in page_books.iterrows():
        book = row.to_dict()
        book['info'] = url_for('book_info', title=book['Title'])
        book['recommendations_url'] = url_for('book_recommendations', title=book['Title'])
        book['metric_url'] = url_for('book_metric', title=book['Title'])
        books.append(book)

    return render_template('favorite.html', books=books, page=page, total_pages=total_pages)


@app.route('/book/<title>', methods=['GET'])
def book_info(title):
    book = find_book(title)
    book = book[0]

    print(book)
    if not book:
        return "Book not found", 404
    if current_user.is_authenticated:
        tapped_book = TappedBook(book_title=book['Title'], user_id=current_user.id)
        db.session.add(tapped_book)
        db.session.commit()

    return render_template('book_info.html', book=book)


@app.route('/book/rec/<title>', methods=['GET'])
def book_recommendations(title):
    book = find_book(title)
    if not book:
        return "Book not found", 404

    book = book[0]
    if current_user.is_authenticated:

        embs = [find_emb(b.book_title) for b in TappedBook.query.filter_by(user_id=current_user.id).all()]
        recommended_books = main_model.predict(find_emb(title), embs)
    else:
        recommended_books = main_model.predict(find_emb(title))
    recommended_books = [
        {"name": rec_book[0], "url": url_for('book_info', title=rec_book[0]),
         "description": df[df['Title'] == rec_book[0]]['description'].to_string(index=False)}
        for rec_book in recommended_books
    ]
    return render_template('book_recommendations.html', recommended_books=recommended_books)


@app.route('/metric/<title>', methods=['GET'])
@login_required
def book_metric(title):
    book = find_book(title)
    if not book:
        return "Book not found", 404
    book = book[0]
    embs = [find_emb(b) for b in TappedBook.query.filter_by(user_id=current_user.id).all()]
    recommended_books = main_model.predict(find_emb(title), embs)

    recommended_books = [
        {"name": rec_book[0], "url": url_for('book_info', title=rec_book[0]),
         "description": df[df['Title'] == rec_book[0]]['description'].to_string(index=False)}
        for rec_book in recommended_books
    ]

    return render_template('book_metric.html', book=book, recommended_books=recommended_books)


@app.route('/rate', methods=['POST'])
@login_required
def rate_book():
    rating = int(request.form.get('rating'))
    source_book = request.form.get('source_book')
    recommended_book = request.form.get('recommended_book')

    rate = Rating(user_id=current_user.id, rate=rating, source_book=source_book, recommended_book=recommended_book)
    db.session.add(rate)
    db.session.commit()

    return jsonify({"message": "Rating added successfully"})


@app.route('/like', methods=['POST'])
@login_required
def like_book():
    title = request.form.get('title')
    existing = FavoriteBook.query.filter_by(book_title=title, user_id=current_user.id).first()
    if existing is None:
        favorite_book = FavoriteBook(book_title=title, user_id=current_user.id)
        db.session.add(favorite_book)
        db.session.commit()
        return jsonify({"message": "Book added successfully"})
    else:
        return jsonify({"message": "You already added this book"})

@app.route('/unlike', methods=['POST'])
@login_required
def unlike_book():
    title = request.form.get('title')
    existing = FavoriteBook.query.filter_by(book_title=title, user_id=current_user.id).first()
    if existing is None:
        return jsonify({"message": "You already unkile this book"})
    else:
        db.session.delete(existing)
        db.session.commit()
        return jsonify({"message": "Book removed successfully"})

@app.route('/search', methods=['GET'])
def search():
    title = request.args.get('title')  # Получаем параметр title из строки запроса
    recommended_books = search_by_title.closest_title(title, 10)
    recommended_books_links = [
        df[df['Title'] == title] for title in recommended_books
        if not df[df['Title'] == title].empty  # Проверка на существование названия в df
    ]
    return render_template('closest_titles.html', recommended_books=recommended_books_links)


@app.route('/suggest/by_description', methods=['GET'])
def suggest_by_description():
    description = request.args.get('description')  # Получаем параметр title из строки запроса
    emb = embedding_producer.create_embedding(description)
    if current_user.is_authenticated:
        embs = [find_emb(b) for b in TappedBook.query.filter_by(user_id=current_user.id).all()]
        recommended_books = main_model.predict(emb, embs)
    else:
        recommended_books = main_model.predict(emb)
    recommended_books = [
        {"name": rec_book[0], "url": url_for('book_info', title=rec_book[0]),
         "description": df[df['Title'] == rec_book[0]]['description'].to_string(index=False)}
        for rec_book in recommended_books
        if not df[df['Title'] == rec_book[0]].empty
    ]

    return render_template('book_recommendations.html', recommended_books=recommended_books)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/change/llm', methods=['POST'])
def change_llm():
    global main_model
    llm_choice = request.form.get("llm")

    match llm_choice:
        case "our_model":
            flash("Changed successfully")
            main_model = recsys
        case "other_model":
            flash("Changed successfully")
            main_model = extra_model
        case _:
            flash("Unknown model selected")

    return redirect(url_for('home'))

def find_book(title):
    book = df.loc[df['Title'] == title]
    if book.empty:
        book = second_dataset.loc[second_dataset['Title'] == title]

    return book.to_dict(orient='records')


def find_emb(title):
    titles = fs_books["titles"]
    embeddings = fs_books["embeddings"]

    title = str(title)

    try:
        index = np.where(titles == title)[0][0]
        return embeddings[index]
    except IndexError:
        print(f"Title '{title}' not found.")
        return None

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
