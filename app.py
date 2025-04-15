import sys
import os
from flask import Flask, request, jsonify, render_template, url_for, redirect, flash
import pandas as pd
from flask import request, render_template, url_for
import logging
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from my_models import db
from my_models.User import User
from my_models.FavoriteBook import FavoriteBook
from my_models.TappedBook import TappedBook

from flask_sqlalchemy import SQLAlchemy

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Book-RecSys', 'src'))
sys.path.append(SRC_PATH)

from models.modules import BookDescriptionEmbeddingSimilarity
from models.modules import RecommendUsingGraph
from models.modules import EmbeddingsProducer
from models.modules import SearchBooksByTitle

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:2006yura@localhost/book_recsys_users'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Import User model after db initialization

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, user_id)

# Чтение данных из CSV при старте приложения
df = pd.read_csv('./data/raw_data/LEHABOOKS.csv')
metric = pd.read_csv('./data/test_data/rating.csv')
recsys_with_emb = BookDescriptionEmbeddingSimilarity(
    "./data/embeddings/books_embeddings_dataset.npy")
recsys = RecommendUsingGraph("./data/graphs/book_graph.json",recsys_with_emb)
embedding_producer = EmbeddingsProducer()
search_books_by_title = SearchBooksByTitle('./data/raw_data/LEHABOOKS.csv')

df = df[df['Description'].notna()]

# Главная страница с пагинацией
@app.route('/', methods=['GET'])
@app.route('/page/<int:page>', methods=['GET'])
def home(page=1):
    per_page = 10  # Количество книг на одной странице
    total_pages = (len(df) // per_page) + 1  # Всего страниц
    start = (page - 1) * per_page
    end = start + per_page

    # Выбираем нужные книги для текущей страницы
    books = df[start:end].to_dict(orient='records')

    # Генерируем URL для каждой книги
    for book in books:
        book['info'] = url_for('book_info', title=book['Title'])
        book['recommendations_url'] = url_for('book_recommendations', title=book['Title'])
        book['metric_url'] = url_for('book_metric', title=book['Title'])

    return render_template('index.html', books=books, page=page, total_pages=total_pages)

# Детальная страница книги
@app.route('/book/<title>', methods=['GET'])
def book_info(title):
    book = df[df['Title'] == title].to_dict(orient='records')
    book = book[0]
    if not book:
        return "Book not found", 404
    if current_user.is_authenticated:
        tapped_book = TappedBook(book_title=book['Title'], user_id=current_user.id)
        db.session.add(tapped_book)
    return render_template('book_info.html', book=book)

@app.route('/book/rec/<title>', methods=['GET'])
def book_recommendations(title):
    # Ищем книгу по названию
    book = df[df['Title'] == title].to_dict(orient='records')

    if not book:
        return "Book not found", 404

    book = book[0]  # Берём первую найденную запись
    recommended_books = recsys.find_closest_books(title, n=20)
    recommended_books = [
        {"name": rec_book[0], "url": url_for('book_info', title=rec_book[0]),
         "description": df[df['Title'] == rec_book[0]]['Description'].to_string(index=False)}
        for rec_book in recommended_books
    ]
    return render_template('book_recommendations.html', recommended_books=recommended_books)

@app.route('/metric/<title>', methods=['GET'])
@login_required
def book_metric(title):
    book = df[df['Title'] == title].to_dict(orient='records')

    if not book:
        return "Book not found", 404
    book = book[0]
    recommended_books = recsys.find_closest_books(title, n=100)
    recommended_books = [
        {"name": rec_book[0], "url": url_for('book_info', title=rec_book[0]),
         "description":  df[df['Title'] == rec_book[0]]['Description'].to_string(index=False)}
        for rec_book in recommended_books
    ]

    return render_template('book_metric.html', book=book, recommended_books=recommended_books)

@app.route('/rate', methods=['POST'])
@login_required
def rate_book():
    global metric
    rating = int(request.form.get('rating'))
    source_book = request.form.get('source_book')
    recommended_book = request.form.get('recommended_book')

    new_row = pd.DataFrame([{
        'Title': source_book,
        'Title_for_rate': recommended_book,
        'rate': rating
    }])

    # Объединение новой строки с существующим DataFrame
    metric = pd.concat([metric, new_row], ignore_index=True)

    # Сохранение изменений в CSV
    metric.to_csv('./data/test_data/rating.csv', index=False)
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
        flash("Book added successfully", "success")
    else:
        flash("You already liked this book", "info")
    return redirect(url_for('book_info', title=title))


@app.route('/search', methods=['GET'])
def search():
    title = request.args.get('title')  # Получаем параметр title из строки запроса
    recommended_books = search_books_by_title.closest_title(title, 10)
    recommended_books_links = [
        df[df['Title'] == title] for title in recommended_books
        if not df[df['Title'] == title].empty  # Проверка на существование названия в df
    ]
    return render_template('closest_titles.html', recommended_books=recommended_books_links)

@app.route('/suggest/by_description', methods=['GET'])
def suggest_by_description():
    description = request.args.get('description')  # Получаем параметр title из строки запроса
    emb = embedding_producer.create_embedding(description)
    recommended_books = recsys_with_emb.recommend_by_embedding(emb.T, n=20)
    recommended_books = [
        {"name": rec_book[0], "url": url_for('book_info', title=rec_book[0]),
         "description": df[df['Title'] == rec_book[0]]['Description'].to_string(index=False)}
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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
    