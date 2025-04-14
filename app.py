import sys
import os
from flask import Flask, request, jsonify, render_template, url_for, redirect
import pandas as pd
from flask import request, render_template, url_for
import logging
import flask_login

from flask_sqlalchemy import SQLAlchemy
from models import User


SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Book-RecSys', 'src'))
sys.path.append(SRC_PATH)

from models.modules import BookDescriptionEmbeddingSimilarity
from models.modules import RecommendUsingGraph
from models.modules import EmbeddingsProducer
from models.modules import SearchBooksByTitle
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,  # Логи будут выводиться на уровне INFO
    handlers=[
        logging.FileHandler("flask_requests.log"),  # Запись логов в файл
        logging.StreamHandler()  # Вывод логов в консоль
    ]
)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:2006yura@localhost/book_recsys_users'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


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


if __name__ == '__main__':
    # app.run(host='192.168.0.105')
    app.run(host="0.0.0.0", port=3000, debug=True)
    