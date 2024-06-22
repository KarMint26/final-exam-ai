from flask import Flask, render_template, request
from newspaper import Article
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lxml_html_clean import clean_html 

app = Flask(__name__)

# Download data NLTK (jika belum)
nltk.download('punkt')

def summarize_article(url):
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()

    # Tokenisasi kalimat
    sentences = sent_tokenize(article.text)

    # Menghitung kemiripan antar kalimat (TextRank)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Memilih kalimat-kalimat penting
    scores = []
    for i in range(len(sentences)):
        score = sum(similarity_matrix[i])
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)

    # Ambil beberapa kalimat teratas sebagai ringkasan
    summary_sentences = [sentences[i] for i, _ in scores[:5]]  # Ganti 5 dengan jumlah kalimat yang diinginkan
    summary = ' '.join(summary_sentences)

    return summary

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/summary', methods=['POST'])
def summary():
    url = request.form['url']
    summary = summarize_article(url)
    return render_template('result.html', summary=summary, original_url=url)

if __name__ == '__main__':
    app.run(debug=True)
