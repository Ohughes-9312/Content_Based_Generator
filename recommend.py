import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

app = Flask(__name__)

# Load dataset from CSV file
def load_data():
    try:
        dataset = pd.read_csv('movies.csv')  # Load dataset from movies.csv
        return dataset
    except Exception as e:
        print("Error loading data:", e)
        return pd.DataFrame()  # Return empty DataFrame if load fails

# Preprocess text: remove special characters and convert to lowercase
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\\W', ' ', text)  # Replace non-word characters with space
    return text.lower()

# Vectorize text using TF-IDF to measure word importance
def vectorize_text(data):
    vectorizer = TfidfVectorizer(stop_words='english')  # Exclude common stop words
    tfidf_matrix = vectorizer.fit_transform(data['description'].apply(preprocess_text))
    return vectorizer, tfidf_matrix

# Compute cosine similarity between user input and dataset descriptions
def compute_similarity(user_input, vectorizer, tfidf_matrix):
    user_tfidf = vectorizer.transform([user_input])  # Vectorize user input
    return cosine_similarity(user_tfidf, tfidf_matrix).flatten()

# Filter dataset based on genre and minimum rating
def filter_data(data, genre=None, min_rating=None):
    filtered_data = data
    if genre:
        filtered_data = filtered_data[filtered_data['genre'].str.contains(genre, case=False, na=False)]
    if min_rating:
        filtered_data = filtered_data[filtered_data['rating'] >= min_rating]
    return filtered_data

# Generate top N movie recommendations based on similarity scores
def get_recommendations(user_input, data, vectorizer, tfidf_matrix, genre=None, min_rating=None, N=5):
    filtered_data = filter_data(data, genre, min_rating)
    if filtered_data.empty:
        return pd.DataFrame()
    cosine_similarities = compute_similarity(user_input, vectorizer, tfidf_matrix)
    N = min(N, len(filtered_data))
    if N == 0:
        return pd.DataFrame()
    top_n_indices = cosine_similarities.argsort()[-N:][::-1]
    return filtered_data.iloc[top_n_indices]

# Flask route to handle form submission and render recommendations
@app.route('/', methods=['GET', 'POST'])
def index():
    data = load_data()
    recommendations = None
    if not data.empty:
        vectorizer, tfidf_matrix = vectorize_text(data)
        if request.method == 'POST':
            user_input = request.form.get('user_input', '')
            genre = request.form.get('genre', '')
            min_rating = request.form.get('min_rating', type=float, default=None)
            N = request.form.get('N', type=int, default=5)
            if user_input:
                recommendations = get_recommendations(user_input, data, vectorizer, tfidf_matrix, genre, min_rating, N)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
