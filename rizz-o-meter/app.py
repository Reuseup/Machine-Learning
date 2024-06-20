from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)

# Muat model dan TF-IDF Vectorizer (hanya sekali saat aplikasi dimulai)
try:
    with open('rizz_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Model atau TF-IDF Vectorizer tidak ditemukan. Pastikan Anda telah melatih dan menyimpannya.")

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Fungsi untuk menggabungkan question dan answer
def combine_text(question, answer):
    question = clean_text(question)
    answer = clean_text(answer)
    return question + ' ' + answer

# Rute untuk halaman utama (/) dengan formulir input
@app.route('/', methods=['GET'])
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rizz-O-Meter</title>
    </head>
    <body>
        <h1>Rizz-O-Meter</h1>
        <form method="POST" action="/rizz">
            <textarea name="text" placeholder="Masukkan teks di sini" rows="4" cols="50"></textarea><br>
            <input type="submit" value="Prediksi Rizz">
        </form>
    </body>
    </html>
    '''

# Rute untuk memproses prediksi (/rizz)
@app.route('/rizz', methods=['POST'])
def predict_rizz():
    try:
        text = request.form.get('text')  # Ambil teks dari formulir

        # Validasi input teks
        if not text:
            return jsonify({'error': 'Teks tidak boleh kosong'}), 400

        # Pra-pemrosesan teks
        combined_text = combine_text(text, text)  # Menggabungkan teks dengan placeholder

        # Debug: Tampilkan teks yang digabungkan
        print(f"Combined text for prediction: {combined_text}")

        # Transformasi teks menggunakan TF-IDF Vectorizer
        text_tfidf = tfidf_vectorizer.transform([combined_text])

        # Prediksi
        prediction = model.predict(text_tfidf)[0]

        # Debug: Tampilkan prediksi
        print(f"Predicted label: {prediction}")

        return jsonify({'rizz_level': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=7000)
