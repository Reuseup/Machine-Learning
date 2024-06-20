{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "46e73827-278b-4143-919c-9d016858e334",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import re\n",
    "from flask import Flask, request, jsonify\n",
    "import pickle\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# Muat model dan TF-IDF Vectorizer\n",
    "with open('rizz_model.pkl', 'rb') as f:\n",
    "    model = pickle.load(f)\n",
    "with open('tfidf_vectorizer.pkl', 'rb') as f:\n",
    "    tfidf_vectorizer = pickle.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "14c747dd-8049-4a26-9f21-6d10f5d98c2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_text(text):\n",
    "    text = re.sub(r'[^\\w\\s]', '', text)\n",
    "    text = text.lower()\n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5fef05f8-0b0c-40cd-beeb-2f38310c9e31",
   "metadata": {},
   "outputs": [],
   "source": [
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/rizz', methods=['POST'])\n",
    "def predict_rizz():\n",
    "    data = request.get_json()\n",
    "    text = data['text']\n",
    "\n",
    "    # Pra-pemrosesan teks\n",
    "    text = clean_text(text)\n",
    "\n",
    "    # Transformasi teks menggunakan TF-IDF Vectorizer\n",
    "    text_tfidf = tfidf_vectorizer.transform([text])\n",
    "\n",
    "    # Prediksi\n",
    "    prediction = model.predict(text_tfidf)[0]\n",
    "\n",
    "    # Kembalikan hasil prediksi sebagai JSON\n",
    "    return jsonify({'rizz_level': prediction})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f49c3d43-bf42-4ccb-b5de-21440a636983",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
