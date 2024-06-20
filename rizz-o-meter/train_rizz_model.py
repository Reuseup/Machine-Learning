import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import numpy as np

# Load Dataset
df = pd.read_csv('rizz_dataset.csv')

# Eksplorasi Data (EDA)
print("First 5 rows:")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

print("\nUnique values distribution for 'answer':")
print(df['answer'].value_counts().to_markdown(numalign="left", stralign="left"))

# Hitung jumlah sampel dan fitur
print(f"\nJumlah sampel: {df.shape[0]}")
print(f"Jumlah fitur: {df.shape[1]}")

# Visualisasikan distribusi label 'answer' - Bar Chart
# Hitung jumlah kemunculan setiap label 'answer'
answer_counts = df['answer'].value_counts()

# Buat DataFrame baru dengan kolom 'label' dan 'count'
answer_counts_df = pd.DataFrame({
    'label': answer_counts.index,
    'count': answer_counts.values
})

# Buat bar chart menggunakan Altair
chart = alt.Chart(answer_counts_df, title='Distribusi Label "Rizz"').mark_bar().encode(
    x=alt.X('label:N', axis=alt.Axis(labelAngle=-45), title='Label'),
    y=alt.Y('count:Q', title='Frekuensi'),
    tooltip=['label', 'count']
).interactive()

# Simpan chart dalam format JSON
chart.save('answer_distribution_bar_chart.json')

# Pra-pemrosesan Teks
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = text.lower()  # Ubah ke huruf kecil
    return text

df['question'] = df['question'].astype(str).apply(clean_text)
df['answer'] = df['answer'].astype(str).apply(clean_text)

# Feature Extraction (TF-IDF)
# Combine 'question' and 'answer' for context
df['text'] = df['question'] + ' ' + df['answer']

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['text'])
y = df['answer']  # Labels are the 'answer' column

# Tentukan threshold frekuensi minimum untuk label
min_frequency = 5

# Hitung frekuensi label dan gabungkan label yang jarang muncul
label_counts = Counter(y)
common_labels = [label for label, count in label_counts.items() if count >= min_frequency]

print(f"Label distribution before merging rare labels:\n{label_counts}")
print(f"\nCommon labels:\n{common_labels}")

y = y.apply(lambda label: label if label in common_labels else 'other')

print(f"\nLabel distribution after merging rare labels:\n{Counter(y)}")

# Split Data (Train-Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluasi Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Confusion Matrix
# Get unique labels that appear in both y_test and y_pred
unique_labels = np.unique(np.concatenate((y_test, y_pred)))
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

# Visualisasikan confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Simpan Model dan TF-IDF Vectorizer
with open('rizz_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
