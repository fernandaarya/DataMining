import pandas as pd
import re
from collections import defaultdict
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Membaca file CSV
df = pd.read_csv('D:/Kelompok2_22230008.csv')
tweets = df['Isi Tweet'].dropna().tolist()

# Inisialisasi stopword remover
factory = StopWordRemoverFactory()
stop_words = set(factory.get_stop_words())

# Fungsi preprocessing: hapus URL, karakter non-alfabet, ubah ke huruf kecil, hapus stopwords
def preprocess(text):
    text = re.sub(r"http\S+", "", text)                  # Hapus URL
    text = re.sub(r"[^a-zA-Z\s]", "", text)              # Hapus karakter selain huruf dan spasi
    text = text.lower()                                  # Ubah ke huruf kecil
    tokens = text.split()
    return ' '.join([w for w in tokens if w not in stop_words])  # Hapus stopwords

# Preprocessing semua tweet
tweets_cleaned = [preprocess(t) for t in tweets]

# Ekstraksi frasa dengan panjang tertentu dari setiap tweet
def extract_phrases(text, min_words=2, max_words=4):
    words = text.split()
    result_phrases = []
    for size in range(min_words, max_words + 1):
        for i in range(len(words) - size + 1):
            phrase = ' '.join(words[i:i+size])
            result_phrases.append(phrase)
    return result_phrases

# Membangun indeks frasa ke tweet
phrase_to_docs = defaultdict(set)
for idx, tweet in enumerate(tweets_cleaned):
    phrases = extract_phrases(tweet)
    for phrase in phrases:
        phrase_to_docs[phrase].add(idx)

# Ambang batas minimum tweet dalam 1 cluster
min_docs_per_cluster = 4

# Menyaring frasa umum yang muncul di >= min_docs_per_cluster tweet
common_phrase = {
    phrase: docs for phrase, docs in phrase_to_docs.items()
    if len(docs) >= min_docs_per_cluster
}

# Membentuk cluster berdasarkan frasa umum
clusters = defaultdict(set)
for phrase, docs in common_phrase.items():
    clusters[f"Cluster: '{phrase}'"] = docs

# Output hasil clustering
if not clusters:
    print("Tidak ada cluster yang memenuhi ambang batas !!!!")
else:
    for i, (cluster_name, doc_ids) in enumerate(clusters.items(), start=1):
        print(f"\n{i}. {cluster_name} (total: {len(doc_ids)} tweet)")
        for doc_id in doc_ids:
            print(f" - {tweets[doc_id]}")
