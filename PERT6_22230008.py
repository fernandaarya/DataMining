import pandas as pd 
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

df = pd.read_csv('D:/Kelompok2_22230008.csv')
tweets = df['Isi Tweet'].dropna().tolist()

factory = StopWordRemoverFactory()
stop_words = set(factory.get_stop_words())

def preprocess(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text) 
    text = text.lower()
    tokens = text.split()
    return ' '.join([w for w in tokens if w not in stop_words])

tweets_cleaned = [preprocess(t) for t in tweets]

vectorizer = TfidfVectorizer(max_df=0.8, min_df=2)
X = vectorizer.fit_transform(tweets_cleaned)

k = 5
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)

for i in range(k):
    print(f"\n Cluster {i+1}")
    indices = [j for j, label in enumerate(model.labels_) if label == i]
    for idx in indices[:5]:  # tampilkan 5 tweet pertama dari cluster ini
        print(f" - {tweets[idx]}")
