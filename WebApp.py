#conda activate base, cd C:\Users\Sikma\Jupyter\KompasPolityczny, python WebApp.py

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

#załadowanie datasetów (w CSV) zawierających wypowiedzi, punktację i embeddingi i stworzenie z nich dataframemów
def CSVload_datasets_embedded(directory):
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        df = pd.read_csv(filepath, encoding="utf-8")
        df["embedding"] = df["embedding"].apply(lambda x: np.array(list(map(float, x.split(",")))))

        dataset_name = f'df_{filename.replace("_embeddings.csv", "")}'
        
        globals()[dataset_name] = df
CSVload_datasets_embedded("datasets_embeddings")
print(globals().keys())

#załadowanie modelu
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

#funkcja do tworzenia embeddingów z tekstu
def get_embedding(text):
    return model.encode(text)

#funkcja realizująca porównanie wypowiedzi użytkownika (embeddingu) z wypowiedziami (embeddingami) z odpowiedniego datasetu i zwracająca ostateczną punktację na skali -1.0 do 1.0
def get_similarity_upgraded(user_input, df_dataset):
    # Wektor wypowiedzi użytkownika
    user_embedding = get_embedding(user_input)
    
    # Oblicz podobieństwo dla każdej wypowiedzi w DataFrame
    df_dataset["similarity"] = df_dataset["embedding"].apply(lambda emb: cosine_similarity([user_embedding], [emb])[0][0])
    # Posortuj według podobieństwa malejąco
    df_sorted = df_dataset.sort_values(by="similarity", ascending=False)

    top_n = 5  # Weź 5 najbliższych wypowiedzi
    top_similar = df_sorted.head(top_n)

    if top_similar.iloc[0]["similarity"] > 0.9:
        avg_score = top_similar.iloc[0]["score"]
    else:
        avg_score = top_similar["score"].mean()
        lower_bound = max(avg_score - 0.8, -1.0)
        upper_bound = min(avg_score + 0.8, 1.0)

        df_dataset_filtered = df_dataset[(df_dataset["score"] >= lower_bound) & (df_dataset["score"] <= upper_bound)]
        df_dataset_filtered = df_dataset_filtered.sort_values(by="similarity", ascending=False)
        top_similar = df_dataset_filtered.head(top_n)
        avg_score = top_similar["score"].mean()
        
    return avg_score, top_similar[["similarity", "score", "statement"]].values.tolist()


CATEGORIES = ['aborcja',
              'armia_ue',
              'bron',
              'cpk',
              'dochodowy',
              'euro',
              'eutanazja',
              'imigranci',
              'invitro',
              'kara_smierci',
              'katastralny',
              'osiemset',
              'samochody',
              'sluzba_wojskowa',
              'ue',
              'wdowia',
              'zus']
CATEGORIES2 = ['1','2','3','4']

@app.route('/kompas', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        opinion = request.form.get('opinion')
        category = request.form.get('category')

        print(f"Kategoria: {category}, Opinia: {opinion}")

    return render_template('index.html', categories=CATEGORIES)

if __name__ == '__main__':
    app.run(debug=True)