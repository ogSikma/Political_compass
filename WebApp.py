#conda activate base, cd C:\Users\Sikma\Jupyter\KompasPolityczny, python WebApp.py

from flask import Flask, render_template, request, session, redirect
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
app.secret_key = 'klucz'

#załadowanie datasetów (w CSV) zawierających wypowiedzi, punktację i embeddingi i stworzenie z nich dataframemów
def CSVload_datasets_embedded(directory):
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        df = pd.read_csv(filepath, encoding="utf-8")
        df["embedding"] = df["embedding"].apply(lambda x: np.array(list(map(float, x.split(",")))))

        dataset_name = f'df_{filename.replace("_embeddings.csv", "")}'
        
        globals()[dataset_name] = df
CSVload_datasets_embedded("datasets_embeddings")

#załadowanie modelu
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("max_seq_length domyślnie:", model.max_seq_length)
model.max_seq_length = 512
print("max_seq_length po zmianie na 512:", model.max_seq_length)

#funkcja do tworzenia embeddingów z tekstu
def get_embedding(text):
    return model.encode(text)

#funkcja realizująca porównanie wypowiedzi użytkownika (embeddingu) z wypowiedziami (embeddingami) z odpowiedniego datasetu i zwracająca ostateczną punktację na skali -1.0 do 1.0
def get_similarity_upgraded(user_input, df_dataset):
    user_embedding = get_embedding(user_input)
    
    df_dataset["similarity"] = df_dataset["embedding"].apply(lambda emb: cosine_similarity([user_embedding], [emb])[0][0])
    df_sorted = df_dataset.sort_values(by="similarity", ascending=False)

    top_n = 5
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

def get_similarity_politician(user_input, df_dataset):
    user_embedding = get_embedding(user_input)
    
    df_dataset["similarity"] = df_dataset["embedding"].apply(lambda emb: cosine_similarity([user_embedding], [emb])[0][0])
    df_sorted = df_dataset.sort_values(by="similarity", ascending=False)

    top_n = 3
    top_similar = df_sorted.head(top_n)

    return top_similar[["similarity", "politician", "political_club", "utterance"]].values.tolist()

pacyfizm_militaryzm = ['bron', 'obronnosc', 'sluzba_wojskowa']
nacjonalizm_kosmopolityzm = ['obronnosc', 'sluzba_wojskowa', 'armia_ue', 'euro', 'cpk', 'ue', 'imigranci']
ekologia_industrializm = ['samochody', 'cpk']
eurofederalizm_eurosceptyzm = ['euro', 'armia_ue', 'ue', 'samochody']
progresywizm_tradycjonalizm = ['aborcja', 'eutanazja', 'invitro', 'kara_smierci', 'bron']
solidaryzm_liberalizm = ['osiemset', 'zus', 'wdowia']
interwencjonizm_leseferyzm =  ['zus', 'dochodowy', 'katastralny']

class User:
    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(data):
        user = User()
        user.__dict__.update(data)
        return user

    def __init__(self):
        self.pacyfizm_militaryzm_score = 0
        self.pacyfizm_militaryzm_answers = 0
        
        self.nacjonalizm_kosmopolityzm_score = 0
        self.nacjonalizm_kosmopolityzm_answers = 0

        self.ekologia_industrializm_score = 0
        self.ekologia_industrializm_answers = 0

        self.eurofederalizm_eurosceptyzm_score = 0        
        self.eurofederalizm_eurosceptyzm_answers = 0

        self.progresywizm_tradycjonalizm_score = 0        
        self.progresywizm_tradycjonalizm_answers = 0

        self.solidaryzm_liberalizm_score = 0        
        self.solidaryzm_liberalizm_answers = 0

        self.interwencjonizm_leseferyzm_score = 0        
        self.interwencjonizm_leseferyzm_answers = 0

        self.nolan_gospodarka_score = 0
        self.nolan_gospodarka_answers = 0

        self.nolan_obyczajowe_score = 0
        self.nolan_obyczajowe_answers = 0

    def add_score_to_compass(self, user_statement, chosen_topic, chosen_dataset):
        user_score, similiar_results = get_similarity_upgraded(user_statement, chosen_dataset)

        if chosen_topic in solidaryzm_liberalizm or chosen_topic in interwencjonizm_leseferyzm:
            if chosen_topic in solidaryzm_liberalizm:
                self.solidaryzm_liberalizm_score += user_score
                self.solidaryzm_liberalizm_answers += 1   
                
            if chosen_topic in interwencjonizm_leseferyzm:
                self.interwencjonizm_leseferyzm_score += user_score
                self.interwencjonizm_leseferyzm_answers += 1   
                
            self.nolan_gospodarka_score += user_score
            self.nolan_gospodarka_answers += 1
            
        else:  
            if chosen_topic in pacyfizm_militaryzm:       
                self.pacyfizm_militaryzm_score += user_score
                self.pacyfizm_militaryzm_answers += 1   
    
            if chosen_topic in nacjonalizm_kosmopolityzm:        
                self.nacjonalizm_kosmopolityzm_score += user_score
                self.nacjonalizm_kosmopolityzm_answers += 1   
                
            if chosen_topic in ekologia_industrializm:        
                self.ekologia_industrializm_score += user_score
                self.ekologia_industrializm_answers += 1   
                
            if chosen_topic in eurofederalizm_eurosceptyzm:       
                self.eurofederalizm_eurosceptyzm_score += user_score
                self.eurofederalizm_eurosceptyzm_answers += 1   
                
            if chosen_topic in progresywizm_tradycjonalizm:
                self.progresywizm_tradycjonalizm_score += user_score
                self.progresywizm_tradycjonalizm_answers += 1
                
            self.nolan_obyczajowe_score += user_score
            self.nolan_obyczajowe_answers += 1


    def display_scores(self):
        if self.pacyfizm_militaryzm_answers != 0:
            print(f'Wartość dla pacyfizm-militaryzm wynosi {self.pacyfizm_militaryzm_score/self.pacyfizm_militaryzm_answers}')
        if self.nacjonalizm_kosmopolityzm_answers != 0:
            print(f'Wartość dla nacjonalizm-kosmopolityzm wynosi {self.nacjonalizm_kosmopolityzm_score/self.nacjonalizm_kosmopolityzm_answers}')
        if self.ekologia_industrializm_answers != 0:
            print(f'Wartość dla ekologia-industrializm wynosi {self.ekologia_industrializm_score/self.ekologia_industrializm_answers}')
        if self.eurofederalizm_eurosceptyzm_answers != 0:
            print(f'Wartość dla eurofederalizm-eurosceptyzm wynosi {self.eurofederalizm_eurosceptyzm_score/self.eurofederalizm_eurosceptyzm_answers}')
        if self.progresywizm_tradycjonalizm_answers != 0:
            print(f'Wartość dla progresywizm-tradycjonalizm wynosi {self.progresywizm_tradycjonalizm_score/self.progresywizm_tradycjonalizm_answers}')
        if self.solidaryzm_liberalizm_answers != 0:
            print(f'Wartość dla socjalizm-liberalizm wynosi {self.solidaryzm_liberalizm_score/self.solidaryzm_liberalizm_answers}')
        if self.interwencjonizm_leseferyzm_answers != 0:
            print(f'Wartość dla regulacjonizm-leseferyzm wynosi {self.interwencjonizm_leseferyzm_score/self.interwencjonizm_leseferyzm_answers}')
        
        if self.nolan_obyczajowe_answers != 0:
            print(f'Wartość dla diagramu Nolana konserwatyzm-liberalizm wynosi {self.nolan_obyczajowe_score/self.nolan_obyczajowe_answers}')
        if self.nolan_gospodarka_answers != 0:
            print(f'Wartość dla diagramu Nolana socjalizm-wolny rynek wynosi {self.nolan_gospodarka_score/self.nolan_gospodarka_answers}')

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
        user = User.from_dict(session['user'])    

        opinion = request.form.get('opinion')

        category = request.form.get('category')
        chosen_dataset_name = f'df_dataset_{category}'
        chosen_dataset = globals()[chosen_dataset_name]

        user.add_score_to_compass(opinion, category, chosen_dataset)

        session['categories'] = [x for x in session['categories'] if x != category]
        session['user'] = user.to_dict()

        print(f"Kategoria: {category}, Opinia: {opinion}")
        print(user.display_scores())

        session['answer_count'] += 1
        show_button = session['answer_count'] >= 4

        return render_template('index.html', categories=session['categories'], show_button=show_button)
    
    session['answer_count'] = 0
    session['categories'] = CATEGORIES.copy()
    session['user'] = User().to_dict()
    return render_template('index.html', categories=session['categories'], show_button=False)

@app.route('/wyniki')
def wyniki():
    user = User.from_dict(session['user'])    

    scores = {
        'pacyfizm_militaryzm': user.pacyfizm_militaryzm_score / user.pacyfizm_militaryzm_answers if user.pacyfizm_militaryzm_answers else None,
        'nacjonalizm_kosmopolityzm': user.nacjonalizm_kosmopolityzm_score / user.nacjonalizm_kosmopolityzm_answers if user.nacjonalizm_kosmopolityzm_answers else None,
        'ekologia_industrializm': user.ekologia_industrializm_score / user.ekologia_industrializm_answers if user.ekologia_industrializm_answers else None,
        'eurofederalizm_eurosceptyzm': user.eurofederalizm_eurosceptyzm_score / user.eurofederalizm_eurosceptyzm_answers if user.eurofederalizm_eurosceptyzm_answers else None,
        'progresywizm_tradycjonalizm': user.progresywizm_tradycjonalizm_score / user.progresywizm_tradycjonalizm_answers if user.progresywizm_tradycjonalizm_answers else None,
        'solidaryzm_liberalizm': user.solidaryzm_liberalizm_score / user.solidaryzm_liberalizm_answers if user.solidaryzm_liberalizm_answers else None,
        'regulacjonizm_leseferyzm': user.interwencjonizm_leseferyzm_score / user.interwencjonizm_leseferyzm_answers if user.interwencjonizm_leseferyzm_answers else None,
    }

    nolan = {  
        'konserwatyzm_liberalizm': user.nolan_obyczajowe_score / user.nolan_obyczajowe_answers if user.nolan_obyczajowe_answers else None,
        'socjalizm_wolny-rynek': user.nolan_gospodarka_score / user.nolan_gospodarka_answers if user.nolan_gospodarka_answers else None
    }

    return render_template('wyniki.html', scores = scores, nolan = nolan)


if __name__ == '__main__':
    app.run(debug=True)