#conda activate base, cd C:\Users\Sikma\Jupyter\KompasPolityczny, python WebApp.py

from flask import Flask, render_template, request, session, redirect, jsonify
import pandas as pd
import numpy as np
import math
import os
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity
from flask_session import Session


app = Flask(__name__)
app.secret_key = 'klucz'

app.config['SESSION_TYPE'] = 'filesystem'  # Możesz też użyć 'redis', 'sqlalchemy', itp.
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')  # katalog na pliki sesji
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True  # dodatkowe zabezpieczenie
os.makedirs('flask_session', exist_ok=True)
Session(app)

#załadowanie datasetów (w CSV) zawierających wypowiedzi, punktację i embeddingi i stworzenie z nich dataframemów
def CSVload_datasets_embedded(directory):
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        df = pd.read_csv(filepath, encoding="utf-8")
        df["embedding"] = df["embedding"].apply(lambda x: np.array(list(map(float, x.split(",")))))

        dataset_name = f'df_{filename.replace("_embeddings.csv", "")}'
        
        globals()[dataset_name] = df
CSVload_datasets_embedded("datasets_embeddings_polbert")
# POLITYCY --------------------------------------
df_politycy = pd.read_csv('politycy_embeddings_polbert.csv', encoding='utf-8', sep=';')
df_politycy["embedding"] = df_politycy["embedding"].apply(lambda x: np.array(list(map(float, x.split(',')))))

#załadowanie modelu
#model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
#model = SentenceTransformer('intfloat/multilingual-e5-small')
#model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#model = SentenceTransformer('sdadas/st-polish-paraphrase-from-distilroberta')
#model = SentenceTransformer('BAAI/bge-m3')

word_embedding_model = models.Transformer(
    './polbert-fine-tune/PolBERT_trained2',
    max_seq_length=128
)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

print(model.get_sentence_embedding_dimension())
print("max_seq_length domyślnie:", model.max_seq_length)
#model.max_seq_length = 512
#print("max_seq_length po zmianie na 512:", model.max_seq_length)

#funkcja do tworzenia embeddingów z tekstu
def get_embedding(text):
    return model.encode(text)

#funkcja wzmacniająca wynik dla wyższych wartości
def amplify(x, scale=1):
    return math.tanh(x * scale) / math.tanh(scale)

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
        avg_score = amplify(avg_score)
        
    return avg_score, top_similar[["similarity", "score", "statement"]].values.tolist()

def get_similarity_politicians(user_input, df_dataset):
    user_embedding = get_embedding(user_input)
    df_dataset["similarity"] = df_dataset["embedding"].apply(lambda emb: cosine_similarity([user_embedding], [emb])[0][0])
    max_similarity_per_politician = df_dataset.groupby("politician")["similarity"].max()

    return max_similarity_per_politician.to_dict()

pacyfizm_militaryzm = ['bron', 'obronnosc', 'sluzba_wojskowa']
kosmopolityzm_nacjonalizm = ['obronnosc', 'sluzba_wojskowa', 'armia_ue', 'euro', 'cpk', 'ue', 'imigranci']
ekologia_industrializm = ['samochody', 'cpk']
eurofederalizm_eurosceptyzm = ['euro', 'armia_ue', 'ue', 'samochody']
progresywizm_tradycjonalizm = ['aborcja', 'eutanazja', 'invitro', 'kara_smierci', 'bron']
solidaryzm_liberalizm = ['osiemset', 'zus', 'wdowia']
interwencjonizm_leseferyzm =  ['zus', 'dochodowy', 'katastralny']

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class User:
    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(data):
        user = User()
        user.__dict__.update(data)
        return user

    def __init__(self):
        # POLITYCY --------------------------------------
        self.politicians_dict = {name: [0, club] for name, club in zip(df_politycy['politician'], df_politycy['political_club'])}

        self.pacyfizm_militaryzm_score = 0
        self.pacyfizm_militaryzm_answers = 0
        
        self.kosmopolityzm_nacjonalizm_score = 0
        self.kosmopolityzm_nacjonalizm_answers = 0

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
    
            if chosen_topic in kosmopolityzm_nacjonalizm:        
                self.kosmopolityzm_nacjonalizm_score += user_score
                self.kosmopolityzm_nacjonalizm_answers += 1   
                
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

    def add_politicians_similarity(self, user_statement):
        new_similarities = get_similarity_politicians(user_statement, df_politycy)

        for pol, score in new_similarities.items():
            self.politicians_dict[pol][0] += score

        #sorted_politicians = sorted(self.politicians_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_politicians = sorted(self.politicians_dict.items(),key=lambda x: x[1][0], reverse=True)

        print("Top 3 polityków wg podobieństwa:")
        for pol, (score, club) in sorted_politicians[:3]:
            print(f"{pol} ({club}): {score:.4f}")

    def display_scores(self):
        if self.pacyfizm_militaryzm_answers != 0:
            print(f'Wartość dla pacyfizm-militaryzm wynosi {self.pacyfizm_militaryzm_score/self.pacyfizm_militaryzm_answers}')
        if self.kosmopolityzm_nacjonalizm_answers != 0:
            print(f'Wartość dla nacjonalizm-kosmopolityzm wynosi {self.kosmopolityzm_nacjonalizm_score/self.kosmopolityzm_nacjonalizm_answers}')
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
        # POLITYCY --------------------------------------
        user.add_politicians_similarity(opinion)

        session['categories'] = [x for x in session['categories'] if x != category]
        session['user'] = user.to_dict()

        print(f"Kategoria: {category}, Opinia: {opinion}")
        print(user.display_scores())

        session['answer_count'] += 1
        show_button = session['answer_count'] >= 4

        return render_template('index.html', categories=session['categories'], show_button=show_button)
    
    if 'user' not in session:
        session['answer_count'] = 0
        session['categories'] = CATEGORIES.copy()
        session['user'] = User().to_dict()

    return render_template('index.html', categories=session['categories'], show_button=False)

@app.route('/wyniki')
def wyniki():
    user = User.from_dict(session['user'])    

    scores = {
        'pacyfizm_militaryzm': user.pacyfizm_militaryzm_score / user.pacyfizm_militaryzm_answers if user.pacyfizm_militaryzm_answers else None,
        'kosmopolityzm_nacjonalizm': user.kosmopolityzm_nacjonalizm_score / user.kosmopolityzm_nacjonalizm_answers if user.kosmopolityzm_nacjonalizm_answers else None,
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

    parties_points = [
        ("Platforma Obywatelska", (0.1, -0.4)),
        ("Inicjatywa Polska", (-0.4, -0.9)),
        ("Prawo i Sprawiedliwość", (-0.3, 0.7)),
        ("Polska 2050", (0, -0.1)),
        ("Polskie Stronnictwo Ludowe", (0.1, 0.3)),
        ("Nowa Lewica", (-0.6, -0.8)),
        ("Polska Partia Socjalistyczna", (-0.7, -0.9)),
        ("Razem", (-0.8, -0.7)),
        ("Nowa Nadzieja", (0.8, 0.7)),
        ("Ruch Narodowy", (0.4, 0.8)),
        ("Konfederacja Korony Polskiej", (0.6, 1))
    ]

    reference_point = (nolan['socjalizm_wolny-rynek'], nolan['konserwatyzm_liberalizm'])
    similar_parties = sorted(parties_points, key=lambda party: distance(party[1], reference_point))
    print(similar_parties)

    # POLITYCY --------------------------------------
    politicians = sorted(user.politicians_dict.items(), key=lambda x: x[1][0], reverse=True)[:10]
    names = [i[0] for i in politicians]
    similarities = [(i[1][0]/session['answer_count'])*100 for i in politicians]
    parties = [i[1][1] for i in politicians]
    labels = [f'{name} ({party})' for name, party in zip(names,parties)]
    #create_barplot(nazwiska,podobienstwa)

    return render_template('wyniki.html', scores=scores, nolan=nolan, labels=labels, percentages=similarities, parties=parties, top3_parties = similar_parties[:3])



if __name__ == '__main__':
    app.run(debug=True)