from flask import Flask, render_template, request

app = Flask(__name__)

# Lista kategorii
CATEGORIES = ["Aborcja", "Eutanazja", "Prawo do posiadania broni", "Podatki", "Edukacja", "Imigracja", "Ochrona środowiska", "Opieka zdrowotna", "Równość płci", "Edukacja seksualna", "Kara śmierci", "Podatek progresywny", "Kara śmierci", "Religia w szkole", "Małżeństwa jednopłciowe", "System emerytalny", "Legalizacja marihuany", "Cenzura w internecie", "Służba wojskowa", "Uchodźcy"]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        opinion = request.form.get('opinion')
        category = request.form.get('category')
        
        # Tu można dodać logikę do zapisywania opinii w bazie danych lub dalszej analizy
        print(f"Kategoria: {category}, Opinia: {opinion}")

    return render_template('index.html', categories=CATEGORIES)

if __name__ == '__main__':
    app.run(debug=True)