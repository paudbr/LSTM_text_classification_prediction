import random
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib

def dividir_letras(letra, tamaño_cachito):
    versos = letra.splitlines()
    cachitos = []
    for verso in versos:
        palabras = verso.split()
        inicio = 0
        while inicio < len(palabras):
            fin = inicio + tamaño_cachito
            cachito = " ".join(palabras[inicio:fin])
            if cachito:
                cachitos.append(cachito)
            inicio = fin
    random.shuffle(cachitos)
    return cachitos

def preprocess(text):
    pattern = r'[^a-zA-Z0-9\s]'
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    clean_text = re.sub(pattern, '', text.lower())
    tokens = word_tokenize(clean_text)
    filtered_token = [word for word in tokens if word not in stop_words]
    stemmed_token = [ps.stem(word) for word in filtered_token]
    return ' '.join(stemmed_token)

def train_and_predict(lyric, model=None, vectorizer=None):
    df = pd.read_csv('data/songs_modif.csv', header=0)
    df["Album"] = df["Album"].replace({"folklore": "folkmore", "evermore": "folkmore"})
    nuevo_df = df.assign(Lyrics=df['Lyrics'].apply(dividir_letras, tamaño_cachito=35)).explode('Lyrics')
    nuevo_df.reset_index(drop=True, inplace=True)

    albumes_deseados = ['folkmore', 'The Tortured Poets Department', 'Speak Now (Taylor’s Version)', 'Red (Taylor’s Version)', 'Midnights (3am Edition)', 'Lover', 'Fearless (Taylor’s Version)', 'reputation']

    df_filtrado = nuevo_df[nuevo_df['Album'].isin(albumes_deseados)]

    if vectorizer is None:
        vectorizer = CountVectorizer(max_features=5000)
        vectorizer.fit(df_filtrado['Lyrics'])

    trf = vectorizer.transform(df_filtrado['Lyrics']).toarray()
    X = trf
    y = df_filtrado['Album']

    le = LabelEncoder()
    y_label = le.fit_transform(y)

    if model is None:
        model = MultinomialNB()

    X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.2)
    model.fit(X_train, y_train)

    letra_preprocesada = preprocess(lyric)
    letra_vectorizada = vectorizer.transform([letra_preprocesada]).toarray()
    prediccion = model.predict(letra_vectorizada)

    album_names = df_filtrado['Album'].unique()
    label_to_album = {i: album for i, album in enumerate(album_names)}
    nombre_album = label_to_album[prediccion[0]]

    # Guardar el modelo y el vectorizador después del reentrenamiento
    joblib.dump(model, 'modelo_clasificacion.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return nombre_album

