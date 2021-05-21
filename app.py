from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle
from sklearn.externals import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("data/Youtube01-Psy.csv")
    corpus = df['CONTENT']

    #data cleaning
    corpus_clean = []
    for txt in corpus:
        word_tokens = [word.lower() for word in word_tokenize(txt)]
        clean_words = [word for word in word_tokens if (not word in set(stopwords.words('english')) and word.isalpha())]
        lemmmatizer = WordNetLemmatizer()
        clean_words = [lemmmatizer.lemmatize(word.lower()) for word in clean_words]
        corpus_clean.append(' '.join(clean_words))

    df_data = df[['CONTENT', 'CLASS']]

    # Features and Labels
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS

    #TFIDF
    corpus = df_x
    X = TfidfVectorizer(min_df=1, stop_words='english').fit_transform(corpus)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    LR_model = LogisticRegression(solver='liblinear')
    LR_model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score as ac

    pred = LR_model.predict(X_test)
    ac(y_test, pred)

    # Save Model
    joblib.dump(LR_model, 'model.pkl')
    print("Model dumped!")

    # ytb_model = open('spam_model.pkl', 'rb')
    clf = joblib.load('model.pkl')
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = X.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)