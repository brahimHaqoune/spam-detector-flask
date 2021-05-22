from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("data/Youtube01-Psy.csv")

    # Features and Labels
    df_x = df['CONTENT']
    df_y = df.CLASS

    #TFIDF
    cv = TfidfVectorizer(min_df=1, stop_words='english')
    X = cv.fit_transform(df_x)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

    #Logistic Regression
    from sklearn.linear_model import LogisticRegression

    LR_model = LogisticRegression(solver='liblinear')
    LR_model.fit(X_train, y_train)

    # Save Model
    pickle.dump(LR_model, open('model.pkl', 'wb'))

    clf = pickle.load(open('model.pkl', 'rb'))
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)