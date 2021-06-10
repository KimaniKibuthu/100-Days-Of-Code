import sys
import joblib
import pandas as pd
import sqlalchemy

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

database_path = 'sqlite:///df.db'
model_path = 'model.pkl'



def load_data(database_filepath):
    engine = sqlalchemy.create_engine(database_filepath).connect()
    df = pd.read_sql_table('df', engine)

    category_names = ['related', 'request', 'offer',
                      'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                      'security', 'military', 'child_alone', 'water', 'food', 'shelter',
                      'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
                      'infrastructure_related', 'transport', 'buildings', 'electricity',
                      'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                      'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                      'other_weather', 'direct_report']

    x = df['message'].values
    y = df[category_names].values

    return x, y, category_names


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return tokens


def build_model():
    pipeline = Pipeline([
        ('count_vector', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, x_test, y_test, category_names):
    predictions = model.predict(x_test)

    for key, value in enumerate(category_names):
        print(f'The {value} column')
        print(classification_report(y_test[:, key], predictions[:, key]))
        print('-' * 50)


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath, compress=1)


def main():
    print('Loading data...\n    DATABASE: {}'.format(database_path))
    x, y, category_names = load_data(database_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(x_train, y_train)

    print('Evaluating model...')
    evaluate_model(model, x_test, y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_path))
    save_model(model, model_path)

    print('Trained model saved!')


if __name__ == '__main__':
    main()
