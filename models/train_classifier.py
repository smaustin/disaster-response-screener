import sys
import os
from time import time

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data(database_filepath):
    """Load the database file and return features, target variables and categories.
    
    Args:
    database_filepath: string. Full filepath to database db file.

    Return:
    X: array. features
    Y: array. target variables
    target_names: array of category names
    """
    # get the database name from the database_filepath
    database_name = os.path.splitext(database_filepath)[0]

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql(database_name,con=engine)
    X = df.message.values
    Y = df.loc[:, 'related':'direct_report'].values
    target_names = df.loc[:, 'related':'direct_report'].columns.values

    return (X, Y, target_names)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Need to lemmatize and use built in tokenizer and case normalizer from CountVectorizer(),
# Create a custom Class inheriting from CountVectorizer()
def lemmatize(tokens):
    """Helper function to lemmatize text during tokenization."""   
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize both nouns and verbs so two passes, 
        clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok), pos='v')
        clean_tokens.append(clean_tok)

    return clean_tokens

class CustomVectorizer(CountVectorizer):
    """Custom vectorizer that inherits from CountVectorizer.
    Allows for lemmatization to happening during CountVectorizer tokenization 
    and utilizes built-in preprocessing for case etc."""
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(lemmatize(tokenize(doc)))


def build_model():
    """Build a model to process text."""
    # classifier must 1) support sparse matrix - returned from TfidfTransformer,
    # 2) implement predict_proba method - for use with MultiOutputClassifier,
    # and 3) handle targets with only one binary label (all 0 or 1)
    pipeline = Pipeline([
        ('vect', CustomVectorizer()), # text tokenization 
        ('tfidf', TfidfTransformer()),  # feature normalization
        ('clf', MultiOutputClassifier(RandomForestClassifier())) # classifier
    ])
    # RandomForestClassifier(min_samples_split=2, n_estimators=100)
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        # 'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        t0 = time()
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print("Loading Time:", round(time() - t0, 3), "s")
        # print(Y.shape)
        # sys.exit()

        print('Building model...')
        t0 = time()
        model = build_model()
        print("Building time:", round(time() - t0, 3), "s")

        print('Training model...')
        t0 = time()
        model.fit(X_train, Y_train)
        print("Training time:", round(time() - t0, 3), "s")
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()