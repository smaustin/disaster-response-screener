import sys
import os
from time import time
from custom_vectorizer import CustomVectorizer

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
import pickle
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def load_data(database_filepath):
    """Load the database file and return features, target variables and categories.
    
    Args:
    database_filepath: string. Full filepath to database db file.

    Returns:
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


def build_model():
    """Use Pipeline and GridSearchCV to build a model for message processing 
    using multilabel classification.
    
    Returns:
    search: object. A grid search object with optimized hyperparameters.
    """
    
    # TODO: Model speed of performance could be improved using LogisticRegression
    # however some training targets only contain one class. May be able to 
    # construct a custom classifier to account for these target values.

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

    search = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)

    return search


def evaluate_model(model, X_test, Y_test, category_names):
    """Use the model to predict using a test set, then print evaluation results to console.
    
    Args:
    model: object. Model object created using build_model.
    X_test: numpy.ndarray. Array of text messages to predict labels
    Y_test: numpy.ndarray. Array of true labels for X_test
    category_name: 

    Returns:
    None
    """
    # predict on test data
    Y_pred = model.predict(X_test)

    # loop over the indexes of first row
    for idx, _ in enumerate(Y_pred[0,:]):
        
        # pass each column into metric 
        accuracy = accuracy_score(Y_test[:, idx], Y_pred[:, idx])
        precision, recall, fscore, support = precision_recall_fscore_support(Y_test[:, idx], Y_pred[:, idx], average='binary', zero_division=0)
        print(category_names[idx])
        # print(f'    Accuracy: {accuracy:.4f}', f'    Precision: {precision:.4f}', f'    Recall: {recall:.4f}')
        print(f'\tAccuracy: {accuracy:.4f}', f'\tPrecision: {precision:.4f}', f'\tRecall: {recall:.4f}', f'\tF1 Score: {fscore:.4f}')
        print()


def save_model(model, model_filepath):
    """Save the model to a pickle file.
    
    Args:
    model: object. Trained model object
    model_filepath: string. Filepath to save file.
    
    Returns:
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        t0 = time()
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print("Loading Time:", round(time() - t0, 3), "s")
        # print(type(Y_train))
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