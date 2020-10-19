import sys
import os

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split


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
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print(Y.shape)
        sys.exit()

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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