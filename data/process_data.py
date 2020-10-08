import sys
import os
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load the data files into a single Pandas DataFrame.
    
    Args:
    messages_filepath: string. Full filepath to messages CSV file.
    categories_filepath: string. Full filepath to categories CSV file.

    Return:
    df: Pandas DataFrame of merged files
    """

    # load datasets from files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.join(categories.set_index('id'), on = 'id')
    
    # create a dataframe of the individual category columns
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe and use for column names
    row = categories.iloc[0]
    categories.columns = row.apply(lambda x: x[:-2])

    # convert category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # replace categories column in df with new category columns
    df = df.drop(columns=['categories']).join(categories)

    return df

def clean_data(df):
    """Remove duplicates from DataFrame (df) and return."""

    return df.drop_duplicates()


def save_data(df, database_filepath):
    """Save the clean dataset into a sqlite database.
    
    Args:
    df: Pandas DataFrame. DataFrame of dataset.
    database_filepath: string. File path with name of database to save DataFrame to.
    """  
    # get the table name from the database_filepath
    table_name = os.path.splitext(database_filepath)[0]

    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql(table_name, con=engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()