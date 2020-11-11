import sys
import os
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load the data files into a single Pandas DataFrame.
    
    Args:
    messages_filepath: string. Filepath to messages CSV file.
    categories_filepath: string. Filepath to categories CSV file.

    Returns:
    df: Pandas DataFrame. DataFrame of merged files
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
    
    # ensure boolean values, replace values greater than 1 with 1
    categories[categories > 1] = 1

    # replace categories column in df with new category columns
    df = df.drop(columns=['categories']).join(categories)

    return df

def clean_data(df):
    """Remove duplicates from DataFrame and return.
    
    Args:
    df: Pandas DataFrame. DataFrame of dataset.

    Returns:
    df: Pandas DataFrame. DataFrame with duplicates removed.
    """

    return df.drop_duplicates()


def save_data(df, database_filepath):
    """Save the clean dataset into a sqlite database.
    
    Args:
    df: Pandas DataFrame. DataFrame of dataset.
    database_filepath: string. File path with name of database to save DataFrame.

    Returns:
    None
    """  
    # get the table name from the database_filepath
    file_name = os.path.splitext(database_filepath)[0]
    table_name = os.path.basename(file_name)
    print(table_name)

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