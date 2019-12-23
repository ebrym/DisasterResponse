import sys
import os
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
       INPUT:
           messages_filepath (str): messages csv files path
           categories_filepath (str): categories csv file path
       OUTPUT:
           df: dataframe having messages and cateries details
       DESCRIPTION:
               read messages csv file as messages dataframe and
               categories csv file as categories dataframe
               merge both the dataframes as df applying inner join on ['id'] column
    '''

    df_messages = pd.read_csv(messages_filepath, encoding='latin-1')
    df_categories = pd.read_csv(categories_filepath, encoding='latin-1')

    # merge datasets
    df = pd.merge(df_messages, df_categories, how='inner', on='id')
    return df


def clean_data(df):
    '''
       INPUT:
          The function takes the dataframe as merges from 'load_data' and re-creates a columns from the data
          while dropping the category column.
          arg: dataframe
       OUTPUT:
           df: dataframe having messages and cateries details
    '''

    # create a dataframe of the each of the category type
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories in the dataframe
    row = categories.iloc[0, :]

    # convert the row cells to columns using lambda expression.
    cols = row.apply(lambda x: x[:-2])

    # bind new columns to the `categories` dataframe.
    categories.columns = cols

    # convert category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop('categories', axis=1)  # drop the original categories column from df
    df = pd.concat([df, categories], axis=1)

    df = df.drop_duplicates()  # drop the duplicates

    return df


def save_data(df, database_filename):
    '''
       INPUT:
           cleansed dataframe having messages and their belonging categories details
       OUTPUT:
           database having Messages table
       DESCRIPTION:
           Insert dataframe into sql table<DisasterMessages> in database file to be used as input.
           This checks if the table exist, if it does, it is dropped and recreated. Data is inserted in batches
           based on the chunksize.
    '''
    table = 'DisasterMessages'
    engine = create_engine('sqlite:///{}'.format(database_filename))

    df.to_sql(name=table, con=engine, if_exists='replace', chunksize=10, index=False)


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