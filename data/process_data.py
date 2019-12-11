import sys
import pandas as pd

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
    pass


def save_data(df, database_filename):
    pass  


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