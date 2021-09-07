import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    """
    Loads two .csv datasets and merges them together as one
    
    Parameters:
    messages_filepath: csv file of the dataset messages
    categories_filepath: csv file of the dataset category
    
    returns:
    df: contains the merged dataset into a pandas datafram
    """
    dataImport = pd.read_csv(messages_filepath)
    catImport = pd.read_csv(categories_filepath)
    df = pd.merge(dataImport, catImport, on='id', how='left')
    
    return df

def clean_data(df):
    
    """
    With the dataframe we will than clean up the dataset to be used for our           training and testing.
    
    Parameters:
    df: The merged dataframe
    
    returns:
    df: contains the merged dataset into a pandas datafram that has been cleaned
    """
    
    category = df["categories"].str.split(";", expand=True)
    row = category.iloc[[0]]
    catCol = row.applymap(lambda z: z[:-2]).iloc[0,:].tolist()
    
    category.columns = catCol
    
    for col in category:
        category[col] = category[col].str[-1]
        category[col] = category[col].astype(int)
        
    df = df.drop("categories", axis=1)
    df = pd.concat([df, category], axis=1)
    
    df = df.drop_duplicates()
    
    return df
    
    


def save_data(df, database_filename):
    """
    The dataframe will be saved to be used for our training.
    
    Parameters:
    df: The merged dataframe
    database_filename: The name of what our database will be called
    
    returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disasterapp', engine, index=False, if_exists = 'replace')
    print(df.dtypes)
    print(type(df))


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