import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    loads data into a dataframe from 2 csvs and merges them to one

    Parameters:
        messages_filepath: path to the messages csv file.
        categories_filepath: path to the categoies csv file.

    Returns:
        df: a merged dataframe of the messages and categories files

    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    # merging the datasets on the id field
    df = pd.merge(messages_df, categories_df, how = 'left', on = 'id')

    return df


def clean_data(df):
    """
    performs transformations of the merged dataframe to clean it up

    Parameters:
        df: merged dataframe with the messages and categories.

    Retruns:
        df: a cleaned-up dataframe where each category is a column with a numeric value only.
    """
    # splits the semi-colon separated categories into its own columns
    split_categories = df['categories'].str.split(';', expand = True)

    # get the first row  of the categories df and create a category columns list with the first character values
    row = split_categories.iloc[0]
    category_colnames = row.transform(lambda x: x[:-2]).tolist()

    split_categories.columns = category_colnames
    for column in split_categories:

        # set each value to be the last character of the string
        split_categories[column] = split_categories[column].transform(lambda x: x[-1:])
        # convert column from string to numeric
        split_categories[column] = pd.to_numeric(split_categories[column])

    # drop the categories column from `df`
    df.drop('categories', axis = 1, inplace = True)

    # concat the original dataframe with the new split_categories dataframe
    df = pd.concat([df, split_categories], axis = 1)

    # drop any duplicates in the dataframe
    df.drop_duplicates(inplace = True)

    #remove rows with related = 2 as they are not informative for the model
    df = df[df['related'] != 2]

    return df


def save_data(df, database_filename):
    """
    saves a dataframe to an sqllite database

    Parameters:
        df: a cleaned dataframe
        database_filename: a string of the name of database to save the data

    Returns:
        None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')

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