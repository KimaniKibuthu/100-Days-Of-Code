import sqlalchemy
import pandas as pd

messages_path = 'disaster_messages.csv'
categories_path = 'disaster_categories.csv'
database_path = 'sqlite:///df.db'


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_names = row.str.split('-', expand=True)[0].values
    categories.columns = category_names

    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(str)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    engine = sqlalchemy.create_engine(database_filename)
    df.to_sql('df', engine, index=False)


def main():

    df = load_data(messages_path, categories_path)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_path))
    save_data(df, database_path)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()
