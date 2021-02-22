import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    loads data into a dataframe from an sqllite database and splits the dataframe into messages df and the categories df

    Parameters:
        database_filepath: the filepath to the sqllite database
    Returns:
        X: a dataframe with the messages column
        Y: a dataframe with only the categories columns
        category_names: a list of all category column names

    """
     # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)

    # Create X and Y datasets
    X = df['message']
    Y = df.drop(df.columns[0:4], axis = 1)

    # Create list containing all category names
    category_names = list(Y.columns.values)

    return X, Y, category_names


def tokenize(text):
    """
    Normalize, lemmatize, and tokenize  text string

    Parameters:
        text: a string containing a sentence to be proceesed

    Returns:
        stemmed: list of strings containing normalized and stemmed word tokens

    """
    # Convert text to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize words
    tokens = word_tokenize(text)

    #remove stop words
    text = [t for t in text if t not in stopwords.words("english")]

    # Lemmatise word tokens
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).strip()
        clean_tokens.append(clean_token)

    return clean_tokens

def build_model():
    """
    Build a machine learning(ML) pipeline

    Parameters:
        None

    Returns:
        cv: Gridsearchcv object that transforms the data, creates a
        model and gets the best model parameters.

    """
    # Create a pipeline
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Create parameters dictionary
    parameters = {
        'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'text_pipeline__vect__max_features': (None, 5000, 10000),
        'text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 10)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Gets testing dataset accuracy, precision, recall and F1 score for fitted model

    Parameters:
        model: Fitted model object.
        X_test: dataframe containing test features dataset.
        Y_test: dataframe containing test target dataset.
        category_names: list of the category names.

    Returns:
        None

    """
    # get predicted labels for test dataset
    Y_pred = model.predict(X_test)

    #create an empty metrics list
    metrics_list = []

    # get evaluation metrics for each of the labels
    for i in range(len(category_names)):
        # convert Y_test to an array
        Y_test = np.array(Y_test)

        accuracy = accuracy_score(Y_test[:, i], Y_pred[:, i])
        precision = precision_score(Y_test[:, i],Y_pred[:, i], average=None)
        recall = recall_score(Y_test[:, i], Y_pred[:, i], average=None)
        f1 = f1_score(Y_test[:, i], Y_pred[:, i], average=None)

        metrics_list.append([accuracy, precision, recall, f1])

    # Create dataframe containing metrics and print results
    metrics = np.array(metrics_list)
    metrics_df = pd.DataFrame(data = metrics, columns = ['Accuracy', 'Precision', 'Recall', 'F1'], index = category_names, )
    print(metrics_df)

def save_model(model, model_filepath):
    """
    Save the fitted model

    Parameters:
        model: fitted model object.
        model_filepath: filepath string for where fitted model should be saved

    Returns:
        None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

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