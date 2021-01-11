import json
import plotly
import pandas as pd
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('wordnet')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(sentence):
    """
    Normalize, tokenize and lemmatize sentence string

    Parameters:
        sentence: a string containing a sentence to be proceesed

    Returns:
        clean_tokens: list of strings containing normalized and lemmatized word tokens

    """
    tokens = word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens

# load datadatadata
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(x.upper() for x in genre_counts.index)
    categories_corr_df = df.drop(df.columns[0:4], axis = 1).corr()

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    y=genre_names,
                    x=genre_counts,
                    orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Genre"
                },
                'xaxis': {
                    'title': "Count"
                },
            }
        },
        {
            'data': [
                Heatmap(
                    z=categories_corr_df,
                    x=categories_corr_df.columns,
                    y=categories_corr_df.index
                )
            ],

            'layout': {
                'title': 'Correlation matirx of the categories',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Category"
                },
                "autosize": False,
                "width": 1000,
                "height": 1000,
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph--{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()