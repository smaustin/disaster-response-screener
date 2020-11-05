import sys
sys.path.append("../models/")
from custom_vectorizer import CustomVectorizer
import json
import plotly
import pandas as pd

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # genre data
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # category data
    df_category = df.loc[:, 'related':'direct_report']
    category_counts = df_category.sum()
    category_names = df_category.columns.str.replace('_', ' ')

    # category with genre data
    df_cat_gen = df.loc[:, 'genre':'direct_report']
    df_cat_gen_counts = df_cat_gen.groupby('genre').sum()

    
    # create visuals
    graphs = [
        # dist by genre
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # dist by category
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                ) 
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        # dist by category grouped by genre
        {
            'data': [
                Bar(
                    x=category_names,
                    y=row,
                    name=genre
                ) for genre, row in df_cat_gen_counts.iterrows()
            ],

            'layout': {
                'barmode': 'stack',
                'title': 'Distribution of Message Categories by Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
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