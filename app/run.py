import json
import plotly
import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import re

import operator
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):

    # Normalize text
    filter_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    words = word_tokenize(filter_text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words




# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # category extractions
    category = list(df)[4:]
    category_counts = [np.sum(df[column]) for column in category]
    
    categories = df.iloc[:,4:]
    categories_mean = categories.mean().sort_values(ascending=False)[1:6]
    categories_names = list(categories_mean.index)

    


    # message representation by category
    '''
    Try to get most words used in messages
    '''
    words_occurrences=[]                              
    
    #Tokenize the message collumn
    for text in df['message'].values:
        tokenized_ = tokenize(text)
        words_occurrences.extend(tokenized_)

    
    #create a counter to words
    word_dictionary = Counter(words_occurrences)     
    
    sorted_word_dictionary = dict(sorted(word_dictionary.items(),
                                         key=operator.itemgetter(1),
                                         reverse=True))   
                                         
    top_word_count, top_10_words =0, {}

    #looping through the word dictionary and selecting the top 10 occurrence
    for k,v in sorted_word_dictionary.items():
        top_10_words[k]=v
        top_word_count+=1
        if top_word_count==10:
            break
    words=list(top_10_words.keys())
  
    count_props=100*np.array(list(top_10_words.values()))/df.shape[0]
    #


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Message by Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=count_props
                )
            ],

            'layout': {
                'title': 'Top 10 words representation(%)',
                'yaxis': {
                    'title': '% Occurrence',
                    'automargin': True
                },
                'xaxis': {
                    'title': 'Words',
                    'automargin': True
                }
            }
        },
        {
            'data': [
                    Bar(
                        x=category,
                        y=category_counts
                        )
                    ],
              'layout': {
              'title': 'Message by categories',
              'yaxis': {
              'title': "Count"
              },
              'xaxis': {
              'title': "Category"
              }
              }
        },
        {
              'data': [
                       Bar(
                           x=categories_names,
                           y=categories_mean
                           )
                       ],
              'layout': {
              'title': 'Top 5 categories',
              'yaxis': {
              'title': "Count"
              },
              'xaxis': {
              'title': "Categories"
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