
import sys
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet', 'stopwords'])
import sqlalchemy as sqla
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import pickle

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    '''
          INPUT:
              database_filepath (str): messages csv files path
          OUTPUT:
              X: actual message columns
              Y: corresponding category values for the message
              category_columns : category columns
          DESCRIPTION:
                 Reads data from sqlite db initially created and selects the columns for the
                 training and testing model.
    '''
    # connect to the database
    conn = sqla.create_engine('sqlite:///{}'.format(database_filepath))
    # run a query and assign value to a dataframe
    df = pd.read_sql('SELECT * FROM DisasterMessages', conn)

    # prepare modeling data
    X, Y = df['message'], df.iloc[:, 4:]

    # mapping extra values to `1`
    Y['related'] = Y['related'].map(lambda x: 1 if x == 2 else x)

    # reassign the new columns to dataframe
    category_columns = Y.columns.values

    return X, Y, category_columns


def tokenize(text):
    '''
        The function is to process the sentence, token the words and lower it.
        arg: str text
        return:list
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    words = word_tokenize(text)

    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]

    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words


def build_model():
    '''
        The function is to build a pipeline and using gridsearch to training model.
        The pipeline including countVectorizer, TfidfTransformer to process the text and using
        RandomForestClassifier to fit the dataset
    '''

    # create ML pipeline

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(
                             OneVsRestClassifier(LinearSVC())))])

    # hyper-parameter grid
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # create model
    model = GridSearchCV(estimator=pipeline,
                         param_grid=parameters,
                         verbose=3,
                         cv=3)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        The function is to return the results of prediction on test dataset, including precision socre,
        f1-score and recall score.
        args: model, test dataset and category names
        return: dict - the classification report of category names
    '''

    y_preds = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test.values, y_preds, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_preds)))



def save_model(model, model_filepath):
    '''
       INPUT:
           model (str): trained model
           model_filepath (str): pickle file path to save the model
       OUTPUT:
       DESCRIPTION:
               save the model passed as the path given as input
       '''

    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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