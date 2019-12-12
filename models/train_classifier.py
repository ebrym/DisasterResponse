import sys
import nltk

import sqlalchemy as create_engine
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import  CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
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
    conn = create_engine('sqlite:///{}'.format(database_filepath))
    # run a query and assign value to a dataframe
    df = pd.read_sql('SELECT * FROM DisasterMessages', conn)

    # prepare modeling data
    X = df['message']

    # for the category values columns
    Y = df.iloc[:, 4:]

    # identify the category columns

    category_columns = Y.coulumns.values

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
    word_list = word_tokenize(text)

    # remove stop words
    tokens = [w for w in word_list if w not in stopwords.words("english")]

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
        The function is to build a pipeline and using gridsearch to training model.
        The pipeline including countVectorizer, TfidfTransformer to process the text and using
        RandomForestClassifier to fit the dataset
    '''

    # create ML pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # specify parameters for grid search
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [5, 7, 9]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        The function is to return the results of prediction on test dataset, including precision socre,
        f1-score and recall score.
        args: model, test dataset and category names
        return: dict - the classification report of category names
    '''

    y_pred = model.predict(X_test)  # prediction
    prediction = pd.DataFrame(y_pred.reshape(-1, 36), columns=category_names)  # transform list to dataframe
    report = dict()
    for i in category_names:
        # iterate the category names and add its classification scores to dictionary
        classification = classification_report(Y_test[i], prediction[i])
        report[i] = classification
    return report


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